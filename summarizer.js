/**
 * CharSummaryception — Summarization Pipeline
 *
 * Batch summarization with per-character perception filtering and layered
 * compression. Adapted from Summaryception's approach (by Lodactio, AGPL-3.0)
 * with key changes:
 *   - Per-character: each character gets their own summary in their Data Bank file.
 *   - Perception-aware: only messages the character witnessed are included.
 *   - Speaker names preserved (not "Player:"/"Assistant:").
 *   - Markdown storage with layer/range attributes (Vector Storage compatible).
 *   - No ghosting — messages stay visible for Vector Storage to index.
 */

import {
    parseMemories,
    serializeMemories,
    buildPassageForWitness,
    hasPartialPerception,
    stripNonDiegetic,
    cleanSummarizerOutput,
    isRepetitiveGarbage,
    substitutePromptTemplate,
    getTimestamp,
    countMemories,
} from './lib.js';
import {
    sendSummarizerRequest,
    ConnectionError,
} from './connection.js';

const LOG_PREFIX = '[CharSummaryception]';

const RETRY_CONFIG = {
    maxRetries: 5,
    baseDelay: 2000,
    maxDelay: 60000,
    backoffMultiplier: 2,
    retryableStatuses: [429, 500, 502, 503, 504],
};

// Per-call timeout: activity resets on every chunk, hard cap is absolute.
const CALL_TIMEOUT_MS = {
    activity: 120_000,
    hardCap:  600_000,
};

function parseRetryAfter(error) {
    try {
        const retryAfter = error?.response?.headers?.['retry-after']
            ?? error?.retryAfter
            ?? error?.data?.retry_after;
        // `0` is a valid "retry now" value, so check explicitly.
        if (retryAfter === undefined || retryAfter === null || retryAfter === '') return null;
        const seconds = Number(retryAfter);
        if (!isNaN(seconds)) return seconds * 1000;
        const date = new Date(retryAfter);
        if (!isNaN(date.getTime())) return Math.max(0, date.getTime() - Date.now());
    } catch (e) { /* ignore */ }
    return null;
}

function isRetryableError(error) {
    if (error?.name === 'AbortError') return false;

    const msg = (error?.message || error?.toString() || '').toLowerCase();

    // Context-size errors are never retryable — the input won't shrink on retry.
    if (msg.includes('context length')) return false;
    if (msg.includes('context window')) return false;
    if (msg.includes('maximum context')) return false;
    if (msg.includes('too many tokens')) return false;
    if (msg.includes('token limit')) return false;
    if (msg.includes('prompt is too long')) return false;
    if (msg.includes('context_length_exceeded')) return false;
    if (msg.includes('request too large')) return false;
    if (msg.includes('input too long')) return false;
    if (msg.includes('reduce the length')) return false;

    if (error?.name === 'ConnectionError' && typeof error.retryable === 'boolean') return error.retryable;
    if (error?.name === 'TypeError' && msg.includes('fetch')) return true;
    const status = error?.status || error?.response?.status || error?.statusCode;
    if (status && RETRY_CONFIG.retryableStatuses.includes(status)) return true;
    if (msg.includes('rate limit')) return true;
    if (msg.includes('too many requests')) return true;
    if (msg.includes('server error')) return true;
    if (msg.includes('timeout')) return true;
    if (msg.includes('timed out')) return true;
    if (msg.includes('econnreset')) return true;
    if (msg.includes('econnrefused')) return true;
    if (msg.includes('enotfound')) return true;
    if (msg.includes('epipe')) return true;
    if (msg.includes('network')) return true;
    if (msg.includes('overloaded')) return true;
    if (msg.includes('capacity')) return true;
    if (msg.includes('temporarily unavailable')) return true;
    if (msg.includes('service unavailable')) return true;
    if (msg.includes('bad gateway')) return true;
    if (msg.includes('socket hang up')) return true;
    return false;
}

/**
 * Snapshot all prompt-manager toggles (for isolation during default-mode calls).
 */
function snapshotPromptToggles() {
    const snapshot = new Map();
    try {
        const ctx = SillyTavern.getContext();
        const pm = ctx.promptManager;
        if (!pm) return snapshot;
        const collection = pm.getPromptCollection();
        if (!collection?.collection) return snapshot;
        const orderList = pm.getPromptOrderEntries();
        if (!orderList) return snapshot;
        for (const entry of collection.collection) {
            for (const orderEntry of orderList) {
                if (orderEntry.identifier === entry.identifier) {
                    snapshot.set(entry.identifier, orderEntry.enabled);
                }
            }
        }
    } catch (e) { /* ignore */ }
    return snapshot;
}

function disableAllPromptToggles() {
    try {
        const ctx = SillyTavern.getContext();
        const pm = ctx.promptManager;
        if (!pm) return;
        const orderList = pm.getPromptOrderEntries();
        if (!orderList) return;
        for (const entry of orderList) {
            if (entry.enabled) entry.enabled = false;
        }
    } catch (e) { /* ignore */ }
}

function restorePromptToggles(snapshot) {
    if (!snapshot || snapshot.size === 0) return;
    try {
        const ctx = SillyTavern.getContext();
        const pm = ctx.promptManager;
        if (!pm) return;
        const orderList = pm.getPromptOrderEntries();
        if (!orderList) return;
        for (const entry of orderList) {
            if (snapshot.has(entry.identifier)) {
                entry.enabled = snapshot.get(entry.identifier);
            }
        }
    } catch (e) { /* ignore */ }
}

/**
 * Call the summarizer LLM with retry/backoff and prompt-toggle isolation.
 *
 * @param {object} deps Injected dependencies (settings, getUserName, etc.)
 * @param {string} systemPrompt
 * @param {string} userPrompt
 * @param {AbortSignal} [externalSignal] Optional external abort signal.
 *        When aborted, the call returns '' as soon as possible.
 * @param {object} [diagnosticsOut] Receives the underlying Error on failure
 *        ({ lastError: null }). Cleared on success.
 * @returns {Promise<string>} The cleaned summary text, or '' on failure.
 */
export async function callSummarizer(deps, systemPrompt, userPrompt, externalSignal, diagnosticsOut) {
    const { settings } = deps;
    const s = settings();

    const isDefaultMode = !s.connectionSource || s.connectionSource === 'default';
    const snapshot = isDefaultMode ? snapshotPromptToggles() : null;
    if (isDefaultMode) disableAllPromptToggles();

    // If the external signal is already aborted, bail out immediately.
    if (externalSignal?.aborted) {
        if (isDefaultMode && snapshot) restorePromptToggles(snapshot);
        return '';
    }

    let lastError = null;
    const recordError = (err) => {
        lastError = err;
        if (diagnosticsOut) diagnosticsOut.lastError = err ?? null;
    };
    const clearError = () => {
        if (diagnosticsOut) diagnosticsOut.lastError = null;
    };

    try {
        for (let attempt = 0; attempt <= RETRY_CONFIG.maxRetries; attempt++) {
            if (externalSignal?.aborted) return '';

            // Per-attempt controller: a timeout aborts the in-flight request
            // so a stalled call doesn't keep burning tokens server-side.
            // External aborts propagate into every attempt.
            const attemptController = new AbortController();
            const onExternalAbort = () => attemptController.abort();
            if (externalSignal) {
                externalSignal.addEventListener('abort', onExternalAbort, { once: true });
            }

            try {
                if (attempt > 0) {
                    deps.debug?.(`Retry attempt ${attempt}/${RETRY_CONFIG.maxRetries}`);
                }

                // Activity timer resets on each chunk via onProgress; hard cap never resets.
                // `settled` guards against post-settlement onProgress calls leaking a timer.
                let activityTimer;
                let hardCapTimer;
                let onAbort;
                let outerReject;
                let settled = false;

                const activityMessage = `Request timed out — no activity for ${CALL_TIMEOUT_MS.activity / 1000}s`;
                const hardCapMessage = `Request timed out — hard cap of ${CALL_TIMEOUT_MS.hardCap / 1000}s exceeded`;

                const onProgress = () => {
                    if (settled) return;
                    clearTimeout(activityTimer);
                    activityTimer = setTimeout(() => {
                        // Settle the race as a timeout FIRST, then stop the
                        // stalled request (order matters — abort() would
                        // otherwise reject the race as 'Aborted').
                        outerReject(new Error(activityMessage));
                        attemptController.abort();
                    }, CALL_TIMEOUT_MS.activity);
                };

                const timeoutPromise = new Promise((_, reject) => {
                    outerReject = reject;
                    activityTimer = setTimeout(() => {
                        outerReject(new Error(activityMessage));
                        attemptController.abort();
                    }, CALL_TIMEOUT_MS.activity);
                    hardCapTimer = setTimeout(() => {
                        outerReject(new Error(hardCapMessage));
                        attemptController.abort();
                    }, CALL_TIMEOUT_MS.hardCap);
                    onAbort = () => {
                        clearTimeout(activityTimer);
                        clearTimeout(hardCapTimer);
                        reject(new Error('Aborted'));
                    };
                    attemptController.signal.addEventListener('abort', onAbort, { once: true });
                });

                let result;
                try {
                    result = await Promise.race([
                        sendSummarizerRequest(s, systemPrompt, userPrompt, attemptController.signal, onProgress),
                        timeoutPromise,
                    ]);
                } finally {
                    settled = true;
                    clearTimeout(activityTimer);
                    clearTimeout(hardCapTimer);
                    if (onAbort) attemptController.signal.removeEventListener('abort', onAbort);
                }

                let trimmed = (result || '').trim();
                trimmed = cleanSummarizerOutput(trimmed, s.stripPatterns || []);

                if (!trimmed) throw new Error('Empty response from summarizer');

                if (isRepetitiveGarbage(trimmed)) {
                    deps.debug?.('Detected repetitive output, discarding');
                    return '';
                }

                clearError();
                return trimmed;

            } catch (err) {
                recordError(err);
                if (externalSignal?.aborted || err.message === 'Aborted') return '';
                if (!isRetryableError(err)) break;
                if (attempt >= RETRY_CONFIG.maxRetries) break;

                let delay;
                const retryAfterMs = parseRetryAfter(err);
                if (retryAfterMs != null) {
                    delay = Math.min(retryAfterMs, RETRY_CONFIG.maxDelay);
                } else {
                    const exponentialDelay = RETRY_CONFIG.baseDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, attempt);
                    const jitter = Math.random() * RETRY_CONFIG.baseDelay;
                    delay = Math.min(exponentialDelay + jitter, RETRY_CONFIG.maxDelay);
                }

                deps.debug?.(`Attempt ${attempt + 1} failed. Retrying in ${(delay / 1000).toFixed(1)}s...`, err.message || err);

                await new Promise((resolve) => {
                    let retryTimer;
                    const onRetryAbort = () => { clearTimeout(retryTimer); resolve(); };
                    retryTimer = setTimeout(() => {
                        externalSignal?.removeEventListener('abort', onRetryAbort);
                        resolve();
                    }, delay);
                    externalSignal?.addEventListener('abort', onRetryAbort, { once: true });
                });
            } finally {
                if (externalSignal) {
                    externalSignal.removeEventListener('abort', onExternalAbort);
                }
            }
        }

        console.error(LOG_PREFIX, 'Summarization failed after all retries:', lastError);
        return '';
    } finally {
        if (isDefaultMode && snapshot) {
            restorePromptToggles(snapshot);
        }
    }
}

/**
 * Default per-layer prompts.
 * Layer 0 = fresh turn summaries. Layer 1+ = compression.
 * Layer 2+ inherits Layer 1's prompts unless overridden in settings.layerPrompts[N].
 */
export const DEFAULT_LAYER_PROMPTS = {
    0: {
        systemPrompt:
            'You are the memory keeper for the ongoing story of {{charName}}. Output only bullet points — no preamble, no commentary, no markdown headers.',
        userPrompt:
            `You are the memory keeper for {{charName}}. Summarize what {{charName}} would actually remember from the passage below.

PASSAGE (what just happened):
{{passage}}

METHOD — two passes:
1. Silently scan the passage. Find the moments worth remembering once this stretch is over — the words and actions that actually changed something for {{charName}} or the people around them.
2. Then write one bullet per moment, in the order it happened.

RULES:
1. Third person, past tense. Name the actors: "{{charName}} plugged in the keyboard" — not "I plugged in" or "the keyboard was plugged in".
2. Recap what happened, don't label it. "{{charName}} argued with the innkeeper over the bill, refused to pay, and got thrown into the street" is good. "{{charName}} had trouble at the inn" is useless.
3. Skip mood-only beats unless the mood itself is load-bearing (a threat, a confession, a shift in allegiance).
4. Write as many bullets as the passage warrants — typically 2 to 6. Prefer fewer sharp bullets over many vague ones.
5. First bullet is a cast tag: "- [Cast: A, B — short gist]".

Output ONLY the bullets. No headers, no commentary.`,
    },
    1: {
        systemPrompt:
            'You are a memory compressor for long-form roleplay. Merge related bullets, drop duplicates, and tighten prose without losing any plot-affecting fact. Output only bullet points — no preamble, no commentary, no markdown headers.',
        userPrompt:
            `You are compressing {{charName}}'s memory bullets. The bullets below come from earlier scene summaries. Merge related beats, drop duplicates, and tighten wording — but preserve every distinct fact, decision, relationship change, and revealed secret.

PRIOR CONTEXT (already-compressed context — do NOT repeat):
{{priorContext}}

BULLETS TO MERGE (compress these):
{{passage}}

COMPRESSION METHOD — two passes:
1. Silently cluster bullets by the event, actor, or arc they describe. Mark near-duplicates and any beats that a stronger bullet already covers.
2. Then emit one merged bullet per cluster.

RULES:
1. Preserve every distinct fact, event, item, relationship, decision, and reveal. Never drop information just to shorten.
2. Merge near-duplicates into one bullet ("{{charName}} met the merchant" + "{{charName}} spoke to the merchant" → "{{charName}} met and spoke with the merchant").
3. Combine related bullets into denser ones ("{{charName}} bought a sword" + "{{charName}} spent 50 gold" → "{{charName}} bought a sword for 50 gold").
4. Drop a weaker bullet ONLY when a stronger bullet in the same cluster fully subsumes it. Never drop a bullet that carries unique information.
5. Keep the third-person, past-tense style. Preserve named actors and quantities.
6. Each merged bullet stays one line and starts with "- ".
7. First bullet is a cast tag: "- [Cast: A, B — short gist]".

Output ONLY the merged bullets. No headers, no commentary.`,
    },
};

/**
 * Modality extraction prompt.
 *
 * Splits a single raw message into two modality-pure versions so partial-
 * perception witnesses (see-only / hear-only) get clean text. Output is
 * strict JSON.
 */
const DEFAULT_MODALITY_EXTRACTION_PROMPT = {
    systemPrompt:
        'You split a single roleplay message into two modality-pure versions. Output strict JSON with keys "visual" and "audio". No prose, no markdown, no commentary — only the JSON object.',
    userPrompt:
        `Split the message below into two modality-pure versions.

MESSAGE FROM {{speakerName}}:
{{message}}

DEFINITIONS:
- visual = ONLY what an observer could SEE if they were in the room but could NOT hear anything. Actions, gestures, facial expressions, body language, movement, visible props, and scene changes. Report nothing about spoken words, tone, volume, or sound. If nothing visibly happens, return an empty string.
- audio  = ONLY what an observer could HEAR if they were blindfolded in the room. Spoken words (quoted exactly as said), sound effects, tone of voice, volume. Report nothing about visible actions or appearance. If nothing audible happens, return an empty string.

RULES:
1. Preserve the original tense and third-person voice.
2. Quote dialogue verbatim inside the audio version. Do not paraphrase.
3. Both fields may be non-empty for a mixed message (e.g. "drew his sword and shouted 'stop!'" → visual: "drew his sword", audio: "shouted 'stop!'").
4. If the whole message is purely descriptive narration with no character action or speech, put it in visual and leave audio empty.
5. Output ONLY the JSON object. Example: {"visual": "A drew his sword", "audio": "A shouted 'Get out!'"}`,
};

/**
 * Extract modality-pure visual/audio versions of a single message.
 *
 * Throws on ANY failure (network, malformed JSON, both-empty, retries
 * exhausted). User-initiated abort throws an AbortError instead, so
 * summarizeBatch can distinguish graceful stops from real failures.
 *
 * @param {object} deps Injected dependencies (same shape as callSummarizer).
 * @param {object} msg SillyTavern message.
 * @param {string} speakerName Display name for the message's speaker.
 * @returns {Promise<{visual: string, audio: string}>}
 * @throws {Error|AbortError} on any failure — caller distinguishes by name.
 */
export async function extractModalitySplit(deps, msg, speakerName) {
    // Strip non-diegetic content BEFORE sending so partial-perception slices
    // match what buildPassageForWitness feeds to full-perception witnesses.
    const rawText = stripNonDiegetic(msg?.mes || '').trim();
    if (!rawText) return { visual: '', audio: '' };

    const userPrompt = substitutePromptTemplate(DEFAULT_MODALITY_EXTRACTION_PROMPT.userPrompt, {
        speakerName: speakerName || 'Character',
        message: rawText,
    });

    const result = await callSummarizer(
        deps,
        DEFAULT_MODALITY_EXTRACTION_PROMPT.systemPrompt,
        userPrompt,
        deps.abortSignal?.(),
    );

    // callSummarizer returns '' on abort — rethrow as AbortError so the
    // caller's break-on-abort path fires.
    if (deps.abortSignal?.()?.aborted) {
        const err = new Error('Modality extraction aborted');
        err.name = 'AbortError';
        throw err;
    }

    if (!result || !result.trim()) {
        throw new Error('Modality extraction returned empty response');
    }

    // The summarizer pipeline may wrap output in code fences or stray prose.
    const trimmed = result.trim().replace(/^```(?:json)?/i, '').replace(/```$/i, '').trim();
    const start = trimmed.indexOf('{');
    const end = trimmed.lastIndexOf('}');
    if (start === -1 || end === -1 || end <= start) {
        throw new Error(`Modality extraction returned non-JSON response: ${result.slice(0, 200)}`);
    }
    const jsonStr = trimmed.slice(start, end + 1);

    let parsed;
    try {
        parsed = JSON.parse(jsonStr);
    } catch (err) {
        throw new Error(`Modality extraction returned unparseable JSON: ${err.message}`);
    }

    const visual = typeof parsed.visual === 'string' ? stripNonDiegetic(parsed.visual).trim() : '';
    const audio = typeof parsed.audio === 'string' ? stripNonDiegetic(parsed.audio).trim() : '';

    // If both came back empty, the LLM failed to extract anything useful.
    if (!visual && !audio) {
        throw new Error('Modality extraction returned both fields empty');
    }

    return { visual, audio };
}

/**
 * Resolve the prompt pair for a given layer.
 * Layer 0/1 resolve against their own slot. Layer 2+ uses an explicit override
 * if present (with field-level fallback to Layer 1), else inherits Layer 1.
 *
 * @param {object} s Settings object (extension_settings[MODULE_NAME]).
 * @param {number} layer Target layer for the call.
 * @returns {{ systemPrompt: string, userPrompt: string }}
 */
export function resolveLayerPrompt(s, layer) {
    const targetLayer = Number.isFinite(layer) && layer >= 0 ? Math.floor(layer) : 0;
    const layerPrompts = s?.layerPrompts;

    if (targetLayer <= 1) {
        const entry = layerPrompts?.[targetLayer] || {};
        const fallback = DEFAULT_LAYER_PROMPTS[targetLayer];
        return {
            systemPrompt: entry.systemPrompt ?? fallback.systemPrompt,
            userPrompt: entry.userPrompt ?? fallback.userPrompt,
        };
    }

    // Layer 2+: explicit override if it defines at least one field.
    const override = layerPrompts?.[targetLayer];
    const layer1 = layerPrompts?.[1] || {};
    const layer1Fallback = DEFAULT_LAYER_PROMPTS[1];
    const layer1System = layer1.systemPrompt ?? layer1Fallback.systemPrompt;
    const layer1User = layer1.userPrompt ?? layer1Fallback.userPrompt;

    if (override && (override.systemPrompt || override.userPrompt)) {
        return {
            systemPrompt: override.systemPrompt ?? layer1System,
            userPrompt: override.userPrompt ?? layer1User,
        };
    }
    return { systemPrompt: layer1System, userPrompt: layer1User };
}

/**
 * Build the summarization prompt for a specific character at a given layer.
 *
 * @param {object} deps
 * @param {string} charName
 * @param {string} passage
 * @param {string} priorContext
 * @param {number} [layer=0] Target layer. 0 = turn summary, 1+ = compression.
 * @returns {{ systemPrompt: string, userPrompt: string }}
 */
export function buildSummarizerPrompt(deps, charName, passage, priorContext, layer = 0) {
    const s = deps.settings();
    const { systemPrompt: sysTemplate, userPrompt: userTemplate } = resolveLayerPrompt(s, layer);

    // System prompts support {{charName}} only; passage-level variables
    // ({{passage}}, {{priorContext}}) stay exclusive to the user prompt.
    const systemPrompt = substitutePromptTemplate(sysTemplate, { charName });
    const userPrompt = substitutePromptTemplate(userTemplate, {
        charName,
        priorContext: priorContext || '(none yet)',
        passage: passage || '(empty)',
    });

    return { systemPrompt, userPrompt };
}

/**
 * Build prior context from a character's existing memory blocks.
 * Uses the most recent N Layer 0 blocks + the most recent deeper-layer blocks.
 *
 * @param {Array} blocks Parsed memory blocks for this character.
 * @param {number} recentCount How many Layer 0 blocks to include.
 * @param {number} deepCount Cap on deeper-layer (1+) blocks, so prior-context
 *        size stays bounded on long chats.
 * @returns {string}
 */
export function buildPriorContext(blocks, recentCount = 5, deepCount = 10) {
    if (!blocks || blocks.length === 0) return '(none yet)';

    // Promoted blocks have already been compressed into a higher layer.
    const layer0 = blocks.filter(b => (b.layer || 0) === 0 && !b.promoted);
    const deeper = blocks.filter(b => (b.layer || 0) > 0 && !b.promoted).slice(-deepCount);

    const parts = [];

    for (const b of deeper) {
        parts.push(b.bullets.join(' '));
    }

    const recent = layer0.slice(-recentCount);
    for (const b of recent) {
        parts.push(b.bullets.join(' '));
    }

    return parts.length > 0 ? parts.join(' ') : '(none yet)';
}

/**
 * Run summarization for ALL witnessing characters across a message range.
 *
 * The range [startIdx, endIdx] is split into sub-batches of size `interval`
 * to stay within the model's context window when processing a large backlog.
 *
 * @param {object} deps Injected dependencies.
 * @param {number} startIdx Start message index (inclusive).
 * @param {number} endIdx End message index (inclusive).
 * @returns {Promise<{totalSummaries: number, charactersProcessed: number, lastCommittedEnd: number|null}>}
 *          `lastCommittedEnd` is the MIN across all characters of the last
 *          persisted sub-batch endIdx, so partial runs aren't redone from
 *          scratch (range-based dedup absorbs residual overlaps).
 */
export async function summarizeBatch(deps, startIdx, endIdx) {
    const {
        settings,
        getChat,
        getTargets,
        getSpeakerAvatarFn,
        getUserName,
        readFile,
        writeFile,
        log,
    } = deps;

    const s = settings();
    const chat = getChat();
    const targets = getTargets();
    if (!Array.isArray(targets) || targets.length === 0) {
        log('No targets to summarize — getTargets() returned empty.', 'warning');
        return { totalSummaries: 0, charactersProcessed: 0, lastCommittedEnd: null };
    }
    if (!Array.isArray(chat) || chat.length === 0) {
        log('No chat to summarize.', 'warning');
        return { totalSummaries: 0, charactersProcessed: 0, lastCommittedEnd: null };
    }
    const userName = getUserName();

    // Split the range into sub-batches to avoid exceeding the LLM's context
    // window when processing a large backlog. Batch size comes from the
    // "Batch size (messages per summarization)" setting (s.interval).
    const batchSize = Math.max(1, s.interval || 10);
    const subBatches = [];
    // Defensively clamp the requested range to the actual chat length.
    const safeStart = Math.max(0, Math.min(startIdx, chat.length - 1));
    const safeEnd = Math.max(safeStart, Math.min(endIdx, chat.length - 1));
    for (let bs = safeStart; bs <= safeEnd; bs += batchSize) {
        subBatches.push([bs, Math.min(bs + batchSize - 1, safeEnd)]);
    }

    deps.debug?.(`summarizeBatch: ${safeStart}-${safeEnd} → ${subBatches.length} sub-batch(es) of ≤${batchSize} msgs`);

    // Modality-extraction pre-pass: for every message with at least one
    // partial-perception viewer, split raw text into visual+audio slices.
    // Done once per message (not per witness). Real failures propagate;
    // abort stops the pre-pass and falls back to raw text downstream.
    const splits = new Map();
    for (let i = safeStart; i <= safeEnd; i++) {
        if (deps.abortSignal?.()?.aborted) break;
        const msg = chat[i];
        if (!msg || !msg.mes) continue;
        if (msg.is_system && !msg.is_user && !msg.name) continue;
        if (!hasPartialPerception(msg, targets, getSpeakerAvatarFn)) continue;
        const speakerAvatar = getSpeakerAvatarFn(msg);
        if (!speakerAvatar) continue;
        const speakerName = msg.is_user ? (userName || 'User') : (msg.name || 'Character');
        deps.debug?.(`summarizeBatch: extracting modality split for message #${i + 1} (${speakerName})`);
        try {
            const split = await extractModalitySplit(deps, msg, speakerName);
            splits.set(i, split);
        } catch (err) {
            if (err?.name === 'AbortError' || deps.abortSignal?.()?.aborted) {
                deps.debug?.(`summarizeBatch: extraction aborted at message #${i + 1}`);
                break;
            }
            throw err;
        }
    }

    let totalSummaries = 0;
    let charactersProcessed = 0;
    // MIN across targets of the last persisted sub-batch endIdx.
    let lastCommittedEnd = null;

    for (const target of targets) {
        // Load existing memories once per character.
        // null = transient read error: SKIP this character entirely. Writing
        // with an empty block list would clobber the existing memory file.
        let existingBlocks = [];
        let readFailed = false;
        try {
            const existingContent = await readFile(target.avatar, target.fileName);
            if (existingContent === null) {
                readFailed = true;
            } else {
                existingBlocks = parseMemories(existingContent);
            }
        } catch (err) {
            log(`Could not read existing memories for ${target.name}: ${err.message}`, 'warning');
            readFailed = true;
        }
        if (readFailed) {
            log(`Skipping ${target.name} — memory file unreadable (transient error); refusing to overwrite`, 'warning');
            // Contribute safeStart-1 so the global cursor doesn't advance past
            // this character's gap; the range is retried on the next run.
            if (lastCommittedEnd === null || safeStart - 1 < lastCommittedEnd) {
                lastCommittedEnd = safeStart - 1;
            }
            continue;
        }

        let targetSummaries = 0;
        let targetLastCommitted = null;
        // Tracks LLM failures (empty/garbage output) so the global index
        // doesn't advance past this character's unrecoverable gap. Legitimate
        // perception-empty passages do NOT set this — they have nothing to
        // summarize, not a failure.
        let targetHadFailure = false;

        for (const [batchStart, batchEnd] of subBatches) {
            if (deps.abortSignal?.()?.aborted) break;

            const passage = buildPassageForWitness(
                chat, batchStart, batchEnd,
                target.avatar,
                getSpeakerAvatarFn, userName,
                splits,
            );

            if (!passage.text || passage.messageCount === 0) {
                log(`Skipping ${target.name} — witnessed nothing in range ${batchStart}-${batchEnd}`);
                continue;
            }

            // Build prior context from existing blocks (updated each sub-batch).
            const priorContext = buildPriorContext(existingBlocks, 5);

            const { systemPrompt, userPrompt } = buildSummarizerPrompt(
                deps, target.name, passage.text, priorContext, 0,
            );

            log(`Summarizing for ${target.name}: ${passage.messageCount} msgs (${batchStart}-${batchEnd})`);

            const summary = await callSummarizer(deps, systemPrompt, userPrompt, deps.abortSignal?.());

            if (!summary || !summary.trim()) {
                log(`Empty summary for ${target.name} — skipping batch ${batchStart}-${batchEnd}`, 'warning');
                targetHadFailure = true;
                continue;
            }

            const bullets = summary.split('\n')
                .map(l => l.trim())
                .filter(l => l.startsWith('- '))
                .map(l => l.replace(/^-\s+/, '').trim())
                .filter(Boolean);

            if (bullets.length === 0) {
                log(`No valid bullets in summary for ${target.name} — skipping batch ${batchStart}-${batchEnd}`, 'warning');
                targetHadFailure = true;
                continue;
            }

            const timestamp = getTimestamp();
            const newBlock = {
                chat: target.name,
                date: timestamp,
                bullets,
                layer: 0,
                range: [batchStart, batchEnd],
            };

            // Range-based dedup: if a Layer-0 block for this exact range is
            // already on disk (from a re-processed aborted run), replace it
            // in place rather than appending.
            const dupIdx = existingBlocks.findIndex(b =>
                (b.layer || 0) === 0
                && !b.promoted
                && Array.isArray(b.range)
                && b.range[0] === batchStart
                && b.range[1] === batchEnd
            );
            if (dupIdx >= 0) {
                existingBlocks[dupIdx] = newBlock;
                log(`Replaced existing Layer-0 block for range ${batchStart}-${batchEnd} (${target.name})`);
            } else {
                // Boundary-shift dedup: drop Layer-0 blocks fully contained in
                // the new range (left behind when a previous run used a
                // different batch size) so re-processing doesn't duplicate them.
                let absorbed = 0;
                for (let i = existingBlocks.length - 1; i >= 0; i--) {
                    const b = existingBlocks[i];
                    if ((b.layer || 0) === 0
                        && !b.promoted
                        && Array.isArray(b.range)
                        && b.range[0] >= batchStart
                        && b.range[1] <= batchEnd) {
                        existingBlocks.splice(i, 1);
                        absorbed++;
                    }
                }
                if (absorbed > 0) {
                    log(`Absorbed ${absorbed} stale Layer-0 block(s) inside range ${batchStart}-${batchEnd} (${target.name})`);
                }
                existingBlocks.push(newBlock);
            }
            targetSummaries += bullets.length;

            // Persist after every sub-batch so an interrupt can't lose work.
            await writeFile(serializeMemories(existingBlocks), target.avatar, target.fileName);
            targetLastCommitted = batchEnd;

            log(`Saved ${bullets.length} bullets for ${target.name} (batch ${batchStart}-${batchEnd})`, 'success');
        }

        // Fold this target's contribution into the global MIN.
        // Characters that committed something contribute their last commit.
        // Characters that failed every sub-batch contribute safeStart-1 so
        // the global cursor doesn't advance past their unrecoverable gap
        // (next run retries the range; range-dedup absorbs the overlap for
        // already-successful characters). Characters that legitimately
        // witnessed nothing (perception filter) contribute null and are
        // excluded — they have nothing to summarize.
        const contribution = (targetLastCommitted !== null)
            ? targetLastCommitted
            : (targetHadFailure ? safeStart - 1 : null);
        if (contribution !== null) {
            if (lastCommittedEnd === null || contribution < lastCommittedEnd) {
                lastCommittedEnd = contribution;
            }
        }

        if (targetSummaries === 0) continue;

        charactersProcessed++;

        // Check for layer promotion. Promoted and protected blocks are
        // excluded from the threshold count.
        const layer0Blocks = existingBlocks.filter(b => (b.layer || 0) === 0 && !b.promoted && !b.blockPromotion);
        const snippetsPerLayer = s.snippetsPerLayer || 30;

        if (layer0Blocks.length > snippetsPerLayer && (s.maxLayers || 5) > 1 && !deps.abortSignal?.()?.aborted) {
            const promoted = await promoteLayer(deps, target, existingBlocks, 0);
            if (promoted) existingBlocks = promoted;
        }

        // Final save captures any layer-promotion edits.
        const serialized = serializeMemories(existingBlocks);
        await writeFile(serialized, target.avatar, target.fileName);

        totalSummaries += targetSummaries;
        const finalLayer0 = existingBlocks.filter(b => (b.layer || 0) === 0 && !b.promoted).length;
        log(`Done with ${target.name}: ${targetSummaries} bullets, ${finalLayer0} active Layer 0 blocks`, 'success');
    }

    return { totalSummaries, charactersProcessed, lastCommittedEnd };
}

/**
 * Promote the oldest N Layer-X blocks to Layer X+1 via LLM re-summarization.
 * The promoted blocks are marked `promoted=true` ONLY after the LLM call
 * succeeds, so a failed promotion leaves Layer X untouched and retryable.
 *
 * Behaviours:
 *   - Progressive merge count. When settings.progressiveMerge is on (the
 *     default), the merge count scales with the source layer so higher
 *     layers are compressed more aggressively (L0→L1 = base, L1→L2 =
 *     base×2, …), capped at MAX_MERGE_CAP. Off restores flat behaviour.
 *   - Upward cascade. After settling layer N, we check layer N+1 and
 *     recurse upward if it has crossed the threshold.
 *
 * @param {object} deps
 * @param {object} target { avatar, name, fileName }
 * @param {Array} blocks All memory blocks for this character (mutated).
 * @param {number} layerIndex Layer to promote from.
 * @param {number} [depth=0] Internal recursion depth guard.
 * @returns {Promise<Array|null>} Updated blocks array, or null on failure.
 */
export async function promoteLayer(deps, target, blocks, layerIndex, depth = 0) {
    const { settings, log } = deps;
    const s = settings();

    const maxLayers = s.maxLayers || 5;
    if (layerIndex >= maxLayers - 1) return null;
    // Hard recursion guard.
    if (depth >= 100) {
        log(`Promotion recursion depth exceeded for ${target.name} — stopping`, 'warning');
        return null;
    }

    const snippetsPerLayer = s.snippetsPerLayer || 30;

    // Progressive merge count: higher source layers merge more blocks per call.
    // Layer 0 is always ×1, so the L0→L1 cadence is preserved when
    // progressiveMerge is on; turning it off restores flat behaviour.
    const baseMerge = Math.max(2, s.snippetsPerPromotion || 3);
    const MAX_MERGE_CAP = 30;
    const multiplier = s.progressiveMerge !== false
        ? (layerIndex + 1) * (Number(s.layerMergeMultiplier) || 1)
        : 1;
    const snippetsPerPromotion = Math.min(
        MAX_MERGE_CAP,
        Math.max(2, Math.round(baseMerge * multiplier)),
    );

    // Protected blocks are never selected for merging.
    const layerBlocks = blocks.filter(b => (b.layer || 0) === layerIndex && !b.promoted && !b.blockPromotion);
    if (layerBlocks.length <= snippetsPerLayer) return null;

    log(`Promoting Layer ${layerIndex} → ${layerIndex + 1} for ${target.name} (${layerBlocks.length} > ${snippetsPerLayer}, merging ${snippetsPerPromotion})`);

    // Take the oldest N blocks for promotion.
    const toMerge = layerBlocks.slice(0, snippetsPerPromotion);

    // Build the passage from their bullets. Track the input bullet count so
    // we can log the actual compression ratio after the LLM call.
    const inputBullets = toMerge.reduce((n, b) => n + b.bullets.length, 0);
    const storyText = toMerge.map(b => b.bullets.join(' ')).join('\n');

    // Build prior context from deeper (non-promoted) layers.
    const deeperBlocks = blocks.filter(b => (b.layer || 0) > layerIndex && !b.promoted);
    const priorContext = buildPriorContext(deeperBlocks, 5);

    // Compression call: target the layer we're promoting INTO. Layer 2+
    // inherits Layer 1's prompts unless an explicit override exists.
    const { systemPrompt, userPrompt } = buildSummarizerPrompt(
        deps, target.name, storyText, priorContext, layerIndex + 1,
    );
    // Pass a diagnostics object so we can surface the real failure cause below.
    const promotionDiag = { lastError: null };
    const metaSummary = await callSummarizer(deps, systemPrompt, userPrompt, deps.abortSignal?.(), promotionDiag);

    if (!metaSummary || !metaSummary.trim()) {
        const reason = promotionDiag.lastError?.message || 'empty response';
        log(`Promotion failed for ${target.name} — keeping Layer ${layerIndex} blocks (${reason})`, 'warning');
        return null;
    }

    const bullets = metaSummary.split('\n')
        .map(l => l.trim())
        .filter(l => l.startsWith('- '))
        .map(l => l.replace(/^-\s+/, '').trim())
        .filter(Boolean);

    if (bullets.length === 0) {
        const reason = promotionDiag.lastError?.message || 'no valid bullets in output';
        log(`Promotion produced no valid bullets for ${target.name} (${reason})`, 'warning');
        return null;
    }

    // Compute the union range spanning ALL merged blocks (not just the first).
    const ranges = toMerge.map(b => b.range).filter(Boolean);
    let range = undefined;
    if (ranges.length > 0) {
        const starts = ranges.map(r => r[0]).filter(Number.isFinite);
        const ends = ranges.map(r => r[1]).filter(Number.isFinite);
        if (starts.length > 0 && ends.length > 0) {
            range = [Math.min(...starts), Math.max(...ends)];
        }
    }

    const promotedBlock = {
        chat: target.name,
        date: getTimestamp(),
        bullets,
        layer: layerIndex + 1,
        promoted: false,
    };
    if (range) promotedBlock.range = range;

    blocks.push(promotedBlock);

    // Mark the old blocks as promoted ONLY after the LLM call succeeded.
    for (const b of toMerge) {
        b.promoted = true;
    }

    log(`Promoted ${toMerge.length} blocks to Layer ${layerIndex + 1} for ${target.name} (${inputBullets} → ${bullets.length} bullets)`, 'success');

    // Recursive: check if this layer still has too many unpromoted blocks.
    const stillOver = blocks.filter(b => (b.layer || 0) === layerIndex && !b.promoted && !b.blockPromotion).length;
    if (stillOver > snippetsPerLayer && !deps.abortSignal?.()?.aborted) {
        return await promoteLayer(deps, target, blocks, layerIndex, depth + 1);
    }

    // Upward cascade: the block we just pushed to Layer N+1 may have pushed
    // that layer over its threshold.
    if (layerIndex + 1 <= maxLayers - 1 && !deps.abortSignal?.()?.aborted) {
        const aboveBlocks = blocks.filter(b => (b.layer || 0) === layerIndex + 1 && !b.promoted && !b.blockPromotion);
        if (aboveBlocks.length > snippetsPerLayer) {
            return await promoteLayer(deps, target, blocks, layerIndex + 1, depth + 1);
        }
    }

    return blocks;
}
