/**
 * CharSummaryception — Per-Character Memory with Perception
 *
 * Combines Summaryception's batch summarization + layered compression with
 * CharMemory's per-character Data Bank storage and perception-aware witness
 * filtering.
 *
 * Each character gets their own memory file. Only messages they witnessed are
 * summarized into their file. Vector Storage retrieves relevant memories at
 * generation time.
 *
 * AGPL-3.0 (adapted from Summaryception by Lodactio)
 */

import {
    eventSource,
    event_types,
    saveSettingsDebounced,
    saveChatDebounced,
    chat_metadata,
    characters,
    this_chid,
} from '../../../../script.js';
import { convertTextToBase64, getStringHash } from '../../../utils.js';
import {
    getContext,
    extension_settings,
    renderExtensionTemplateAsync,
    saveMetadataDebounced,
} from '../../../extensions.js';
import {
    getFileAttachment,
    uploadFileAttachment,
    deleteFileFromServer,
} from '../../../chats.js';
import { callGenericPopup, POPUP_TYPE } from '../../../popup.js';

import {
    escapeAttr,
    escapeHtml,
    parseMemories,
    serializeMemories,
    countMemories,
    stripNonDiegetic,
    substitutePromptTemplate,
    getTimestamp,
    USER_AVATAR,
    hasPartialPerception,
} from './lib.js';

import {
    renderConnectionDetailsHTML,
    hydrateConnectionDetails,
} from './connection.js';

import {
    summarizeBatch,
    promoteLayer,
    buildPriorContext,
    callSummarizer,
    DEFAULT_LAYER_PROMPTS,
    resolveLayerPrompt,
} from './summarizer.js';

const MODULE_NAME = 'charSummaryception';
const LOG_PREFIX = '[CharSummaryception]';
const DEFAULT_FILE_NAME = 'char-memories.md';

function log(...args) {
    if (extension_settings[MODULE_NAME]?.debugMode) console.log(LOG_PREFIX, ...args);
}

function trace(...args) {
    if (extension_settings[MODULE_NAME]?.traceMode) console.log(LOG_PREFIX, '[TRACE]', ...args);
}

const defaultSettings = {
    enabled: true,
    interval: 10,
    protectedMessages: 5,
    snippetsPerLayer: 30,
    snippetsPerPromotion: 3,
    maxLayers: 5,

    // Progressive layer compression. When on, the merge count at each promotion
    // scales with the source layer (capped at 30). layerMergeMultiplier scales
    // the progression further (0.5 = gentler, 2 = more aggressive).
    progressiveMerge: true,
    layerMergeMultiplier: 1.0,

    // Per-layer prompts. Each entry is { systemPrompt, userPrompt }.
    // Layer 2+ inherits Layer 1 unless its key is present here.
    //
    // Migration: pre-per-layer installs used flat `summarizerSystemPrompt` /
    // `summarizerUserPrompt` strings, rolled forward into layerPrompts[0] by
    // migrateOldPrompts().
    layerPrompts: {
        0: { ...DEFAULT_LAYER_PROMPTS[0] },
        1: { ...DEFAULT_LAYER_PROMPTS[1] },
    },

    stripPatterns: [
        '<|channel>thought',
        '<channel|>',
        '<output>',
        '</output>',
        '<thinking>',
        '</thinking>',
    ],

    connectionSource: 'default',
    connectionProfileId: '',
    openaiUrl: '',
    openaiKey: '',
    openaiModel: '',
    openaiMaxTokens: 0,
    openrouterKey: '',
    openrouterModel: '',
    openrouterMaxTokens: 0,
    openrouterReasoning: '',

    fileName: '',
    perChat: false,
    characterFileNames: {},

    // Prompt presets. savedCustomPrompts stores per-layer structures
    // { 0: {systemPrompt,userPrompt}, 1: {...}, ... }; lastCustomPrompt holds
    // the most recent Custom-mode snapshot, so presets can be flipped without
    // losing edits.
    promptPreset: 'narrative',
    savedCustomPrompts: {},
    lastCustomPrompt: null,

    debugMode: false,
    traceMode: false,
};

// Each preset is a sparse { layer: { systemPrompt, userPrompt } } map.
// Layer 2+ inherits Layer 1 unless the user adds an explicit override.

const PROMPT_PRESETS = {
    // Narrative tracks DEFAULT_LAYER_PROMPTS so the default preset and the
    // "Reset to default" target stay in sync.
    narrative: {
        0: { ...DEFAULT_LAYER_PROMPTS[0] },
        1: { ...DEFAULT_LAYER_PROMPTS[1] },
    },

    gamestate: {
        0: {
            systemPrompt:
                'You are a game-state tracker for roleplay. Extract concrete, retrievable facts with source attribution. Output only bullet points — no preamble, no commentary, no markdown.',
            userPrompt:
                `You are tracking concrete game state on behalf of {{charName}}. Extract every retrievable fact from the passage below.

PRIOR CONTEXT (state already tracked — do NOT repeat):
{{priorContext}}

PASSAGE (new events to extract state from):
{{passage}}

METHOD — two passes:
1. Silently identify every span where state changes: a quest given or completed, a location entered, an item gained or lost, a resource spent, a relationship shifted, a status effect applied, a decision made, a world fact revealed.
2. Then write one atomic fact per identified span, tagged by category.

RULES:
1. One fact per bullet. No compound clauses.
2. Prefix every bullet with a category tag: [QUEST] [LOC] [ITEM] [REL] [STATUS] [DECISION] [WORLD] [RESOURCE].
3. Record quantities and exact states: "has 47 gold", "HP: badly wounded", "owns 3 potions" — never "has some gold".
4. End every bullet with its provenance in parentheses: "(said by NAME)" for things a character stated, or "(observed)" for what {{charName}} directly perceived. A rumor and a witnessed fact are different.
5. Only include what {{charName}} could perceive.
6. Do NOT repeat facts already in PRIOR CONTEXT. If a value changed, record only the new value.
7. Examples:
   "- [QUEST] {{charName}} accepted 'Recover the Amulet' from the merchant (said by Merchant)."
   "- [ITEM] {{charName}} now carries a steel dagger (observed)."
   "- [RESOURCE] {{charName}} spent 30 gold, has 20 remaining (observed)."
8. Write 1-10 bullets. Skip categories with nothing new.

Output ONLY the bullets. No headers, no commentary.`,
        },
        1: {
            systemPrompt:
                'You are a game-state compressor. Reconcile superseded values, mark lifecycle changes, and deduplicate tagged facts. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are compressing {{charName}}'s tracked game-state bullets. The bullets below come from earlier scenes. Reconcile progression, deduplicate, and consolidate — but preserve every distinct fact.

PRIOR CONTEXT (already-compressed state — do NOT repeat):
{{priorContext}}

BULLETS TO MERGE (compress these):
{{passage}}

COMPRESSION METHOD — two passes:
1. Silently group bullets by entity (the same quest, item, or relationship). Detect superseded values and lifecycle changes.
2. Then emit one reconciled bullet per entity.

RULES:
1. Preserve every distinct fact. Track: quests, locations, items, resources, relationships, status effects, decisions, world-state.
2. Reconcile superseded values explicitly: "had 50 gold" + "spent 30 gold" → "has 20 gold (was 50)".
3. Mark quest/objective lifecycle: started → in progress → completed [DONE] / abandoned [DROPPED].
4. When two bullets describe the same entity at different points, keep the latest state and drop the stale one — UNLESS the stale value is itself load-bearing (a debt, a lie, a hidden condition).
5. Keep category tags: [QUEST] [LOC] [ITEM] [REL] [STATUS] [DECISION] [WORLD] [RESOURCE].
6. Preserve provenance suffixes: (said by NAME) / (observed). If a later observation contradicts an earlier rumor, keep the contradiction visible: "(said by Merchant; contradicted by observation)".
7. Each merged bullet stays one line and starts with "- ".

Output ONLY the merged bullets. No headers, no commentary.`,
        },
    },

    emotional: {
        0: {
            systemPrompt:
                'You are an emotional and relationship tracker for roleplay. Record feelings, bonds, and interpersonal shifts as the character would perceive them. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are tracking the emotional landscape around {{charName}} — what {{charName}} feels, what others seem to feel toward {{charName}}, and how relationships are shifting. Summarize the passage below through that lens.

PRIOR CONTEXT (emotional state already tracked — do NOT repeat):
{{priorContext}}

PASSAGE (what just happened):
{{passage}}

METHOD — two passes:
1. Silently identify spans where an emotion is shown, a bond shifts, a vulnerability surfaces, a tension builds or releases, or an unspoken feeling becomes legible.
2. Then write one bullet per identified span.

RULES:
1. Third person, past tense. Name the actors: "{{charName}} felt betrayed" — not "I felt betrayed".
2. Track three things: (a) {{charName}}'s own feelings, (b) how others APPEAR to feel toward {{charName}} (clearly marked as inferred — "Marcus seemed angry at {{charName}}"), (c) shifts in the bond itself.
3. Record the DIRECTION and MAGNITUDE of shifts, not just static state: "{{charName}}'s trust in Marcus dropped from warm to wary" beats "{{charName}} distrusts Marcus".
4. Tag relationship beats: [FEELING] [BOND] [TENSION] [RESOLVED] [REVEAL].
5. Preserve the trigger: "{{charName}} grew colder toward Marcus after Marcus lied about the gold".
6. Only what {{charName}} could witness or feel. Clearly mark inference; never mind-read.
7. Do NOT repeat states already in PRIOR CONTEXT. Record only the new shift or its confirmation.
8. First bullet is a cast tag: "- [Cast: A, B — emotional gist of the scene]".

Output ONLY the bullets. No headers, no commentary.`,
        },
        1: {
            systemPrompt:
                'You are an emotional-state compressor for roleplay. Roll feelings and bonds forward to their latest state, archive resolved tensions, and preserve every distinct shift. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are compressing {{charName}}'s emotional and relationship bullets. The bullets below come from earlier scenes. Roll emotional arcs forward, archive resolved tensions, and deduplicate — but preserve every distinct shift and its trigger.

PRIOR CONTEXT (already-compressed state — do NOT repeat):
{{priorContext}}

BULLETS TO MERGE (compress these):
{{passage}}

COMPRESSION METHOD — two passes:
1. Silently cluster bullets by relationship (the same pair of characters) and by emotional arc. Identify the latest state of each bond and any tensions that have since resolved.
2. Then emit one reconciled bullet per arc.

RULES:
1. Roll emotional states forward: keep the LATEST feeling/trust level, but preserve the arc that produced it ("{{charName}}'s trust in Marcus: warm → wary → cold after the lie and the betrayal").
2. Archive resolved tensions as completed arcs, not deleted ones: "[RESOLVED] {{charName}} forgave Marcus for the lie (was [TENSION])".
3. Never drop a named trigger (the event that caused a shift) — only compress the wording.
4. Merge near-duplicate readings of the same moment.
5. Keep tags: [FEELING] [BOND] [TENSION] [RESOLVED] [REVEAL].
6. Preserve the distinction between {{charName}}'s own feelings and inferred feelings of others.
7. Each merged bullet stays one line and starts with "- ".

Output ONLY the merged bullets. No headers, no commentary.`,
        },
    },

    journal: {
        0: {
            systemPrompt:
                'You are writing the private journal of {{charName}}. Capture how the character would privately reflect on events — first person, introspective, in their own voice. Output only journal entries as bullet points.',
            userPrompt:
                `You are writing {{charName}}'s private journal for this stretch of events. Record what {{charName}} would put to paper if no one else would ever read it — in their own voice, first person.

PRIOR CONTEXT (what {{charName}} has already journaled — do NOT repeat):
{{priorContext}}

PASSAGE (what just happened):
{{passage}}

METHOD — two passes:
1. Silently recall what {{charName}} most strongly felt, realized, decided, or was shaken by in this passage. Filter for what a real person would bother writing down — not a transcript, but the weight of the day.
2. Then write the journal entry.

RULES:
1. First person, past tense. "I met the merchant and I didn't trust him." — not "{{charName}} met the merchant".
2. Write in {{charName}}'s voice: their vocabulary, their biases, their blind spots. A soldier's journal sounds different from a scholar's.
3. Include {{charName}}'s private interpretation — not just what happened, but what it meant to them and what they suspect. Mark suspicions as such ("I think Marcus lied, but I can't prove it yet.").
4. Record feelings, doubts, and unanswered questions — these are the heart of a journal.
5. Only what {{charName}} witnessed, felt, or believes. No omniscient narration.
6. Do NOT repeat entries already in PRIOR CONTEXT. If a thread continues, reference it lightly ("Marcus lied again — I'm done trusting him.").
7. Write 1-5 bullets. Prefer one dense, voicey entry over many flat ones when the scene is quiet.
8. First bullet is a gist header: "- [Journal — short gist of the day]".

Output ONLY the bullets. No headers, no commentary.`,
        },
        1: {
            systemPrompt:
                'You are compressing a character journal into a condensed retrospective. Preserve the voice, the emotional arc, and every private conviction. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are condensing {{charName}}'s older journal entries into a tighter retrospective. The entries below come from earlier stretches of the story. Merge repetitive days, preserve the voice, and keep every conviction, doubt, and turning point.

PRIOR CONTEXT (already-condensed journal — do NOT repeat):
{{priorContext}}

BULLETS TO MERGE (compress these):
{{passage}}

COMPRESSION METHOD — two passes:
1. Silently cluster entries by the thread they describe (a recurring doubt, an ongoing rivalry, a slow realization). Identify entries that are just restatements of an earlier conviction.
2. Then write one condensed entry per thread.

RULES:
1. Stay in {{charName}}'s first-person voice. The compression must still sound like them.
2. Preserve every conviction, named person, doubt, and turning point. Drop only pure repetition.
3. When the same doubt recurs across days, compress it into one line that captures the arc: "I kept circling back to whether Marcus was lying — by the third time, I'd made up my mind".
4. Keep gist headers: "- [Journal — gist]".
5. Never flatten emotional nuance into plot summary. A journal is feelings, not a log.

Output ONLY the merged bullets. No headers, no commentary.`,
        },
    },

    minimal: {
        0: {
            systemPrompt:
                'You are a minimalist memory extractor. Record only load-bearing facts in the fewest words possible. No prose, no filler. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are extracting bare facts for {{charName}}'s memory — token-efficient, no prose, no commentary. Record only what would matter if this were the only thing {{charName}} remembered.

PRIOR CONTEXT (already recorded — do NOT repeat):
{{priorContext}}

PASSAGE (what just happened):
{{passage}}

METHOD — two passes:
1. Silently identify the load-bearing facts: who acted, what changed, what was gained, lost, revealed, or decided. Discard everything that is mood, atmosphere, restatement, or filler.
2. Then write one terse line per fact.

RULES:
1. Maximum brevity. "{{charName}} bought sword, -50g" beats "{{charName}} purchased a sword from the merchant for 50 gold pieces".
2. Drop articles and filler where possible. Telegraphese is fine.
3. One fact per line. No compound clauses.
4. Third person, past tense. Name the actor.
5. Only what {{charName}} perceived. No inference.
6. Do NOT repeat anything in PRIOR CONTEXT.
7. Hard cap: at most 5 bullets per passage. If the passage has more facts, keep only the 5 most load-bearing.
8. Skip a bullet rather than pad. If nothing important happened, output fewer bullets — even zero.

Output ONLY the bullets. No headers, no commentary.`,
        },
        1: {
            systemPrompt:
                'You are a minimalist memory compressor. Shrink bullets aggressively, drop the weakest first, never exceed half the input count. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are compressing {{charName}}'s minimalist memory bullets. The bullets below come from earlier scenes. Shrink them aggressively, drop the weakest, and merge where possible.

PRIOR CONTEXT (already-compressed — do NOT repeat):
{{priorContext}}

BULLETS TO MERGE (compress these):
{{passage}}

COMPRESSION METHOD — two passes:
1. Silently rank bullets by load-bearing importance. Identify which can be merged into a denser line and which can be dropped entirely without losing a distinct fact.
2. Then emit the smallest set that preserves every distinct fact.

RULES:
1. Hard cap: output AT MOST half the number of input bullets (rounded up). Prefer fewer.
2. Preserve every distinct load-bearing fact. Drop only true duplication or pure mood.
3. Merge related bullets into one terse line: "{{charName}} bought sword, -50g" + "{{charName}} has 20g left" → "{{charName}} bought sword (was 70g, now 20g)".
4. Maximum brevity. Telegraphese is encouraged.
5. One fact per line.
6. Third person, past tense. Named actors.

Output ONLY the merged bullets. No headers, no commentary.`,
        },
    },

    goals: {
        0: {
            systemPrompt:
                'You are a goal and motive tracker for roleplay. Record the objectives, stated reasons, and revealed hidden motives of each character. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are tracking objectives and motives in {{charName}}'s world. Record what each character is trying to do, why they say they are doing it, and — when revealed — what they actually want.

PRIOR CONTEXT (goals already tracked — do NOT repeat):
{{priorContext}}

PASSAGE (what just happened):
{{passage}}

METHOD — two passes:
1. Silently identify spans where a goal is stated, advanced, completed, or abandoned; where a motive is revealed or revised; where a plan forms or fails.
2. Then write one bullet per identified span.

RULES:
1. Third person, past tense. Name the actor: "{{charName}} committed to finding the amulet".
2. Tag every bullet: [GOAL] (an objective), [MOTIVE-STATED] (a claimed reason), [MOTIVE-HIDDEN] (a revealed true reason), [PLAN] (a course of action), [OBSTACLE] (something in the way).
3. Record lifecycle where known: [ACTIVE] / [DONE] / [DROPPED]. If unknown, omit.
4. Distinguish stated from hidden motives explicitly. Record [MOTIVE-HIDDEN] only when the passage actually reveals it — never speculate: "- [MOTIVE-HIDDEN] The merchant wants the amulet for himself (revealed when {{charName}} saw him pocket a second map)".
5. Preserve the trigger that creates or changes a goal: "{{charName}} pledged to find the amulet after the merchant's daughter begged him".
6. Only what {{charName}} could perceive. {{charName}}'s own goals count too.
7. Do NOT repeat goals already in PRIOR CONTEXT. Record only new goals, progress, or changes.
8. Write 1-6 bullets. Skip tags with nothing new.

Output ONLY the bullets. No headers, no commentary.`,
        },
        1: {
            systemPrompt:
                'You are a goal-and-motive compressor. Roll lifecycles forward, preserve every distinct objective and revealed motive, and deduplicate. Output only bullet points — no preamble, no commentary.',
            userPrompt:
                `You are compressing {{charName}}'s goal and motive bullets. The bullets below come from earlier scenes. Roll lifecycles forward, merge duplicates, and preserve every distinct objective, plan, and revealed motive.

PRIOR CONTEXT (already-compressed goals — do NOT repeat):
{{priorContext}}

BULLETS TO MERGE (compress these):
{{passage}}

COMPRESSION METHOD — two passes:
1. Silently cluster bullets by the goal or motive they track. Identify the latest lifecycle state of each and any motives that were later revealed or revised.
2. Then emit one reconciled bullet per goal or motive.

RULES:
1. Roll lifecycle forward: a goal that appears as [ACTIVE], then [DONE], compresses to "[DONE]" — but keep the original trigger ("found the amulet after the merchant's daughter begged him").
2. When a [MOTIVE-STATED] is later revealed as [MOTIVE-HIDDEN], keep both layers explicitly: "Merchant claimed to want justice; actually wanted the amulet (revealed later)".
3. Never drop a distinct goal, even a [DROPPED] one — abandoned goals may resurface. Compress wording only.
4. Merge true duplicates (same goal stated twice with no change).
5. Keep tags: [GOAL] [MOTIVE-STATED] [MOTIVE-HIDDEN] [PLAN] [OBSTACLE] and lifecycle markers [ACTIVE] / [DONE] / [DROPPED].
6. Preserve named triggers — the event that created or changed the goal.
7. Each merged bullet stays one line and starts with "- ".

Output ONLY the merged bullets. No headers, no commentary.`,
        },
    },
};

const MAX_LOG_ENTRIES = 200;
let activityLog = [];

function logActivity(message, type = 'info') {
    const entry = {
        timestamp: new Date().toLocaleTimeString(),
        message,
        type,
    };
    activityLog.unshift(entry);
    if (activityLog.length > MAX_LOG_ENTRIES) activityLog.length = MAX_LOG_ENTRIES;
    updateActivityLogDisplay();
    // Warnings/errors always reach the console; info/success is debug-gated.
    if (type === 'error' || type === 'warning' || extension_settings[MODULE_NAME]?.debugMode) {
        console.log(LOG_PREFIX, `[${type}]`, message);
    }
}

function getSettings() {
    return extension_settings[MODULE_NAME];
}

function loadSettings() {
    if (!extension_settings[MODULE_NAME]) {
        extension_settings[MODULE_NAME] = {};
    }
    // Migrate the old flat prompt format BEFORE seeding defaults, so the
    // user's previously-edited prompts land in layerPrompts[0] rather than
    // being clobbered by the new defaults.
    migrateOldPrompts();
    for (const key of Object.keys(defaultSettings)) {
        if (extension_settings[MODULE_NAME][key] === undefined) {
            const def = defaultSettings[key];
            // Deep-clone object/array defaults so in-place mutations by
            // handlers (e.g. s.savedCustomPrompts[name] = ...) don't pollute
            // the shared module-level defaultSettings. Primitives are safe
            // to assign directly.
            extension_settings[MODULE_NAME][key] = (def && typeof def === 'object')
                ? JSON.parse(JSON.stringify(def))
                : def;
        }
    }
}

/**
 * Roll forward the old flat prompt format into the per-layer structure.
 *
 * Old:  summarizerSystemPrompt / summarizerUserPrompt (flat strings)
 * New:  layerPrompts: { 0: {systemPrompt, userPrompt}, 1: {...}, ... }
 *
 * Idempotent: only migrates legacy slots it hasn't seen before. Old keys are
 * left in place but ignored by defaultSettings going forward.
 */
function migrateOldPrompts() {
    const s = extension_settings[MODULE_NAME];
    if (!s) return;

    if (!s.layerPrompts || typeof s.layerPrompts !== 'object') {
        s.layerPrompts = {
            0: { ...DEFAULT_LAYER_PROMPTS[0] },
            1: { ...DEFAULT_LAYER_PROMPTS[1] },
        };

        // Previously-edited flat prompts become the Layer 0 prompt. The shared
        // system prompt is also copied to Layer 1 so pre-feature compression
        // behaviour (one prompt for everything) is preserved.
        if (typeof s.summarizerSystemPrompt === 'string' && s.summarizerSystemPrompt) {
            s.layerPrompts[0].systemPrompt = s.summarizerSystemPrompt;
            s.layerPrompts[1].systemPrompt = s.summarizerSystemPrompt;
        }
        if (typeof s.summarizerUserPrompt === 'string' && s.summarizerUserPrompt) {
            s.layerPrompts[0].userPrompt = s.summarizerUserPrompt;
            // Also copy to Layer 1 so pre-feature compression behaviour (one
            // prompt for everything) is fully preserved — not just the system
            // prompt but the user prompt too.
            s.layerPrompts[1].userPrompt = s.summarizerUserPrompt;
        }
    }

    // lastCustomPrompt: string → { 0: { userPrompt } }. Other layers fall
    // back to layerPrompts via resolveLayerPrompt.
    if (typeof s.lastCustomPrompt === 'string') {
        const str = s.lastCustomPrompt;
        s.lastCustomPrompt = str ? { 0: { userPrompt: str } } : null;
    }

    // savedCustomPrompts: each string slot → { 0: { userPrompt } }.
    if (s.savedCustomPrompts && typeof s.savedCustomPrompts === 'object') {
        for (const [name, value] of Object.entries(s.savedCustomPrompts)) {
            if (typeof value === 'string') {
                s.savedCustomPrompts[name] = value
                    ? { 0: { userPrompt: value } }
                    : { 0: { ...DEFAULT_LAYER_PROMPTS[0] } };
            }
        }
    }
}

// The prompts tab edits one layer at a time, but the underlying data is a
// sparse { layer: { systemPrompt, userPrompt } } map. We mirror that map
// into a modal-local buffer so the user can switch layers without losing
// in-flight edits. snapshotForm flushes the buffer back on Save.

let promptEditBuffer = null;
let promptEditLayer = 0;
// Modal-local preset selection and "left Custom mid-session" stash. These
// mirror what the modal shows WITHOUT touching live settings, so Cancel
// rolls everything back. Committed to settings only on Save.
let promptEditPreset = 'narrative';
let promptEditStash = undefined;

/**
 * Defensive deep-clone of a layer-prompts map. Always returns keys 0 and 1
 * (falling back to DEFAULT_LAYER_PROMPTS).
 */
function cloneLayerPrompts(src) {
    const out = {};
    if (src && typeof src === 'object') {
        for (const [k, v] of Object.entries(src)) {
            const layer = Number(k);
            if (!Number.isFinite(layer) || layer < 0) continue;
            if (!v || typeof v !== 'object') continue;
            const entry = {};
            if (typeof v.systemPrompt === 'string') entry.systemPrompt = v.systemPrompt;
            if (typeof v.userPrompt === 'string') entry.userPrompt = v.userPrompt;
            if (Object.keys(entry).length > 0) out[layer] = entry;
        }
    }
    if (!out[0]) out[0] = { ...DEFAULT_LAYER_PROMPTS[0] };
    if (!out[1]) out[1] = { ...DEFAULT_LAYER_PROMPTS[1] };
    return out;
}

/**
 * Open a fresh edit buffer for the modal session, seeded from the live
 * settings so edits never mutate settings until Save.
 */
function openPromptEditBuffer(s) {
    promptEditBuffer = cloneLayerPrompts(s.layerPrompts);
    if (!promptEditBuffer[promptEditLayer] && promptEditLayer < 2) {
        promptEditLayer = 0;
    }
}

/**
 * Rebuild the layer-selector dropdown from the buffer + maxLayers.
 * Layers ≥2 without an explicit entry show as "inherits Layer 1".
 */
function rebuildPromptLayerOptions(selectedLayer, maxLayers, buffer) {
    const opts = [];
    opts.push(`<option value="0"${selectedLayer === 0 ? ' selected' : ''}>Layer 0 — Turn Summaries</option>`);
    opts.push(`<option value="1"${selectedLayer === 1 ? ' selected' : ''}>Layer 1 — Compression (default for 1+)</option>`);
    const ceiling = Math.max(2, Number(maxLayers) || 5);
    for (let i = 2; i < ceiling; i++) {
        const hasOverride = !!buffer?.[i] && (!!buffer[i].systemPrompt || !!buffer[i].userPrompt);
        const label = hasOverride ? `Layer ${i} (override)` : `Layer ${i} (inherits Layer 1)`;
        opts.push(`<option value="${i}"${selectedLayer === i ? ' selected' : ''}>${label}</option>`);
    }
    $('#csc_modal_promptLayer').html(opts.join(''));
}

/**
 * Push the textarea contents for the CURRENTLY-shown layer back into the
 * buffer. A no-op when the textareas are disabled (unoverridden Layer 2+).
 */
function syncCurrentLayerToBuffer() {
    if (!promptEditBuffer) return;
    const $sys = $('#csc_modal_systemPrompt');
    const $user = $('#csc_modal_userPrompt');
    if ($sys.prop('disabled') || $user.prop('disabled')) return;
    const layer = promptEditLayer;
    if (!promptEditBuffer[layer]) promptEditBuffer[layer] = {};
    promptEditBuffer[layer].systemPrompt = $sys.val() || '';
    promptEditBuffer[layer].userPrompt = $user.val() || '';
}

/**
 * Hydrate the system/user textareas for a given layer.
 *
 * Layer 0/1: always editable. Layer 2+ with override: editable. Layer 2+
 * without override: textareas show Layer 1's resolved value read-only with
 * a "Create override" button — displayed text is NOT the user's, so inputs
 * are disabled to prevent silently capturing Layer 1's text as an edit.
 */
function loadPromptLayerIntoForm(layer) {
    promptEditLayer = layer;
    if (!promptEditBuffer) return;

    const entry = promptEditBuffer[layer];
    const isOverrideLayer = layer >= 2;
    const hasOverride = !!entry && (!!entry.systemPrompt || !!entry.userPrompt);

    const $sys = $('#csc_modal_systemPrompt');
    const $user = $('#csc_modal_userPrompt');
    const $status = $('#csc_modal_promptLayerStatus');
    const $sysLabel = $('#csc_modal_systemLabel');
    const $create = $('#csc_modal_createOverride');
    const $remove = $('#csc_modal_removeOverride');
    const $reset = $('#csc_modal_resetLayer');

    $sysLabel.text('');

    if (isOverrideLayer && !hasOverride) {
        const inherited = resolveLayerPrompt({ layerPrompts: promptEditBuffer }, 1);
        $sys.val(inherited.systemPrompt).prop('disabled', true);
        $user.val(inherited.userPrompt).prop('disabled', true);
        $status.html(`<i class="fa-solid fa-circle-info"></i> Layer ${layer} inherits Layer 1's prompt. Click <b>Create override</b> to edit it separately.`);
        $create.show();
        $remove.hide();
        $reset.hide();
        return;
    }

    const resolved = resolveLayerPrompt({ layerPrompts: promptEditBuffer }, layer);
    $sys.val(entry?.systemPrompt ?? resolved.systemPrompt).prop('disabled', false);
    $user.val(entry?.userPrompt ?? resolved.userPrompt).prop('disabled', false);

    if (isOverrideLayer) {
        $status.html(`<i class="fa-solid fa-check"></i> Layer ${layer} has its own override.`);
        $sysLabel.text('(override)');
        $create.hide();
        $remove.show();
        $reset.show();
    } else if (layer === 0) {
        $status.html(`<i class="fa-solid fa-info"></i> Layer 0 — used for fresh turn summaries (one call per chat batch).`);
        $create.hide();
        $remove.hide();
        $reset.show();
    } else {
        $status.html(`<i class="fa-solid fa-info"></i> Layer 1 — compression prompt. Layer 2+ inherits this unless overridden.`);
        $create.hide();
        $remove.hide();
        $reset.show();
    }
}

function ensureMetadata() {
    if (!chat_metadata[MODULE_NAME]) {
        chat_metadata[MODULE_NAME] = {
            lastSummarizedIndex: -1,
            messagesSinceSummarization: 0,
        };
    }
    if (chat_metadata[MODULE_NAME].lastSummarizedIndex === undefined) {
        chat_metadata[MODULE_NAME].lastSummarizedIndex = -1;
    }
    if (chat_metadata[MODULE_NAME].messagesSinceSummarization === undefined) {
        chat_metadata[MODULE_NAME].messagesSinceSummarization = 0;
    }
}

function isGroupChat() {
    return !!getContext().groupId;
}

function getCharacterName() {
    const context = getContext();
    if (context.characterId === undefined) return null;
    return context.name2 || characters[this_chid]?.name || 'Character';
}

function getGroupMembers() {
    const context = getContext();
    if (!context.groupId) return [];
    const group = context.groups?.find(g => g.id === context.groupId);
    if (!group) return [];
    const activeMembers = group.members.filter(avatar => !group.disabled_members?.includes(avatar));
    return activeMembers.map(avatar => {
        const charIndex = characters.findIndex(c => c.avatar === avatar);
        const char = characters[charIndex];
        return char ? { name: char.name, avatar, charIndex } : null;
    }).filter(Boolean);
}

function getMemoryTargets() {
    if (isGroupChat()) {
        // Dedupe by avatar: the same character twice in a group shares one
        // attachment list, so a second target could only clobber the first.
        const seen = new Set();
        const targets = [];
        for (const m of getGroupMembers()) {
            if (seen.has(m.avatar)) continue;
            seen.add(m.avatar);
            targets.push({
                name: m.name,
                avatar: m.avatar,
                charIndex: m.charIndex,
                fileName: getMemoryFileNameForCharacter(m.name, m.avatar),
            });
        }
        // Disambiguate filename collisions (different characters whose names
        // sanitize to the same filename) with a deterministic avatar hash, so
        // each member gets their own memory file instead of silently sharing.
        const counts = {};
        for (const t of targets) counts[t.fileName] = (counts[t.fileName] || 0) + 1;
        for (const t of targets) {
            if (counts[t.fileName] > 1) {
                const hash = getStringHash(t.avatar).toString(36);
                t.fileName = `${t.fileName.replace(/\.md$/i, '')}-${hash}.md`;
            }
        }
        return targets;
    }
    const char = characters[this_chid];
    if (!char) return [];
    return [{
        name: char.name,
        avatar: char.avatar,
        charIndex: this_chid,
        fileName: getMemoryFileName(),
    }];
}

function getMemoryFileName() {
    const s = extension_settings[MODULE_NAME];
    const custom = s?.fileName;
    if (custom && custom !== DEFAULT_FILE_NAME) return custom;
    const charName = getCharacterName();
    if (!charName) return DEFAULT_FILE_NAME;
    // Per-character override — applies to 1:1 chats as well as group members
    // (the global override above still wins when set).
    const avatar = characters[this_chid]?.avatar;
    const perChar = avatar ? s?.characterFileNames?.[avatar] : '';
    if (perChar) return perChar;
    const safeName = charName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const perChat = s?.perChat;
    if (perChat) {
        const context = getContext();
        const chatId = context.chatId || 'default';
        return `${safeName}-chat-${chatId}-memories.md`;
    }
    return `${safeName}-memories.md`;
}

function getMemoryFileNameForCharacter(charName, avatar) {
    const s = extension_settings[MODULE_NAME];
    const customNames = s?.characterFileNames || {};
    if (customNames[avatar]) return customNames[avatar];
    const safeName = charName.replace(/[^a-zA-Z0-9_-]/g, '_');
    if (s?.perChat) {
        const context = getContext();
        const chatId = context.chatId || 'default';
        return `${safeName}-chat-${chatId}-memories.md`;
    }
    return `${safeName}-memories.md`;
}

function getSpeakerAvatar(msg) {
    if (!msg) return null;
    if (msg.is_user) return USER_AVATAR;
    if (msg.original_avatar) return msg.original_avatar;
    // ST stamps group character messages with force_avatar.
    if (msg.force_avatar) return msg.force_avatar;
    if (msg.name) {
        // Resolve within the current chat first — a same-named character
        // elsewhere must not hijack attribution.
        if (isGroupChat()) {
            const member = getGroupMembers().find(m => m.name === msg.name);
            if (member) return member.avatar;
        } else if (characters[this_chid]?.name === msg.name) {
            return characters[this_chid].avatar;
        }
        const char = characters.find(c => c.name === msg.name);
        if (char?.avatar) return char.avatar;
    }
    return null;
}

function findMemoryAttachmentForCharacter(avatar, fileName) {
    const attachments = extension_settings.character_attachments?.[avatar];
    if (!attachments) return null;
    return attachments.find(a => a.name === fileName) || null;
}

async function readMemoriesForCharacter(avatar, fileName) {
    const existing = findMemoryAttachmentForCharacter(avatar, fileName);
    if (!existing) return '';
    try {
        const attachment = await getFileAttachment(existing.url);
        if (attachment === undefined) return null;
        return attachment;
    } catch (err) {
        console.error(LOG_PREFIX, `Failed to read ${fileName} for ${avatar}:`, err);
        return null;
    }
}

async function writeMemoriesForCharacter(content, avatar, fileName) {
    if (!extension_settings.character_attachments) {
        extension_settings.character_attachments = {};
    }
    if (!extension_settings.character_attachments[avatar]) {
        extension_settings.character_attachments[avatar] = [];
    }

    // Empty content = "clear the file": remove the old attachment.
    if (!content || !content.trim()) {
        const existingEmpty = findMemoryAttachmentForCharacter(avatar, fileName);
        if (existingEmpty) {
            // silent=true => returns false instead of throwing on failure.
            const deleted = await deleteFileFromServer(existingEmpty.url, true);
            if (!deleted) {
                console.warn(LOG_PREFIX, `Failed to delete memory file ${existingEmpty.url} (orphaned copy may remain on server).`);
            }
            extension_settings.character_attachments[avatar] =
                extension_settings.character_attachments[avatar].filter(a => a.url !== existingEmpty.url);
            saveSettingsDebounced();
        }
        return;
    }

    // Upload the NEW content FIRST. uploadFileAttachment swallows errors
    // (returns undefined instead of throwing), so check the return value.
    const base64Data = convertTextToBase64(content);
    const slug = getStringHash(fileName);
    const ext = /\.md$/i.test(fileName || '') ? '.md' : '.txt';
    const uniqueFileName = `${Date.now()}_${slug}${ext}`;
    const fileUrl = await uploadFileAttachment(uniqueFileName, base64Data);

    if (!fileUrl || typeof fileUrl !== 'string') {
        console.error(LOG_PREFIX, `Upload failed for ${fileName} — old file preserved, aborting write.`);
        toastr.error(`Failed to save memory file "${fileName}". The previous version is intact.`, 'CharSummaryception');
        return;
    }

    // Only AFTER the new file is safely on the server do we delete the old one.
    const existing = findMemoryAttachmentForCharacter(avatar, fileName);
    if (existing) {
        // silent=true => returns false instead of throwing on failure.
        const deleted = await deleteFileFromServer(existing.url, true);
        if (!deleted) {
            console.warn(LOG_PREFIX, `Failed to delete old memory file ${existing.url} (orphaned copy may remain on server).`);
            toastr.warning(`Old file could not be deleted — orphaned copy may remain on server.`, 'CharSummaryception', { timeOut: 6000 });
        }
        extension_settings.character_attachments[avatar] =
            extension_settings.character_attachments[avatar].filter(a => a.url !== existing.url);
    }

    extension_settings.character_attachments[avatar].push({
        url: fileUrl,
        size: new TextEncoder().encode(content).length,
        name: fileName,
        created: Date.now(),
    });
    saveSettingsDebounced();
}

// Generation counter for updateStatusDisplay — guards against stale async closures.
let statusDisplayGen = 0;

// Status-bar memory-count cache, keyed by attachment URL. The URL changes on
// every write, so a cache hit means the file hasn't changed since the last
// count — avoids re-downloading every memory file on every rendered message.
const memoryCountCache = new Map();

function updateStatusDisplay() {
    ensureMetadata();
    const myGen = ++statusDisplayGen;
    const targets = getMemoryTargets();

    if (targets.length > 1) {
        const avatarHtml = targets.map(t =>
            `<img class="charSummary_groupAvatar" src="/thumbnail?type=avatar&file=${encodeURIComponent(t.avatar)}" alt="${escapeHtml(t.name)}" onerror="this.style.display='none'" />`
        ).join('');
        $('#charSummary_statFile').html(`Group: ${avatarHtml}`);
    } else if (targets.length === 1) {
        $('#charSummary_statFile').text(targets[0].fileName);
    } else {
        $('#charSummary_statFile').text('No character');
    }

    if (targets.length === 0) {
        $('#charSummary_statCount').text('0 memories');
    } else {
        let totalCount = 0;
        let settled = 0;
        const total = targets.length;
        const renderCount = () => {
            if (settled !== total || myGen !== statusDisplayGen) return;
            $('#charSummary_statCount').text(`${totalCount} memor${totalCount === 1 ? 'y' : 'ies'}`);
        };
        for (const target of targets) {
            const attachment = findMemoryAttachmentForCharacter(target.avatar, target.fileName);
            const cacheKey = `${target.avatar}|${target.fileName}|${attachment?.url || ''}`;
            if (memoryCountCache.has(cacheKey)) {
                totalCount += memoryCountCache.get(cacheKey);
                settled++;
                continue;
            }
            readMemoriesForCharacter(target.avatar, target.fileName).then(content => {
                // Bail if a newer updateStatusDisplay call has superseded us.
                if (myGen !== statusDisplayGen) return;
                const count = countMemories(parseMemories(content || ''));
                if (memoryCountCache.size > 100) memoryCountCache.clear();
                memoryCountCache.set(cacheKey, count);
                totalCount += count;
                settled++;
                renderCount();
            }).catch(() => {
                // Advance the settled counter even on failure so the display
                // isn't permanently stuck at its previous value.
                if (myGen !== statusDisplayGen) return;
                settled++;
                renderCount();
            });
        }
        // All-cache-hit path (no async reads needed).
        renderCount();
    }

    const msgsSince = chat_metadata[MODULE_NAME]?.messagesSinceSummarization || 0;
    const interval = extension_settings[MODULE_NAME]?.interval || 10;
    $('#charSummary_statProgress').text(`${msgsSince}/${interval} msgs`);

    const enabled = extension_settings[MODULE_NAME]?.enabled;
    $('#charSummary_autoPill').toggleClass('active', !!enabled);
}

function updateActivityLogDisplay() {
    const $container = $('#charSummary_dashActivity');
    if (!$container.length) return;
    if (activityLog.length === 0) {
        $container.html('<div class="charSummary_diagEmpty">No activity yet.</div>');
        return;
    }
    const entries = activityLog.slice(0, 8).map(e =>
        `<div class="charSummary_logEntry charSummary_log_${e.type}"><span class="charSummary_logTime">${e.timestamp}</span>${escapeHtml(e.message)}</div>`
    ).join('');
    $container.html(entries);
}

/**
 * Show a modal offering three ways to handle a large message backlog.
 * @param {number} overflowCount Unsummarized messages beyond the threshold.
 * @param {number} estimatedCalls Estimated LLM calls to process the entire backlog.
 * @returns {Promise<'full'|'skip'|'partial'|null>}
 */
function showCatchupDialog(overflowCount, estimatedCalls) {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'charSummary_catchupOverlay';
        overlay.innerHTML = `
            <div class="charSummary_catchupModal">
                <h3><i class="fa-solid fa-layer-group"></i> Backlog Detected</h3>
                <p>Detected <strong>${overflowCount} unsummarized messages</strong> (beyond your batch threshold).</p>
                <p>Processing all of them will require approximately <strong>${estimatedCalls} LLM call${estimatedCalls === 1 ? '' : 's'}</strong>.</p>
                <div class="charSummary_catchupButtons">
                    <button class="menu_button" data-choice="full"><i class="fa-solid fa-forward-fast"></i> Process Entire Backlog</button>
                    <button class="menu_button" data-choice="partial"><i class="fa-solid fa-play"></i> Just One Batch</button>
                    <button class="menu_button" data-choice="skip"><i class="fa-solid fa-forward-step"></i> Skip Backlog</button>
                </div>
                <small class="charSummary_helperText">You can stop a running summarization with the Stop button.</small>
            </div>
        `;
        document.body.appendChild(overlay);

        const close = (value) => {
            document.removeEventListener('keydown', onKeydown);
            overlay.remove();
            resolve(value);
        };

        // Escape dismisses the dialog (treated as cancel).
        const onKeydown = (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                close(null);
            }
        };
        document.addEventListener('keydown', onKeydown);

        overlay.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-choice]');
            if (btn) {
                close(btn.dataset.choice);
            } else if (e.target === overlay) {
                close(null);
            }
        });

        overlay.querySelector('[data-choice="full"]').focus();
    });
}

/**
 * Ask the user which range Summarize Now should cover.
 * @returns {Promise<'normal'|'all'|null>} 'normal' respects the protected tail,
 *   'all' summarizes up to the last message, null = cancelled.
 */
function showSummarizeModeDialog() {
    const protectedCount = extension_settings[MODULE_NAME]?.protectedMessages ?? 5;
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'charSummary_catchupOverlay';
        overlay.innerHTML = `
            <div class="charSummary_catchupModal">
                <h3><i class="fa-solid fa-wand-magic-sparkles"></i> Summarize Now</h3>
                <p>Choose how much of the chat to summarize.</p>
                <p>Your last <strong>${protectedCount}</strong> message${protectedCount === 1 ? '' : 's'} ${protectedCount === 1 ? 'is' : 'are'} normally kept as fresh context for the chat model.</p>
                <div class="charSummary_catchupButtons">
                    <button class="menu_button" data-choice="normal"><i class="fa-solid fa-play"></i> Summarize <small style="opacity:.7">(keep last ${protectedCount} protected)</small></button>
                    <button class="menu_button" data-choice="all"><i class="fa-solid fa-forward-fast"></i> Summarize All <small style="opacity:.7">(ignore protected, up to last message)</small></button>
                </div>
                <small class="charSummary_helperText">You can stop a running summarization with the Stop button.</small>
            </div>
        `;
        document.body.appendChild(overlay);

        const close = (value) => {
            document.removeEventListener('keydown', onKeydown);
            overlay.remove();
            resolve(value);
        };

        const onKeydown = (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                close(null);
            }
        };
        document.addEventListener('keydown', onKeydown);

        overlay.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-choice]');
            if (btn) {
                close(btn.dataset.choice);
            } else if (e.target === overlay) {
                close(null);
            }
        });

        overlay.querySelector('[data-choice="normal"]').focus();
    });
}

let isSummarizing = false;
let isPromoting = false;
let currentAbortController = null;

function abortSummarization() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
    // NOTE: the isSummarizing lock and the UI are released in
    // triggerSummarization's finally block — the pipeline keeps the lock
    // while it unwinds so its final per-character saves can't race a new run.
}

function setSummarizingUI(active) {
    $('#charSummary_summarizeNow').toggle(!active);
    $('#charSummary_stopSummarize').toggle(active);
    // Disable the Snippets button while summarizing — editing snippets during
    // a run risks clobbering the pipeline's writes (L1) and vice versa.
    $('#charSummary_snippetBtn').prop('disabled', active);
}

function onCharacterMessageRendered(messageIndex, type) {
    if (type === 'swipe' || type === 'continue' || type === 'regenerate') {
        if (extension_settings[MODULE_NAME]?.debugMode) console.log(`[CSC-DIAG] CHARACTER_MESSAGE_RENDERED idx=${messageIndex} type=${type} filtered=true (early return, counter untouched)`);
        return;
    }

    const context = getContext();
    if (context.characterId === undefined && !context.groupId) return;

    inheritPerception(messageIndex);
    injectPerceptionButton(messageIndex);

    if (!extension_settings[MODULE_NAME]?.enabled) return;

    ensureMetadata();
    const beforeCount = chat_metadata[MODULE_NAME].messagesSinceSummarization || 0;
    chat_metadata[MODULE_NAME].messagesSinceSummarization = beforeCount + 1;
    saveMetadataDebounced();
    updateStatusDisplay();

    if (extension_settings[MODULE_NAME]?.debugMode) console.log(`[CSC-DIAG] CHARACTER_MESSAGE_RENDERED idx=${messageIndex} type=${type || 'undefined'} filtered=false oldCounter=${beforeCount} newCounter=${chat_metadata[MODULE_NAME].messagesSinceSummarization}`);

    const count = chat_metadata[MODULE_NAME].messagesSinceSummarization;
    const interval = extension_settings[MODULE_NAME]?.interval || 10;

    if (count >= interval) {
        triggerSummarization();
    }
}

/**
 * User-message counterpart to onCharacterMessageRendered. Combined with the
 * character handler, the interval means "every N total messages" rather
 * than "every N character turns". Also runs perception inheritance so a
 * private user message followed by another user message doesn't leak to
 * the whole group via default "everyone perceives".
 */
function onUserMessageRendered(messageIndex) {
    const context = getContext();
    if (context.characterId === undefined && !context.groupId) return;

    inheritPerception(messageIndex);

    if (!extension_settings[MODULE_NAME]?.enabled) return;

    ensureMetadata();
    const beforeCount = chat_metadata[MODULE_NAME].messagesSinceSummarization || 0;
    chat_metadata[MODULE_NAME].messagesSinceSummarization = beforeCount + 1;
    saveMetadataDebounced();
    updateStatusDisplay();

    if (extension_settings[MODULE_NAME]?.debugMode) console.log(`[CSC-DIAG] USER_MESSAGE_RENDERED idx=${messageIndex} oldCounter=${beforeCount} newCounter=${chat_metadata[MODULE_NAME].messagesSinceSummarization}`);

    const count = chat_metadata[MODULE_NAME].messagesSinceSummarization;
    const interval = extension_settings[MODULE_NAME]?.interval || 10;

    if (count >= interval) {
        triggerSummarization();
    }
}

function onMessageDeleted(newChatLength) {
    if (!extension_settings[MODULE_NAME]?.enabled) return;

    ensureMetadata();
    const meta = chat_metadata[MODULE_NAME];
    const chat = getContext().chat;
    const chatLength = Number.isFinite(newChatLength) ? newChatLength : (chat?.length ?? 0);

    // Clamp lastSummarizedIndex if message deletion has left it pointing
    // beyond the chat. Otherwise the next run would silently bail at the
    // `startIdx > endIdx` guard, stalling auto-summarization indefinitely.
    const currentIdx = meta.lastSummarizedIndex ?? -1;
    if (currentIdx >= chatLength) {
        const clamped = chatLength > 0 ? chatLength - 1 : -1;
        meta.lastSummarizedIndex = clamped;
        logActivity(
            `Clamped lastSummarizedIndex from ${currentIdx} to ${clamped} after message deletion`,
            'warning',
        );
    }

    // Decrement the counter by 1 if the deleted message was likely in the
    // unsummarized tail (above lastSummarizedIndex). The event only passes
    // newChatLength, not the deleted index, so we assume the typical case
    // (last message removed via delete/regenerate/swipe). Wrong heuristics
    // (middle-of-chat deletes) undercount slightly and self-correct on next run.
    const lastIdx = meta.lastSummarizedIndex ?? -1;
    const deletedIdx = chatLength; // msg at this index was removed; now out of range
    const beforeCounter = meta.messagesSinceSummarization || 0;
    let decrementReason = 'no-change (deleted idx <= lastSummarized)';
    if (deletedIdx > lastIdx) {
        meta.messagesSinceSummarization = Math.max(0, beforeCounter - 1);
        decrementReason = 'decremented (deleted idx > lastSummarized)';
    }

    if (extension_settings[MODULE_NAME]?.debugMode) console.log(`[CSC-DIAG] MESSAGE_DELETED newLen=${chatLength} lastSummarized=${lastIdx} deletedIdx=${deletedIdx} oldCounter=${beforeCounter} newCounter=${meta.messagesSinceSummarization} decision=${decrementReason}`);

    saveMetadataDebounced();
    updateStatusDisplay();
}

async function triggerSummarization(opts = {}) {
    const { manual = false, ignoreProtected = false } = opts;
    if (isSummarizing || isPromoting) {
        logActivity('Already summarizing, skipping', 'warning');
        return;
    }

    // Claim the lock BEFORE the first await.
    isSummarizing = true;

    try {
        const context = getContext();
        if (!context.chat || context.chat.length === 0) return;

        ensureMetadata();

        // Remember which chat this run belongs to. If the user switches chats
        // mid-run, memory files still get their content (targets are captured
        // up front), but metadata writes must be skipped — chat_metadata will
        // then belong to the NEW chat.
        const runChatId = context.chatId ?? null;

        const lastSummarized = chat_metadata[MODULE_NAME].lastSummarizedIndex ?? -1;
        const chatLength = context.chat.length;
        const protectedMessages = extension_settings[MODULE_NAME]?.protectedMessages ?? 5;
        const startIdx = lastSummarized + 1;
        const endIdx = ignoreProtected ? (chatLength - 1) : (chatLength - 1 - protectedMessages);

        if (startIdx > endIdx) return;

        const batchSize = endIdx - startIdx + 1;
        const maxBatch = (extension_settings[MODULE_NAME]?.interval || 10) * 3;
        const interval = extension_settings[MODULE_NAME]?.interval || 10;

        // Auto-triggers process exactly one interval per run. Manual triggers
        // keep the wider maxBatch range so the catch-up dialog's full/partial
        // choice stays meaningful.
        let actualEnd = manual
            ? Math.min(endIdx, startIdx + maxBatch - 1)
            : Math.min(endIdx, startIdx + interval - 1);

        // Create the abort controller and switch the UI BEFORE any await
        // (including the catch-up dialog) so Stop works during the dialog too.
        currentAbortController = new AbortController();
        setSummarizingUI(true);

        // For manual triggers, offer a catch-up choice when backlog > 2x threshold.
        if (manual && batchSize > interval * 2) {
            const overflow = batchSize;
            // Estimate: extraction pre-pass (one call per partial-perception
            // message) PLUS per-character summarization calls (one per
            // character per sub-batch).
            const targets = getMemoryTargets();
            const chatArr = context.chat;
            const speakerFn = (m) => getSpeakerAvatar(m);
            let extractionCalls = 0;
            if (Array.isArray(chatArr)) {
                for (let i = startIdx; i <= endIdx; i++) {
                    if (chatArr[i] && hasPartialPerception(chatArr[i], targets, speakerFn)) {
                        extractionCalls++;
                    }
                }
            }
            const subBatchCount = Math.max(1, Math.ceil(batchSize / interval));
            const summaryCalls = Math.max(1, (targets.length || 1)) * subBatchCount;
            const estimatedCalls = extractionCalls + summaryCalls;
            const choice = await showCatchupDialog(overflow, estimatedCalls);
            if (currentAbortController.signal.aborted) {
                logActivity('Summarization aborted by user', 'warning');
                return;
            }
            if (choice === null || choice === 'skip') {
                if (choice === 'skip') {
                    // Treat skip as resolving the backlog without summarizing.
                    chat_metadata[MODULE_NAME].lastSummarizedIndex = endIdx;
                    chat_metadata[MODULE_NAME].messagesSinceSummarization = 0;
                    saveMetadataDebounced();
                    logActivity(`Skipped backlog of ${batchSize} messages (marked ${startIdx}-${endIdx} as summarized)`, 'warning');
                    updateStatusDisplay();
                }
                return;
            }
            if (choice === 'full') {
                actualEnd = endIdx;
            } else if (choice === 'partial') {
                actualEnd = Math.min(endIdx, startIdx + interval - 1);
            }
        } else if (!manual && batchSize > interval) {
            // Auto-trigger capped at one interval; remainder carries in the counter.
            logActivity(
                `Backlog of ${batchSize} msgs; processing first ${interval}. Remainder (${batchSize - interval}) defers to next trigger.`,
                'info',
            );
        } else if (manual && batchSize > maxBatch) {
            logActivity(`Large backlog: ${batchSize} messages. Processing first ${maxBatch}.`, 'warning');
        }

        toastr.info(`Summarizing messages ${startIdx}-${actualEnd}…`, 'CharSummaryception', { timeOut: 3000 });

        const deps = makeDeps();
        const result = await summarizeBatch(deps, startIdx, actualEnd);

        const wasAborted = !!currentAbortController?.signal.aborted;

        // If the user switched chats during the run, chat_metadata now belongs
        // to the new chat — don't write the old chat's progress into it. The
        // old chat re-derives progress via range-based dedup on its next run.
        if ((getContext().chatId ?? null) !== runChatId) {
            logActivity('Chat changed during summarization — memories saved, index update skipped for the old chat', 'warning');
            updateStatusDisplay();
            return;
        }

        // Commit partial progress whenever ANY sub-batch was persisted, even
        // on abort. Advance to lastCommittedEnd (the MIN across characters);
        // range-based dedup inside summarizeBatch absorbs any residual overlaps.
        const previousIdx = chat_metadata[MODULE_NAME].lastSummarizedIndex ?? -1;
        if (result.lastCommittedEnd != null && result.lastCommittedEnd > previousIdx) {
            chat_metadata[MODULE_NAME].lastSummarizedIndex = result.lastCommittedEnd;
            const newChatLength = context.chat.length;
            const remaining = Math.max(0, newChatLength - 1 - protectedMessages - result.lastCommittedEnd);
            chat_metadata[MODULE_NAME].messagesSinceSummarization = remaining;
            saveMetadataDebounced();
        }

        if (wasAborted) {
            const committedDesc = result.lastCommittedEnd != null
                ? `committed up to index ${result.lastCommittedEnd}`
                : 'no sub-batches completed';
            logActivity(`Summarization aborted by user — ${committedDesc}`, 'warning');
            updateStatusDisplay();
            return;
        }

        // Non-aborted completion: surface totals. Index already advanced;
        // fall back to the "no memories" path only when literally nothing
        // was produced across all characters.
        if (result.totalSummaries > 0) {
            logActivity(`Summarized ${result.totalSummaries} memories for ${result.charactersProcessed} character(s)`, 'success');
            toastr.success(`${result.totalSummaries} memories saved`, 'CharSummaryception', { timeOut: 3000 });
        } else {
            logActivity('No new memories from this batch — index not advanced', 'warning');
            toastr.warning('Summarization produced no memories. Backlog retained — check connection/settings.', 'CharSummaryception');
        }

        updateStatusDisplay();
    } catch (err) {
        console.error(LOG_PREFIX, 'Summarization failed:', err);
        logActivity(`Summarization failed: ${err.message}`, 'error');
        toastr.error('Summarization failed. Check console.', 'CharSummaryception');
    } finally {
        isSummarizing = false;
        currentAbortController = null;
        setSummarizingUI(false);
    }
}

function makeDeps() {
    return {
        settings: () => extension_settings[MODULE_NAME],
        getChat: () => getContext().chat,
        getTargets: () => getMemoryTargets(),
        getSpeakerAvatarFn: (msg) => getSpeakerAvatar(msg),
        getUserName: () => getContext().name1 || 'User',
        readFile: (avatar, fileName) => readMemoriesForCharacter(avatar, fileName),
        writeFile: (content, avatar, fileName) => writeMemoriesForCharacter(content, avatar, fileName),
        abortSignal: () => currentAbortController?.signal || null,
        log: (msg, type = 'info') => logActivity(msg, type),
        debug: (...args) => log(...args),
        trace: (...args) => trace(...args),
    };
}

async function repairIfBranched() {
    const { chat } = getContext();
    const meta = chat_metadata[MODULE_NAME];
    if (!chat || chat.length === 0 || !meta) return;
    // ?? not || — index 0 is a valid summarized position.
    if ((meta.lastSummarizedIndex ?? -1) < 0) return;
    // Branch detected: metadata references a message index beyond the current chat.
    if (meta.lastSummarizedIndex < chat.length) return;
    const oldIdx = meta.lastSummarizedIndex;
    meta.lastSummarizedIndex = chat.length - 1;
    meta.messagesSinceSummarization = 0;
    saveMetadataDebounced();
    logActivity(`Branch detected — reset summarized index from ${oldIdx} to ${meta.lastSummarizedIndex}`, 'warning');
    toastr.info('Branch detected — stale summary data was reset.', 'CharSummaryception');

    // Memory blocks summarized from the abandoned branch still exist and
    // would keep being retrieved as memories of events that no longer
    // happened. Offer to drop blocks whose range starts beyond the current
    // chat (blocks without a range can't be attributed and are kept).
    try {
        const confirmPurge = await callGenericPopup(
            `This chat was branched from a longer version. Memory blocks summarizing messages beyond #${chat.length} came from the abandoned branch.<br><br>Remove those stale blocks from the memory file(s)?`,
            POPUP_TYPE.CONFIRM,
            '',
            { okButton: 'Remove stale blocks', cancelButton: 'Keep' },
        );
        if (!confirmPurge) return;

        const targets = getMemoryTargets();
        let removedTotal = 0;
        for (const target of targets) {
            const content = await readMemoriesForCharacter(target.avatar, target.fileName);
            if (content === null) continue; // transient read error — leave the file alone
            const blocks = parseMemories(content || '');
            const kept = blocks.filter(b => !(Array.isArray(b.range) && b.range[0] > chat.length - 1));
            const removed = blocks.length - kept.length;
            if (removed > 0) {
                await writeMemoriesForCharacter(serializeMemories(kept), target.avatar, target.fileName);
                removedTotal += removed;
            }
        }
        if (removedTotal > 0) {
            logActivity(`Purged ${removedTotal} stale memory block(s) from the abandoned branch`, 'success');
            toastr.success(`Removed ${removedTotal} stale memory block(s).`, 'CharSummaryception');
            updateStatusDisplay();
        } else {
            toastr.info('No stale memory blocks found.', 'CharSummaryception');
        }
    } catch (err) {
        console.warn(LOG_PREFIX, 'Branch purge failed:', err);
    }
}

async function onChatChanged() {
    const context = getContext();
    const chatId = context.chatId || '(none)';
    const charName = getCharacterName() || '(none)';
    const msgCount = context.chat ? context.chat.length : 0;

    logActivity(`Chat changed: "${charName}" chat=${chatId} (${msgCount} messages)`);

    ensureMetadata();

    await repairIfBranched();

    const lastIdx = chat_metadata[MODULE_NAME].lastSummarizedIndex ?? -1;
    if (lastIdx >= 0 && msgCount > 0) {
        try {
            const targets = getMemoryTargets();
            const contents = await Promise.all(
                targets.map(t => readMemoriesForCharacter(t.avatar, t.fileName)),
            );
            // If ANY read failed (null), we can't determine whether memories
            // exist — skip the auto-reset so a transient read error doesn't
            // nuke lastSummarizedIndex and trigger re-summarization from 0.
            if (contents.some(c => c === null)) {
                logActivity('Skipped lastSummarizedIndex check — one or more memory files unreadable', 'warning');
            } else {
                const hasAnyMemories = contents.some(c => parseMemories(c || '').length > 0);
                if (!hasAnyMemories) {
                    chat_metadata[MODULE_NAME].lastSummarizedIndex = -1;
                    saveMetadataDebounced();
                    logActivity('Auto-reset lastSummarizedIndex: memory files are empty', 'warning');
                }
            }
        } catch { /* ignore */ }
    }

    updateStatusDisplay();
    updateActivityLogDisplay();

    // Fallback bulk pass in case the DOM observer hasn't picked up the
    // newly rendered messages yet.
    schedulePerceptionScan();
}

function labelForAvatar(avatar, targets) {
    if (avatar === USER_AVATAR) return 'User';
    const t = targets.find(t => t.avatar === avatar);
    return t?.name || avatar;
}

function buildSceneRowHtml(perceiver, speakerAvatar, perception) {
    const entry = perception[speakerAvatar];
    // No entry = everyone perceives the speaker (default): all boxes checked.
    const seesChecked = !entry || (Array.isArray(entry.seenBy) && entry.seenBy.includes(perceiver.avatar));    const hearsChecked = !entry || (Array.isArray(entry.heardBy) && entry.heardBy.includes(perceiver.avatar));

    const cell = (field, checked) => `
        <td class="charSummary_sceneCell" data-cell-label="${field === 'hears' ? 'Hears' : 'Sees'}">
            <div class="charSummary_sceneChips">
                <label class="checkbox_label charSummary_sceneChip ${checked ? 'checked' : ''}">
                    <input type="checkbox" data-field="${field}" data-perceiver="${escapeAttr(perceiver.avatar)}" ${checked ? 'checked' : ''} />
                </label>
            </div>
        </td>`;

    return `
        <tr class="charSummary_sceneRow" data-perceiver="${escapeAttr(perceiver.avatar)}">
            <td class="charSummary_scenePerceiver" data-cell-label="Character">
                <div class="charSummary_scenePerceiverMain">
                    <img class="charSummary_sceneAvatar" src="/thumbnail?type=avatar&file=${encodeURIComponent(perceiver.avatar)}" alt="" onerror="this.style.display='none'" />
                    <span>${escapeHtml(perceiver.name)}</span>
                </div>
            </td>
            ${cell('sees', seesChecked)}
            ${cell('hears', hearsChecked)}
        </tr>
    `;
}

// State: which message the drawer is currently editing. null when no message
// is bound.
let drawerMessageId = null;

function buildSceneStateHtml(targets, messageIndex) {
    const chat = getContext().chat;
    const msg = Number.isInteger(messageIndex) ? chat?.[messageIndex] : null;
    const perception = msg?.extra?.perception || {};
    const speakerAvatar = msg ? getSpeakerAvatar(msg) : null;

    const perceivers = (speakerAvatar ? targets.filter(t => t.avatar !== speakerAvatar) : targets);
    const rows = perceivers.map(t => buildSceneRowHtml(t, speakerAvatar, perception)).join('');

    const entry = speakerAvatar ? perception[speakerAvatar] : undefined;
    const perceivingCount = perceivers.filter(t => {
        if (!entry) return true;
        return (Array.isArray(entry.seenBy) && entry.seenBy.includes(t.avatar))
            || (Array.isArray(entry.heardBy) && entry.heardBy.includes(t.avatar));
    }).length;

    if (msg) {
        const userName = getContext().name1 || 'User';
        const speakerName = msg.is_user ? userName : (msg.name || 'Character');
        const snippet = (stripNonDiegetic(msg.mes) || '').slice(0, 60).replace(/\s+/g, ' ').trim();
        $('#charSummary_sceneDrawer .charSummary_drawerTitle').html(
            `#${messageIndex + 1} ${escapeHtml(speakerName)}${snippet ? `: <span class="charSummary_drawerSnippet">${escapeHtml(snippet)}</span>` : ''}`
        );
    }

    const speakerLabel = speakerAvatar ? labelForAvatar(speakerAvatar, targets) : '—';

    return `
        <div class="charSummary_sceneHeading">Who can perceive <b>${escapeHtml(speakerLabel)}</b>?</div>
        <div class="charSummary_sceneSummary">
            <span><i class="fa-solid fa-users fa-sm"></i> ${perceivers.length} perceiver${perceivers.length === 1 ? '' : 's'}</span>
            <span><i class="fa-solid fa-eye fa-sm"></i> ${perceivingCount} perceiving</span>
        </div>
        <div class="charSummary_sceneTableWrap">
            <table class="charSummary_sceneTable">
                <thead><tr><th>Character</th><th>Sees</th><th>Hears</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
        <small class="charSummary_helperText">
            Uncheck a box to hide this message from that character. All checked = everyone perceives it (the default). Saved automatically.
        </small>
    `;
}

function renderSceneDrawerBody() {
    const $body = $('#charSummary_sceneBody');
    if (!$body.length) return;
    const targets = getMemoryTargets();
    if (targets.length === 0) {
        $body.html('<div class="charSummary_diagEmpty">Open a character or group chat to edit perception.</div>');
        return;
    }
    const chat = getContext().chat;
    if (!Number.isInteger(drawerMessageId) || !chat?.[drawerMessageId]) {
        $body.html('<div class="charSummary_diagEmpty">Use the group icon on a message to edit who perceived it.</div>');
        return;
    }
    $body.html(buildSceneStateHtml(targets, drawerMessageId));
}

function persistSceneDrawerEdits() {
    const $drawer = $('#charSummary_sceneDrawer');
    if (!$drawer.length) return;
    if (!Number.isInteger(drawerMessageId)) return;
    const chat = getContext().chat;
    const msg = chat?.[drawerMessageId];
    if (!msg) return;

    const targets = getMemoryTargets();
    const speakerAvatar = getSpeakerAvatar(msg);
    const perception = {};

    if (speakerAvatar) {
        const perceivers = targets.filter(t => t.avatar !== speakerAvatar);
        const seenBy = [];
        const heardBy = [];
        $drawer.find('input[data-field="sees"]:checked').each(function () { seenBy.push($(this).attr('data-perceiver')); });
        $drawer.find('input[data-field="hears"]:checked').each(function () { heardBy.push($(this).attr('data-perceiver')); });

        const isDefault = perceivers.every(t => seenBy.includes(t.avatar))
            && perceivers.every(t => heardBy.includes(t.avatar));
        if (!isDefault) {
            perception[speakerAvatar] = { seenBy, heardBy };
        }

        log('[PERC] drawer persist', {
            messageId: drawerMessageId,
            speaker: speakerAvatar,
            perceivers: perceivers.map(t => t.avatar),
            seenBy,
            heardBy,
            isDefault,
        });
    } else {
        log('[PERC] drawer persist', { messageId: drawerMessageId, speaker: speakerAvatar });
    }

    if (!msg.extra || typeof msg.extra !== 'object') msg.extra = {};
    msg.extra.perception = perception;
    saveChatDebounced();

    const entry = perception[speakerAvatar];
    const perceivers = speakerAvatar ? targets.filter(t => t.avatar !== speakerAvatar) : targets;
    const perceivingCount = perceivers.filter(t => {
        if (!entry) return true;
        return (Array.isArray(entry.seenBy) && entry.seenBy.includes(t.avatar))
            || (Array.isArray(entry.heardBy) && entry.heardBy.includes(t.avatar));
    }).length;
    $drawer.find('.charSummary_sceneSummary').html(`
        <span><i class="fa-solid fa-users fa-sm"></i> ${perceivers.length} perceiver${perceivers.length === 1 ? '' : 's'}</span>
        <span><i class="fa-solid fa-eye fa-sm"></i> ${perceivingCount} perceiving</span>
    `);

    const hasCustom = !!entry;
    updatePerceptionButtonIndicator(drawerMessageId, hasCustom);

    logActivity(`Perception updated for message #${drawerMessageId + 1} (${Object.keys(perception).length} entries)`);
}

function applyDrawerPosition() {
    const $drawer = $('#charSummary_sceneDrawer');
    if (!$drawer.length || !$drawer.hasClass('open')) return;
    const topBar = document.getElementById('top-settings-holder');
    const topOffset = topBar ? topBar.getBoundingClientRect().bottom : 0;
    const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
    $drawer.css({
        top: topOffset + 'px',
        height: (vh - topOffset) + 'px',
    });
}

let drawerResizeHandler = null;
let drawerViewportHandler = null;

function toggleSceneDrawer() {
    const $drawer = $('#charSummary_sceneDrawer');
    const isOpen = $drawer.hasClass('open');
    const shouldOpen = !isOpen;

    if (shouldOpen) {
        applyDrawerPosition();
        renderSceneDrawerBody();
        if (!drawerResizeHandler) {
            drawerResizeHandler = () => applyDrawerPosition();
            window.addEventListener('resize', drawerResizeHandler);
            window.addEventListener('orientationchange', drawerResizeHandler);
        }
        if (window.visualViewport && !drawerViewportHandler) {
            drawerViewportHandler = () => applyDrawerPosition();
            window.visualViewport.addEventListener('resize', drawerViewportHandler);
        }
    } else {
        if (drawerResizeHandler) {
            window.removeEventListener('resize', drawerResizeHandler);
            window.removeEventListener('orientationchange', drawerResizeHandler);
            drawerResizeHandler = null;
        }
        if (drawerViewportHandler && window.visualViewport) {
            window.visualViewport.removeEventListener('resize', drawerViewportHandler);
            drawerViewportHandler = null;
        }
    }

    $drawer.toggleClass('open', shouldOpen);
}

/**
 * Open the perception drawer bound to a specific message.
 * Re-renders if already open, else toggles it open.
 */
function openPerceptionForMessage(messageId) {
    drawerMessageId = messageId;
    const $drawer = $('#charSummary_sceneDrawer');
    if ($drawer.hasClass('open')) {
        renderSceneDrawerBody();
    } else {
        toggleSceneDrawer();
    }
}

/**
 * Inject a perception button into the message's extra-buttons container.
 * Clicking it opens the perception drawer for that message. Safe to call
 * repeatedly: on an existing button it just refreshes the indicator.
 */
function injectPerceptionButton(messageId) {
    const $mes = $(`.mes[mesid="${messageId}"]`);
    if (!$mes.length) return;
    const $container = $mes.find('.extraMesButtons');
    if (!$container.length) return;

    const chat = getContext().chat;
    const msg = chat?.[messageId];
    const speaker = getSpeakerAvatar(msg);
    const entry = speaker ? msg?.extra?.perception?.[speaker] : undefined;
    const hasCustom = !!entry;

    const $existing = $container.find('.csc_perceptionBtn');
    if ($existing.length) {
        // Already injected — just refresh the indicator.
        $existing.toggleClass('csc_perceptionBtn_custom', hasCustom);
        return;
    }

    const $btn = $(`<div class="mes_button csc_perceptionBtn ${hasCustom ? 'csc_perceptionBtn_custom' : ''}" title="Edit who can see and hear this message"><i class="fa-solid fa-users fa-sm"></i></div>`);
    $btn.on('click', function (e) {
        e.stopPropagation();
        openPerceptionForMessage(messageId);
    });
    $container.append($btn);
}

/**
 * Scan every message currently in the DOM and inject the perception button
 * into any message missing it. Idempotent.
 */
function injectPerceptionButtonsAll() {
    const chat = getContext().chat;
    if (!Array.isArray(chat)) return;
    $('#chat .mes').each(function () {
        const mesid = Number(this.getAttribute('mesid'));
        if (!Number.isInteger(mesid)) return;
        const msg = chat[mesid];
        if (!msg) return;
        injectPerceptionButton(mesid);
    });
}

// Debounce token for the MutationObserver scan.
let perceptionScanTimer = null;
function schedulePerceptionScan() {
    if (perceptionScanTimer) return;
    perceptionScanTimer = setTimeout(() => {
        perceptionScanTimer = null;
        injectPerceptionButtonsAll();
    }, 50);
}

/**
 * Watch the chat container for added message elements and inject the
 * perception button. The render events don't reliably fire on load/switch,
 * so observing the DOM directly catches history load, chat switch, swipes,
 * and edits.
 */
function startPerceptionButtonObserver() {
    const target = document.getElementById('chat');
    if (!target) return;
    const observer = new MutationObserver(muts => {
        for (const m of muts) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                // Only react when a whole message (or its button container)
                // is added — not to streaming text appended inside a message.
                if (node.matches('.mes') ||
                    node.matches('.extraMesButtons') ||
                    node.querySelector('.mes')) {
                    schedulePerceptionScan();
                    return;
                }
            }
        }
    });
    observer.observe(target, { childList: true, subtree: true });
    // Initial pass for messages already in the DOM at startup.
    injectPerceptionButtonsAll();}

/**
 * Refresh the custom-perception indicator on a message's perception button.
 */
function updatePerceptionButtonIndicator(messageId, hasCustom) {
    const $btn = $(`.mes[mesid="${messageId}"] .csc_perceptionBtn`);
    $btn.toggleClass('csc_perceptionBtn_custom', !!hasCustom);
}

/**
 * When a new message renders, inherit the extra.perception from the most
 * recent prior message from the SAME speaker. If that message is at default
 * (no perception map), inheritance stops and the new message stays default.
 * The audience is re-keyed to the new speaker (assumption: the same set of
 * perceivers is still present in the scene). The speaker itself is stripped
 * from its own seenBy/heardBy lists when re-keying.
 */
function inheritPerception(messageId) {
    const chat = getContext().chat;
    if (!Array.isArray(chat) || !Number.isInteger(messageId)) return;
    const msg = chat[messageId];
    if (!msg) return;
    const existing = msg.extra?.perception;
    const currentSpeaker = getSpeakerAvatar(msg);
    log('[PERC] inherit entry', {
        messageId,
        currentSpeaker,
        hasOwnPerception: !!(existing && Object.keys(existing).length > 0),
    });
    if (existing && Object.keys(existing).length > 0) {
        log('[PERC]   -> abort: message already has perception');
        return;
    }
    if (!currentSpeaker) {
        log('[PERC]   -> abort: no currentSpeaker');
        return;
    }

    for (let i = messageId - 1; i >= 0; i--) {
        const prev = chat[i];
        if (!prev) continue;
        const prevSpeaker = getSpeakerAvatar(prev);
        const prevPerception = prev.extra?.perception;
        const prevKeys = prevPerception ? Object.keys(prevPerception) : [];

        if (prevSpeaker !== currentSpeaker) {
            log(`[PERC]   [${i}] speaker="${prevSpeaker}" != current -> skip`, { sameSpeaker: false, percKeys: prevKeys });
            continue;
        }
        if (!prevPerception || prevKeys.length === 0) {
            log(`[PERC]   [${i}] same speaker, NO perception -> STOP (stays default)`, { percKeys: prevKeys });
            return;
        }

        const entry = prevPerception[currentSpeaker];
        if (!entry) {
            log(`[PERC]   [${i}] same speaker, no entry for "${currentSpeaker}" -> skip`, { percKeys: prevKeys });
            continue;
        }

        const cloned = JSON.parse(JSON.stringify(entry));
        if (Array.isArray(cloned.seenBy)) {
            cloned.seenBy = cloned.seenBy.filter(a => a !== currentSpeaker);
        }
        if (Array.isArray(cloned.heardBy)) {
            cloned.heardBy = cloned.heardBy.filter(a => a !== currentSpeaker);
        }

        const audienceSize = (Array.isArray(cloned.seenBy) ? cloned.seenBy.length : 0)
            + (Array.isArray(cloned.heardBy) ? cloned.heardBy.length : 0);
        if (!msg.extra || typeof msg.extra !== 'object') msg.extra = {};
        msg.extra.perception = { [currentSpeaker]: cloned };
        saveChatDebounced();
        log(`[PERC]   [${i}] INHERIT -> wrote perception${audienceSize === 0 ? ' (nobody perceives)' : ''}`, {
            fromSpeaker: prevSpeaker,
            seenBy: cloned.seenBy,
            heardBy: cloned.heardBy,
        });
        return;
    }
    log('[PERC]   -> no same-speaker customized message found -> stays default');
}

function onSceneCheckboxChange(checkbox) {
    const $cb = $(checkbox);
    $cb.closest('.charSummary_sceneChip').toggleClass('checked', checkbox.checked);
    persistSceneDrawerEdits();
}

async function onSummarizeNowClick() {
    if (isSummarizing) {
        toastr.info('Already summarizing…', 'CharSummaryception');
        return;
    }
    const choice = await showSummarizeModeDialog();
    if (choice === null) return;
    const ignoreProtected = (choice === 'all');
    if (ignoreProtected) logActivity('Summarize All: ignoring protected tail', 'info');
    toastr.info('Running summarization…', 'CharSummaryception');
    await triggerSummarization({ manual: true, ignoreProtected });
}

async function showSettingsModal() {
    const s = extension_settings[MODULE_NAME];

    const sourceOptions = [
        { value: 'default', label: 'Default (Main API)' },
        { value: 'profile', label: 'Connection Profile' },
        { value: 'openai', label: 'OpenAI Compatible' },
        { value: 'openrouter', label: 'OpenRouter' },
    ].map(o => `<option value="${o.value}" ${o.value === s.connectionSource ? 'selected' : ''}>${o.label}</option>`).join('');

    const charFileRows = (!Array.isArray(characters) || characters.length === 0)
        ? '<div class="charSummary_diagEmpty">No characters loaded.</div>'
        : characters.map(c => {
            if (!c?.avatar || !c.name) return '';
            const safeName = c.name.replace(/[^a-zA-Z0-9_-]/g, '_');
            const defaultFn = `${safeName}-memories.md`;
            const custom = s.characterFileNames?.[c.avatar] || '';
            return `
                <div class="csc_fileCharRow" data-avatar="${escapeAttr(c.avatar)}">
                    <img class="csc_fileCharAvatar" src="/thumbnail?type=avatar&file=${encodeURIComponent(c.avatar)}" alt="" onerror="this.style.display='none'" />
                    <div class="csc_fileCharMain">
                        <div class="csc_fileCharName">${escapeHtml(c.name)}</div>
                        <input class="text_pole csc_fileCharInput" type="text" data-avatar="${escapeAttr(c.avatar)}" value="${escapeAttr(custom)}" placeholder="${escapeAttr(defaultFn)}" />
                    </div>
                </div>`;
        }).join('');

    const body = `
        <div class="csc_tabs">
            <div class="csc_tabBar">
                <button type="button" class="csc_tab active" data-tab="summarization"><i class="fa-solid fa-layer-group fa-sm"></i> Summarization</button>
                <button type="button" class="csc_tab" data-tab="compression"><i class="fa-solid fa-compress fa-sm"></i> Compression</button>
                <button type="button" class="csc_tab" data-tab="connection"><i class="fa-solid fa-plug fa-sm"></i> Connection</button>
                <button type="button" class="csc_tab" data-tab="files"><i class="fa-solid fa-file-lines fa-sm"></i> Files</button>
                <button type="button" class="csc_tab" data-tab="prompts"><i class="fa-solid fa-scroll fa-sm"></i> Prompts</button>
                <button type="button" class="csc_tab" data-tab="diagnostics"><i class="fa-solid fa-wrench fa-sm"></i> Diagnostics</button>
            </div>

            <div class="csc_tabPanel active" data-panel="summarization">
                <h4 class="charSummary_modalSectionTitle">Summarization</h4>
                <div class="charSummary_sliderRow">
                    <label><small>Batch size</small></label>
                    <small class="charSummary_helperText">How many new messages trigger a summarization run. Lower = fresher memories, but more LLM calls.</small>
                    <input class="neo-range-slider" type="range" id="csc_modal_interval" min="3" max="50" step="1" value="${s.interval}" />
                    <div class="wide100p">
                        <input class="neo-range-input" type="number" min="3" max="50" step="1"
                               data-for="csc_modal_interval" id="csc_modal_intervalCounter" value="${s.interval}" />
                    </div>
                </div>
                <div class="charSummary_sliderRow">
                    <label><small>Protected messages</small></label>
                    <small class="charSummary_helperText">Recent messages kept out of the summarizer so the chat model still sees them as fresh context.</small>
                    <input class="neo-range-slider" type="range" id="csc_modal_protectedMessages" min="3" max="20" step="1" value="${s.protectedMessages}" />
                    <div class="wide100p">
                        <input class="neo-range-input" type="number" min="3" max="20" step="1"
                               data-for="csc_modal_protectedMessages" id="csc_modal_protectedMessagesCounter" value="${s.protectedMessages}" />
                    </div>
                </div>
            </div>

            <div class="csc_tabPanel" data-panel="compression">
                <h4 class="charSummary_modalSectionTitle">Layer Compression</h4>
                <small class="charSummary_helperText">Memories are stored in layers: <b>Layer 0</b> holds fresh per-turn bullets. Each higher layer is a more compressed merge of older ones. When a layer fills up, its oldest bullets are merged into the next layer.</small>
                <div class="charSummary_sliderRow">
                    <label><small>Min bullets per layer</small></label>
                    <small class="charSummary_helperText">How full a layer has to get before its oldest bullets are merged into the next layer.</small>
                    <input class="neo-range-slider" type="range" id="csc_modal_snippetsPerLayer" min="5" max="100" step="1" value="${s.snippetsPerLayer}" />
                    <div class="wide100p"><input class="neo-range-input" type="number" min="5" max="100" step="1" data-for="csc_modal_snippetsPerLayer" id="csc_modal_snippetsPerLayerCounter" value="${s.snippetsPerLayer}" /></div>
                </div>
                <div class="charSummary_sliderRow">
                    <label><small>Bullets per merge</small></label>
                    <small class="charSummary_helperText">How many bullets get compressed together each time a merge runs.</small>
                    <input class="neo-range-slider" type="range" id="csc_modal_snippetsPerPromotion" min="2" max="20" step="1" value="${s.snippetsPerPromotion}" />
                    <div class="wide100p"><input class="neo-range-input" type="number" min="2" max="20" step="1" data-for="csc_modal_snippetsPerPromotion" id="csc_modal_snippetsPerPromotionCounter" value="${s.snippetsPerPromotion}" /></div>
                </div>
                <div class="charSummary_sliderRow">
                    <label><small>Max layers</small></label>
                    <small class="charSummary_helperText">How deep the memory pyramid goes. More layers = older memories get compressed more times.</small>
                    <input class="neo-range-slider" type="range" id="csc_modal_maxLayers" min="1" max="10" step="1" value="${s.maxLayers}" />
                    <div class="wide100p"><input class="neo-range-input" type="number" min="1" max="10" step="1" data-for="csc_modal_maxLayers" id="csc_modal_maxLayersCounter" value="${s.maxLayers}" /></div>
                </div>
                <div class="charSummary_modalFieldGroup">
                    <label class="checkbox_label">
                        <input type="checkbox" id="csc_modal_progressiveMerge" ${s.progressiveMerge !== false ? 'checked' : ''} />
                        Progressive merge
                        <small class="charSummary_helperText">Deeper layers merge more bullets per merge (L0→L1 = base, then ×2, ×3, and so on).</small>
                    </label>
                </div>
                <div class="charSummary_sliderRow">
                    <label><small>Merge multiplier</small></label>
                    <small class="charSummary_helperText">Scales how aggressive progressive merging is. 1.0 = default, 2 = faster compression, 0.5 = gentler.</small>
                    <input class="neo-range-slider" type="range" id="csc_modal_layerMergeMultiplier" min="0.1" max="3" step="0.1" value="${s.layerMergeMultiplier ?? 1.0}" />
                    <div class="wide100p"><input class="neo-range-input" type="number" min="0.1" max="3" step="0.1" data-for="csc_modal_layerMergeMultiplier" id="csc_modal_layerMergeMultiplierCounter" value="${s.layerMergeMultiplier ?? 1.0}" /></div>
                </div>
            </div>

            <div class="csc_tabPanel" data-panel="connection">
                <h4 class="charSummary_modalSectionTitle">LLM Connection</h4>
                <div class="charSummary_modalFieldGroup">
                    <label><small>Connection source</small></label>
                    <small class="charSummary_helperText">Where summarization requests are sent. <b>Default</b> reuses your active SillyTavern connection — no extra setup.</small>
                    <select id="csc_modal_connectionSource" class="text_pole">${sourceOptions}</select>
                </div>
                <div id="csc_modal_connectionDetails"></div>
            </div>

            <div class="csc_tabPanel" data-panel="files">
                <h4 class="charSummary_modalSectionTitle">File Naming</h4>
                <div class="charSummary_modalFieldGroup">
                    <label><small>Global filename override</small></label>
                    <small class="charSummary_helperText">Only affects single-character chats, and wins over per-character names. Leave blank to auto-name a file per character.</small>
                    <input id="csc_modal_fileName" class="text_pole" type="text" value="${escapeAttr(s.fileName || '')}" placeholder="${DEFAULT_FILE_NAME}" />
                </div>
                <div class="charSummary_modalFieldGroup">
                    <label class="checkbox_label">
                        <input type="checkbox" id="csc_modal_perChat" ${s.perChat ? 'checked' : ''} />
                        Per-chat files
                        <small class="charSummary_helperText">Give each chat its own memory file per character, instead of one shared file across all chats.</small>
                    </label>
                </div>
                <hr class="charSummary_separator" />
                <div class="charSummary_modalFieldGroup">
                    <small class="charSummary_sectionLabel">Per-character filenames</small>
                    <small class="charSummary_helperText">Override the memory filename for individual characters — applies to group members and 1:1 chats. Blank = auto-derived from the character's name.</small>
                    <div class="csc_fileCharList">${charFileRows}</div>
                </div>
            </div>

            <div class="csc_tabPanel" data-panel="prompts">
                <h4 class="charSummary_modalSectionTitle">Prompts <small class="charSummary_helperText" style="display:inline; padding:0;">(per-layer)</small></h4>
                <div class="charSummary_modalFieldGroup">
                    <div class="charSummary_inlineRow" style="margin-bottom:6px; gap:8px;">
                        <label style="flex:1; min-width:0;">
                            <small>Layer</small>
                            <small class="charSummary_helperText">Which layer's prompt you're editing. Layer 0 = fresh summaries. Layer 1+ = compression.</small>
                            <select id="csc_modal_promptLayer" class="text_pole"></select>
                        </label>
                        <label style="flex:1; min-width:0;">
                            <small>Preset</small>
                            <small class="charSummary_helperText">Pick a preset that matches your genre. Each shapes how memories are written and compressed — hover an option for details.</small>
                            <select id="csc_modal_promptPreset" class="text_pole">
                                <option value="narrative" title="Balanced story beats — plot, decisions, reveals. Good default for most roleplay." ${s.promptPreset === 'narrative' ? 'selected' : ''}>Narrative</option>
                                <option value="gamestate" title="Tagged facts like [QUEST] [ITEM] [RESOURCE] with source attribution. For RPG and sandbox tracking." ${s.promptPreset === 'gamestate' ? 'selected' : ''}>Game State</option>
                                <option value="emotional" title="Feelings, trust levels, bond shifts, tensions. For romance, drama, slow-burn." ${s.promptPreset === 'emotional' ? 'selected' : ''}>Emotional &amp; Relationships</option>
                                <option value="journal" title="First-person reflective diary in the character's own voice. For immersion and inner life." ${s.promptPreset === 'journal' ? 'selected' : ''}>Journal / Diary</option>
                                <option value="minimal" title="Ultra-brief telegraphic facts only. Token-efficient for long chats or small models." ${s.promptPreset === 'minimal' ? 'selected' : ''}>Concise / Minimal</option>
                                <option value="goals" title="Active objectives, stated versus hidden motives, lifecycle tracking. For plot-driven stories." ${s.promptPreset === 'goals' ? 'selected' : ''}>Goals &amp; Motives</option>
                                <option value="custom" title="Edit prompts freely." ${s.promptPreset === 'custom' ? 'selected' : ''}>Custom</option>
                            </select>
                        </label>
                    </div>
                    <div id="csc_modal_promptLayerStatus" class="charSummary_helperText" style="display:block; margin-bottom:4px;"></div>
                </div>
                <div class="charSummary_modalFieldGroup">
                    <label><small>System prompt</small></label>
                    <small class="charSummary_helperText">Sets the assistant's role for this layer. Supports <code>{{charName}}</code>. <span id="csc_modal_systemLabel"></span></small>
                    <textarea id="csc_modal_systemPrompt" class="text_pole" rows="3" style="width:100%;" placeholder="System prompt for this layer…"></textarea>
                </div>
                <div class="charSummary_modalFieldGroup">
                    <label><small>User prompt</small></label>
                    <small class="charSummary_helperText">Task instructions. Variables: <code>{{charName}}</code>, <code>{{priorContext}}</code> (what's already remembered), <code>{{passage}}</code> (the new text to summarize).</small>
                    <textarea id="csc_modal_userPrompt" class="text_pole" rows="12" style="width:100%;font-family:monospace;font-size:0.85em;" placeholder="User prompt for this layer…"></textarea>
                </div>
                <div class="charSummary_modalFieldGroup" id="csc_modal_layerActions">
                    <div class="charSummary_buttonRow">
                        <button type="button" class="menu_button" id="csc_modal_createOverride" style="display:none;"><i class="fa-solid fa-plus"></i> Create override</button>
                        <button type="button" class="menu_button" id="csc_modal_removeOverride" style="display:none;"><i class="fa-solid fa-eraser"></i> Remove override</button>
                        <button type="button" class="menu_button" id="csc_modal_resetLayer"><i class="fa-solid fa-rotate-left"></i> Reset to default</button>
                    </div>
                </div>
                <div class="charSummary_modalFieldGroup">
                    <small class="charSummary_sectionLabel">Saved prompt slots</small>
                    <small class="charSummary_helperText">Save the full set of per-layer prompts (Layer 0 + Layer 1 + any overrides). Export writes a JSON file. Legacy <code>.txt</code> imports become Layer 0 user prompts.</small>
                    <div class="charSummary_inlineRow" style="margin-bottom:6px;">
                        <input id="csc_modal_promptSlotName" class="text_pole" type="text" placeholder="slot name" />
                        <select id="csc_modal_promptSlots" class="text_pole">
                            ${Object.keys(s.savedCustomPrompts || {}).length > 0
                                ? Object.keys(s.savedCustomPrompts).map(k => `<option value="${escapeAttr(k)}">${escapeHtml(k)}</option>`).join('')
                                : '<option value="">(no saved slots)</option>'}
                        </select>
                    </div>
                    <div class="charSummary_buttonRow">
                        <button type="button" class="menu_button" id="csc_promptSaveSlot"><i class="fa-solid fa-floppy-disk"></i> Save</button>
                        <button type="button" class="menu_button" id="csc_promptLoadSlot"><i class="fa-solid fa-folder-open"></i> Load</button>
                        <button type="button" class="menu_button" id="csc_promptDeleteSlot"><i class="fa-solid fa-trash"></i> Delete</button>
                        <button type="button" class="menu_button" id="csc_promptExport"><i class="fa-solid fa-file-export"></i> Export</button>
                        <button type="button" class="menu_button" id="csc_promptImport"><i class="fa-solid fa-file-import"></i> Import</button>
                        <input type="file" id="csc_promptImportInput" accept=".txt,.md,.json" style="display:none;" />
                    </div>
                </div>
            </div>

            <div class="csc_tabPanel" data-panel="diagnostics">
                <h4 class="charSummary_modalSectionTitle">Diagnostics</h4>
                <div class="charSummary_modalFieldGroup">
                    <label class="checkbox_label">
                        <input type="checkbox" id="csc_modal_debugMode" ${s.debugMode ? 'checked' : ''} />
                        Debug mode
                        <small class="charSummary_helperText">Logs retry attempts and discarded bullets to the browser console.</small>
                    </label>
                </div>
                <div class="charSummary_modalFieldGroup">
                    <label class="checkbox_label">
                        <input type="checkbox" id="csc_modal_traceMode" ${s.traceMode ? 'checked' : ''} />
                        Trace mode
                        <small class="charSummary_helperText">Verbose step-by-step pipeline tracing in the console. Useful when debugging a stuck run.</small>
                    </label>
                </div>
                <hr class="charSummary_separator" />
                <div class="charSummary_modalFieldGroup">
                    <small class="charSummary_sectionLabel">Reset perception</small>
                    <small class="charSummary_helperText">Reset <b>all</b> messages in this chat to default (everyone perceives everything). Custom perception settings will be permanently lost.</small>
                    <div class="charSummary_buttonRow">
                        <button type="button" class="menu_button" id="csc_modal_resetPerception">
                            <i class="fa-solid fa-rotate-left"></i> Reset Perception
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // callGenericPopup tears down the dialog DOM before resolving its
    // promise, so reading inputs after `await popupPromise` would return
    // undefined. Snapshot the form in onClose, which fires before teardown.
    const captured = { conn: {}, snapshotTaken: false };
    const snapshotForm = () => {
        captured.interval = Number($('#csc_modal_interval').val()) || 10;
        captured.protectedMessages = Number($('#csc_modal_protectedMessages').val()) || 5;
        captured.snippetsPerLayer = Number($('#csc_modal_snippetsPerLayer').val()) || 30;
        captured.snippetsPerPromotion = Number($('#csc_modal_snippetsPerPromotion').val()) || 3;
        captured.maxLayers = Number($('#csc_modal_maxLayers').val()) || 5;
        // Progressive merge. Defensive ?? on the checkbox — the row
        // may not exist if a future template edit removes it, in which case
        // we want to preserve the prior setting rather than clobber to false.
        captured.progressiveMerge = $('#csc_modal_progressiveMerge').length
            ? $('#csc_modal_progressiveMerge').prop('checked')
            : undefined;
        // Number() of a float string; NaN falls back to 1.0 via ||.
        captured.layerMergeMultiplier = Number($('#csc_modal_layerMergeMultiplier').val()) || 1.0;
        syncCurrentLayerToBuffer();
        captured.layerPrompts = promptEditBuffer
            ? cloneLayerPrompts(promptEditBuffer)
            : undefined;
        captured.promptPreset = $('#csc_modal_promptPreset').val() || 'narrative';
        captured.customStash = promptEditStash;
        captured.connectionSource = $('#csc_modal_connectionSource').val() || 'default';
        captured.debugMode = $('#csc_modal_debugMode').prop('checked') || false;
        captured.traceMode = $('#csc_modal_traceMode').prop('checked') || false;
        // Files tab — defensive checks: preserve prior setting if a row is absent.
        if ($('#csc_modal_fileName').length) captured.fileName = $('#csc_modal_fileName').val() || '';
        if ($('#csc_modal_perChat').length) captured.perChat = $('#csc_modal_perChat').prop('checked') || false;
        // Per-character filename overrides. Empty inputs are dropped so the
        // default-derived filename takes over again.
        const charFileNames = {};
        $('.csc_fileCharInput').each(function () {
            const avatar = $(this).attr('data-avatar');
            const val = ($(this).val() || '').trim();
            if (avatar && val) charFileNames[avatar] = val;
        });
        captured.characterFileNames = charFileNames;
        // Only capture sub-fields actually rendered for the active source;
        // absent fields are left undefined so prior values are preserved.
        if ($('#csc_conn_openaiUrl').length) captured.conn.openaiUrl = $('#csc_conn_openaiUrl').val() || '';
        if ($('#csc_conn_openaiKey').length) captured.conn.openaiKey = $('#csc_conn_openaiKey').val() || '';
        if ($('#csc_conn_openaiModel').length) captured.conn.openaiModel = $('#csc_conn_openaiModel').val() || '';
        if ($('#csc_conn_openaiMaxTokens').length) captured.conn.openaiMaxTokens = Number($('#csc_conn_openaiMaxTokens').val()) || 0;
        if ($('#csc_conn_openrouterKey').length) captured.conn.openrouterKey = $('#csc_conn_openrouterKey').val() || '';
        if ($('#csc_conn_openrouterModel').length) captured.conn.openrouterModel = $('#csc_conn_openrouterModel').val() || '';
        if ($('#csc_conn_openrouterMaxTokens').length) captured.conn.openrouterMaxTokens = Number($('#csc_conn_openrouterMaxTokens').val()) || 0;
        if ($('#csc_conn_openrouterReasoning').length) captured.conn.openrouterReasoning = $('#csc_conn_openrouterReasoning').val() || '';
        if ($('#csc_conn_profile').length) captured.conn.connectionProfileId = $('#csc_conn_profile').val() || '';
        captured.snapshotTaken = true;
    };

    // Show modal, then hydrate the connection details panel before awaiting.
    const popupPromise = callGenericPopup(body, POPUP_TYPE.CONFIRM, '', {
        okButton: 'Save',
        cancelButton: 'Cancel',
        wide: true,
        large: true,
        allowVerticalScrolling: true,
        onClose: snapshotForm,
    });

    $('#csc_modal_connectionDetails').html(renderConnectionDetailsHTML(s.connectionSource, s));
    hydrateConnectionDetails(s.connectionSource, s).catch(() => { });

    // Per-layer prompt editor: seed the buffer, populate the layer dropdown,
    // and load the currently-selected layer. Done AFTER the popup is in the DOM.
    openPromptEditBuffer(s);
    promptEditPreset = s.promptPreset || 'narrative';
    promptEditStash = undefined;
    rebuildPromptLayerOptions(promptEditLayer, s.maxLayers, promptEditBuffer);
    loadPromptLayerIntoForm(promptEditLayer);

    const result = await popupPromise;

    if (!result) return;
    // Bail if the snapshot never ran (exotic close path).
    if (!captured.snapshotTaken) return;

    s.interval = captured.interval;
    s.protectedMessages = captured.protectedMessages;
    s.snippetsPerLayer = captured.snippetsPerLayer;
    s.snippetsPerPromotion = captured.snippetsPerPromotion;
    s.maxLayers = captured.maxLayers;
    // Only assign progressiveMerge if the field was actually rendered.
    if (captured.progressiveMerge !== undefined) {
        s.progressiveMerge = captured.progressiveMerge;
    }
    if (captured.layerMergeMultiplier !== undefined) {
        s.layerMergeMultiplier = captured.layerMergeMultiplier;
    }
    if (captured.layerPrompts !== undefined) {
        s.layerPrompts = captured.layerPrompts;
    }

    // Prompt presets. Stash the saved buffer as lastCustomPrompt so the
    // user can flip presets and restore edits later.
    s.promptPreset = captured.promptPreset;
    // Restore the "left Custom mid-session" stash only now, on Save.
    if (captured.customStash) {
        s.lastCustomPrompt = captured.customStash;
    }
    if (s.promptPreset === 'custom' && captured.layerPrompts) {
        s.lastCustomPrompt = cloneLayerPrompts(captured.layerPrompts);
    }
    // savedCustomPrompts is mutated in-place by Save/Delete handlers.

    // Connection settings. Only update sub-fields that were present in the DOM.
    s.connectionSource = captured.connectionSource;
    if (captured.conn.openaiUrl !== undefined) s.openaiUrl = captured.conn.openaiUrl;
    if (captured.conn.openaiKey !== undefined) s.openaiKey = captured.conn.openaiKey;
    if (captured.conn.openaiModel !== undefined) s.openaiModel = captured.conn.openaiModel;
    if (captured.conn.openaiMaxTokens !== undefined) s.openaiMaxTokens = captured.conn.openaiMaxTokens;
    if (captured.conn.openrouterKey !== undefined) s.openrouterKey = captured.conn.openrouterKey;
    if (captured.conn.openrouterModel !== undefined) s.openrouterModel = captured.conn.openrouterModel;
    if (captured.conn.openrouterMaxTokens !== undefined) s.openrouterMaxTokens = captured.conn.openrouterMaxTokens;
    if (captured.conn.openrouterReasoning !== undefined) s.openrouterReasoning = captured.conn.openrouterReasoning;
    if (captured.conn.connectionProfileId !== undefined) s.connectionProfileId = captured.conn.connectionProfileId;

    s.debugMode = captured.debugMode;
    s.traceMode = captured.traceMode;

    // Files tab — filename settings + per-character overrides
    if (captured.fileName !== undefined) s.fileName = captured.fileName;
    if (captured.perChat !== undefined) s.perChat = captured.perChat;
    s.characterFileNames = captured.characterFileNames || {};

    saveSettingsDebounced();
    toastr.success('Settings saved.', 'CharSummaryception');
    updateStatusDisplay();
}

function avatarDisplayName(avatar) {
    if (avatar === USER_AVATAR) return 'User';
    const char = characters.find(c => c.avatar === avatar);
    return char?.name || avatar;
}

function buildDataBankList() {
    const attachments = extension_settings.character_attachments || {};
    const entries = [];
    for (const [avatar, files] of Object.entries(attachments)) {
        if (!Array.isArray(files)) continue;
        for (const file of files) {
            entries.push({ avatar, ...file });
        }
    }
    return entries;
}

async function showDataBankBrowser() {
    const entries = buildDataBankList();

    const renderList = () => {
        if (entries.length === 0) {
            return '<div class="charSummary_diagEmpty">No Data Bank files yet. Use Import to add one.</div>';
        }
        const rows = entries.map((e, i) => {
            const name = escapeHtml(avatarDisplayName(e.avatar));
            const fileName = escapeHtml(e.name || '(unnamed)');
            const size = e.size != null ? `${e.size} B` : '';
            const created = e.created ? new Date(e.created).toLocaleString() : '';
            return `
                <div class="csc_dbRow" data-idx="${i}">
                    <div class="csc_dbRowMain">
                        <div class="csc_dbRowTitle">${fileName}</div>
                        <div class="csc_dbRowMeta">
                            <span title="Character avatar"><i class="fa-solid fa-user fa-xs"></i> ${name}</span>
                            ${size ? `<span>${size}</span>` : ''}
                            ${created ? `<span title="Created">${created}</span>` : ''}
                        </div>
                    </div>
                    <div class="csc_dbRowActions">
                        <button class="menu_button csc_dbView" data-idx="${i}" title="View file content"><i class="fa-solid fa-eye"></i></button>
                        <button class="menu_button csc_dbExport" data-idx="${i}" title="Download as .txt"><i class="fa-solid fa-download"></i></button>
                        <button class="menu_button csc_dbDelete" data-idx="${i}" title="Delete file"><i class="fa-solid fa-trash"></i></button>
                    </div>
                </div>`;
        }).join('');
        return `<div class="csc_dbList">${rows}</div>`;
    };

    const body = `
        <div class="csc_dbToolbar">
            <button class="menu_button" id="csc_dbImport"><i class="fa-solid fa-upload"></i> Import</button>
            <input type="file" id="csc_dbImportInput" accept=".txt,.md" style="display:none;" />
            <button class="menu_button" id="csc_dbRefresh"><i class="fa-solid fa-arrows-rotate"></i> Refresh</button>
            <span class="csc_dbCount">${entries.length} file(s)</span>
        </div>
        <div id="csc_dbContainer">${renderList()}</div>
        <div id="csc_dbViewer" style="display:none;margin-top:12px;">
            <hr class="charSummary_separator" />
            <div class="charSummary_modalFieldGroup">
                <label><small id="csc_dbViewerTitle">File content</small></label>
                <textarea id="csc_dbViewerText" class="text_pole" rows="12" readonly style="width:100%;font-family:monospace;font-size:0.85em;"></textarea>
            </div>
        </div>
    `;

    // cancelButton: false explicitly hides the cancel button (otherwise
    // POPUP_TYPE.CONFIRM renders the default "No" label).
    const result = await callGenericPopup(body, POPUP_TYPE.CONFIRM, '', {
        okButton: 'Close',
        cancelButton: false,
        wide: true,
        large: true,
        allowVerticalScrolling: true,
    });

    return result;
}

$(document).on('click', '#csc_dbImport', function () {
    $('#csc_dbImportInput').click();
});

$(document).on('change', '#csc_dbImportInput', async function () {
    const file = this.files?.[0];
    if (!file) return;
    try {
        const text = await file.text();
        const base64Data = convertTextToBase64(text);
        const slug = getStringHash(file.name);
        const uniqueFileName = `${Date.now()}_${slug}.txt`;
        const fileUrl = await uploadFileAttachment(uniqueFileName, base64Data);
        if (!fileUrl || typeof fileUrl !== 'string') {
            toastr.error(`Failed to import "${file.name}" — upload failed.`, 'CharSummaryception');
            return;
        }
        // Attach to the active character (first member in group chats) so
        // Vector Storage actually indexes the import; fall back to the user
        // pseudo-avatar only when no chat is open.
        const importTargets = getMemoryTargets();
        const targetAvatar = importTargets.length > 0 ? importTargets[0].avatar : USER_AVATAR;
        if (!extension_settings.character_attachments) extension_settings.character_attachments = {};
        if (!extension_settings.character_attachments[targetAvatar]) extension_settings.character_attachments[targetAvatar] = [];
        extension_settings.character_attachments[targetAvatar].push({
            url: fileUrl,
            size: new TextEncoder().encode(text).length,
            name: file.name,
            created: Date.now(),
        });
        saveSettingsDebounced();
        toastr.success(`Imported "${file.name}"${importTargets.length > 0 ? ` for ${importTargets[0].name}` : ''}`, 'CharSummaryception');
        refreshDataBankList();
    } catch (err) {
        console.error(LOG_PREFIX, 'Data Bank import failed:', err);
        toastr.error(`Import failed: ${err.message}`, 'CharSummaryception');
    } finally {
        this.value = '';
    }
});

$(document).on('click', '#csc_dbRefresh', refreshDataBankList);

function refreshDataBankList() {
    const entries = buildDataBankList();
    const $container = $('#csc_dbContainer');
    const $count = $('.csc_dbCount');
    if ($count.length) $count.text(`${entries.length} file(s)`);
    $('#csc_dbViewer').hide();
    if (!$container.length) return;
    if (entries.length === 0) {
        $container.html('<div class="charSummary_diagEmpty">No Data Bank files yet. Use Import to add one.</div>');
        return;
    }
    const rows = entries.map((e, i) => {
        const name = escapeHtml(avatarDisplayName(e.avatar));
        const fileName = escapeHtml(e.name || '(unnamed)');
        const size = e.size != null ? `${e.size} B` : '';
        const created = e.created ? new Date(e.created).toLocaleString() : '';
        return `
            <div class="csc_dbRow" data-idx="${i}">
                <div class="csc_dbRowMain">
                    <div class="csc_dbRowTitle">${fileName}</div>
                    <div class="csc_dbRowMeta">
                        <span title="Character avatar"><i class="fa-solid fa-user fa-xs"></i> ${name}</span>
                        ${size ? `<span>${size}</span>` : ''}
                        ${created ? `<span title="Created">${created}</span>` : ''}
                    </div>
                </div>
                <div class="csc_dbRowActions">
                    <button class="menu_button csc_dbView" data-idx="${i}" title="View file content"><i class="fa-solid fa-eye"></i></button>
                    <button class="menu_button csc_dbExport" data-idx="${i}" title="Download as .txt"><i class="fa-solid fa-download"></i></button>
                    <button class="menu_button csc_dbDelete" data-idx="${i}" title="Delete file"><i class="fa-solid fa-trash"></i></button>
                </div>
            </div>`;
    }).join('');
    $container.html(`<div class="csc_dbList">${rows}</div>`);
}

$(document).on('click', '.csc_dbView', async function () {
    const idx = Number($(this).data('idx'));
    const entries = buildDataBankList();
    const entry = entries[idx];
    if (!entry) return;
    try {
        const content = await getFileAttachment(entry.url);
        $('#csc_dbViewerTitle').text(`Viewing: ${entry.name}`);
        $('#csc_dbViewerText').val(content || '(empty)');
        $('#csc_dbViewer').show();
    } catch (err) {
        toastr.error(`Failed to read file: ${err.message}`, 'CharSummaryception');
    }
});

$(document).on('click', '.csc_dbExport', async function () {
    const idx = Number($(this).data('idx'));
    const entries = buildDataBankList();
    const entry = entries[idx];
    if (!entry) return;
    try {
        const content = await getFileAttachment(entry.url) || '';
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = entry.name || 'memory.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toastr.success(`Downloaded "${entry.name}"`, 'CharSummaryception');
    } catch (err) {
        toastr.error(`Export failed: ${err.message}`, 'CharSummaryception');
    }
});

$(document).on('click', '.csc_dbDelete', async function () {
    const idx = Number($(this).data('idx'));
    const entries = buildDataBankList();
    const entry = entries[idx];
    if (!entry) return;
    // Warn more strongly when the file doesn't look like a memory file —
    // it may belong to another extension.
    const looksLikeMemory = /memories\.(md|txt)$/i.test(entry.name || '');
    // entry.name comes from user imports / filename overrides — escape it
    // before interpolating into popup HTML.
    const safeName = escapeHtml(entry.name || '(unnamed)');
    const warning = looksLikeMemory
        ? `Delete "${safeName}" from ${escapeHtml(avatarDisplayName(entry.avatar))}'s Data Bank? This cannot be undone.`
        : `"<b>${safeName}</b>" does not look like a memory file — it may belong to another extension or feature. Delete it anyway? This cannot be undone.`;
    const confirm = await callGenericPopup(warning, POPUP_TYPE.CONFIRM);
    if (!confirm) return;
    // silent=true => deleteFileFromServer returns false instead of throwing.
    const deleted = await deleteFileFromServer(entry.url, true);
    if (!deleted) {
        toastr.error(`Delete failed — the server rejected the request. "${entry.name}" was kept.`, 'CharSummaryception');
        return;
    }
    const list = extension_settings.character_attachments[entry.avatar];
    if (Array.isArray(list)) {
        extension_settings.character_attachments[entry.avatar] = list.filter(a => a.url !== entry.url);
    }
    saveSettingsDebounced();
    toastr.success(`Deleted "${entry.name}"`, 'CharSummaryception');
    refreshDataBankList();
    updateStatusDisplay();
});

const csc_snipCollapsedLayers = new Set();

async function showSnippetBrowser() {
    if (isSummarizing) {
        toastr.info('Summarization is running — please wait to edit snippets.', 'CharSummaryception');
        return;
    }
    const targets = getMemoryTargets();
    if (targets.length === 0) {
        toastr.info('No character selected.', 'CharSummaryception');
        return;
    }

    csc_snipCollapsedLayers.clear();

    const currentTarget = targets[0];

    // Inline character picker (group chats only). Switching the dropdown
    // re-renders the body via the #csc_snipPicker change handler.
    const nameControl = targets.length > 1
        ? `<select id="csc_snipPicker" class="text_pole">${targets.map((t, i) => `<option value="${i}"${i === 0 ? ' selected' : ''}>${escapeHtml(t.name)}</option>`).join('')}</select>`
        : `<strong>${escapeHtml(currentTarget.name)}</strong>`;

    const body = `
        <div class="csc_snipHeader">
            <div class="csc_snipHeaderInfo">
                ${nameControl}
                <span class="csc_snipFile">${escapeHtml(currentTarget.fileName)}</span>
            </div>
            <div class="csc_snipPromoteBar">
                <span id="csc_snipSelectedCount" class="csc_snipSelectedCount">0 selected</span>
                <button type="button" class="menu_button" id="csc_snipPromoteSelected" title="Merge selected bullets into a higher layer using the LLM" disabled><i class="fa-solid fa-arrow-up"></i> Promote Selected</button>
            </div>
        </div>
        <div id="csc_snipBody"><div class="charSummary_diagEmpty">Loading…</div></div>
    `;

    const popupPromise = callGenericPopup(body, POPUP_TYPE.CONFIRM, '', {
        okButton: 'Close',
        cancelButton: false,
        wide: true,
        large: true,
        allowVerticalScrolling: true,
    });

    await renderSnippetBrowserBody(currentTarget);

    const result = await popupPromise;
    return result;
}

async function renderSnippetBrowserBody(target, blocksOverride) {
    const $body = $('#csc_snipBody');
    if (!$body.length || !target) return;
    // blocksOverride lets callers render from an in-memory array (e.g. after
    // adding a fresh block whose empty bullet would otherwise be stripped by
    // parseMemories on a disk re-read).
    if (!blocksOverride) {
        $body.html('<div class="charSummary_diagEmpty">Loading…</div>');
    }
    try {
        let blocks;
        if (Array.isArray(blocksOverride)) {
            blocks = blocksOverride;
        } else {
            const content = await readMemoriesForCharacter(target.avatar, target.fileName);
            blocks = parseMemories(content || '');
        }
        if (blocks.length === 0) {
            $body.html('<div class="charSummary_diagEmpty">No memory blocks. Run summarization first.</div>');
            updateSnipSelectionState();
            return;
        }

        const layerGroups = {};
        for (const b of blocks) {
            const layer = b.layer || 0;
            if (!layerGroups[layer]) layerGroups[layer] = [];
            layerGroups[layer].push(b);
        }

        const layerLabel = (layer) => {
            if (layer === 0) return 'Layer 0 — Turn Summaries';
            if (layer === 1) return 'Layer 1 — Merged';
            return `Layer ${layer} — Meta-Summary`;
        };

        // Build a block→globalIndex lookup once (O(n)) instead of calling
        // blocks.indexOf(b) inside the nested loop (O(n²)).
        const blockIdxMap = new Map();
        for (let i = 0; i < blocks.length; i++) blockIdxMap.set(blocks[i], i);

        const layersHtml = Object.keys(layerGroups).sort((a, b) => Number(a) - Number(b)).map(layer => {
            const blocksAtLayer = layerGroups[layer];
            const blockHtml = blocksAtLayer.map((b) => {
                const globalIdx = blockIdxMap.get(b);
                const bullets = b.bullets.map((bul, j) =>
                    `<li class="csc_snipBullet" data-block="${globalIdx}" data-bullet="${j}"><input type="checkbox" class="csc_snipBulletCheck" title="Select for merging into a higher layer" /><span class="csc_snipBulletText">${escapeHtml(bul)}</span><i class="fa-solid fa-trash csc_snipBulletDelete" title="Delete this bullet"></i></li>`
                ).join('');
                const meta = [
                    b.date ? `<span title="Date">${escapeHtml(b.date)}</span>` : '',
                    b.range ? `<span title="Range">[${b.range[0]}-${b.range[1]}]</span>` : '',
                    b.promoted ? '<span title="Merged up from a lower layer">merged</span>' : '',
                    Number.isFinite(b.seedFromLayer) ? `<span title="Seeded from Layer ${b.seedFromLayer}">seed</span>` : '',
                    b.blockPromotion ? '<span title="Protected from merging" class="csc_snipProtectedTag">protected</span>' : '',
                ].filter(Boolean).join('');
                const protectedCls = b.blockPromotion ? ' csc_protected' : '';
                const shieldIcon = b.blockPromotion ? 'fa-shield-halved' : 'fa-shield';
                const shieldTitle = b.blockPromotion ? 'Remove merge protection' : 'Protect from merging';
                return `
                    <div class="csc_snipBlock${protectedCls}" data-block="${globalIdx}">
                        <div class="csc_snipBlockMeta"><input type="checkbox" class="csc_snipSelectBlockCheck" data-block="${globalIdx}" title="Select every bullet in this block" />${meta}</div>
                        <ul class="csc_snipBullets">${bullets}</ul>
                        <div class="csc_snipBlockActions">
                            <button class="menu_button csc_snipProtectBlock${protectedCls}" data-block="${globalIdx}" title="${shieldTitle}"><i class="fa-solid ${shieldIcon}"></i></button>
                            <button class="menu_button csc_snipAddBullet" data-block="${globalIdx}" title="Add a bullet to this block"><i class="fa-solid fa-plus"></i> Bullet</button>
                            <button class="menu_button csc_snipDeleteBlock" data-block="${globalIdx}" title="Delete this block"><i class="fa-solid fa-trash"></i></button>
                        </div>
                    </div>`;
            }).join('');
            return `<div class="csc_snipLayer" data-layer="${Number(layer)}">
                <div class="csc_snipLayerTitle">
                    <span class="csc_snipLayerTitleText"><button type="button" class="csc_snipLayerToggle" data-layer="${Number(layer)}" title="Collapse/expand layer"><i class="fa-solid fa-chevron-down"></i></button>${escapeHtml(layerLabel(Number(layer)))} <span class="csc_snipLayerCount">${blocksAtLayer.length}</span></span>
                    <div class="csc_snipLayerActions">
                        <label class="csc_snipSelectAllLabel" title="Select every bullet in this layer"><input type="checkbox" class="csc_snipSelectAllCheck" data-layer="${Number(layer)}" /> All</label>
                        <button class="menu_button csc_snipAddBlock" data-layer="${Number(layer)}" title="Add a new memory block to this layer"><i class="fa-solid fa-plus"></i> Block</button>
                    </div>
                </div>
                <div class="csc_snipLayerBody">${blockHtml}</div>
            </div>`;
        }).join('');

        $body.html(`<div class="csc_snipLayers" spellcheck="false">${layersHtml}</div>`);
        if (csc_snipCollapsedLayers.size) {
            csc_snipCollapsedLayers.forEach(layer => {
                $body.find(`.csc_snipLayer[data-layer="${layer}"]`).addClass('collapsed');
            });
        }
        $body.data('target', target);
        $body.data('blocks', blocks);
        // Snapshot the baseline for concurrent-modification detection in
        // saveSnippetEdits. Deep-cloned so in-place edits to `blocks` don't
        // silently update the baseline.
        $body.data('baselineBlocks', JSON.parse(JSON.stringify(blocks)));
        updateSnipSelectionState();
    } catch (err) {
        $body.html(`<div class="charSummary_diagEmpty">Failed to load: ${escapeHtml(err.message)}</div>`);
        updateSnipSelectionState();
    }
}

function _refreshSelectAllForLayer($layer) {
    const $bullets = $layer.find('input.csc_snipBulletCheck');
    const $selectAll = $layer.find('input.csc_snipSelectAllCheck');
    if (!$bullets.length) {
        $selectAll.prop('checked', false).prop('disabled', true);
    } else {
        $selectAll.prop('disabled', false);
        const checkedInLayer = $layer.find('input.csc_snipBulletCheck:checked').length;
        $selectAll.prop('checked', checkedInLayer === $bullets.length);
    }
}

function _refreshSelectAllForBlock($block) {
    const $bullets = $block.find('input.csc_snipBulletCheck');
    const $selectBlock = $block.find('input.csc_snipSelectBlockCheck');
    if (!$bullets.length) {
        $selectBlock.prop('checked', false).prop('disabled', true);
    } else {
        $selectBlock.prop('disabled', false);
        const checkedInBlock = $block.find('input.csc_snipBulletCheck:checked').length;
        $selectBlock.prop('checked', checkedInBlock === $bullets.length);
    }
}

function updateSnipSelectionState(scopeEl) {
    const $body = $('#csc_snipBody');
    if (!$body.length) return;
    const checked = $body.find('input.csc_snipBulletCheck:checked').length;
    $('#csc_snipSelectedCount').text(`${checked} selected`);
    $('#csc_snipPromoteSelected').prop('disabled', checked < 2);

    if (!scopeEl) {
        $body.find('.csc_snipLayer').each(function () { _refreshSelectAllForLayer($(this)); });
        $body.find('.csc_snipBlock').each(function () { _refreshSelectAllForBlock($(this)); });
        return;
    }

    const $scope = $(scopeEl);
    if ($scope.is('.csc_snipLayer')) {
        $scope.find('.csc_snipBlock').each(function () { _refreshSelectAllForBlock($(this)); });
        _refreshSelectAllForLayer($scope);
    } else {
        const $block = $scope.is('.csc_snipBlock') ? $scope : $scope.closest('.csc_snipBlock');
        if ($block.length) _refreshSelectAllForBlock($block);
        const $layer = $scope.closest('.csc_snipLayer');
        if ($layer.length) _refreshSelectAllForLayer($layer);
    }
}

/**
 * Coarse fingerprint of a blocks array for concurrent-modification detection.
 * Catches the common clobber case (summarizer appended/removed blocks) without
 * being expensive. Not a full hash — false negatives are acceptable, false
 * positives are not (a false positive would block a legitimate save).
 */
function fingerprintBlocks(blocks) {
    if (!Array.isArray(blocks) || blocks.length === 0) return '0|0||';
    const totalBullets = blocks.reduce((n, b) => n + (b.bullets?.length || 0), 0);
    const firstText = blocks[0]?.bullets?.[0] || '';
    const lastText = blocks[blocks.length - 1]?.bullets?.[(blocks[blocks.length - 1]?.bullets?.length || 1) - 1] || '';
    return `${blocks.length}|${totalBullets}|${firstText}|${lastText}`;
}

async function saveSnippetEdits(target, blocks) {
    // Concurrent-modification guard: if the on-disk file no longer matches the
    // baseline captured when the snippet browser was opened/last refreshed,
    // another writer (the summarizer) has touched it. Reload instead of
    // clobbering the fresh data with our stale snapshot.
    const baseline = $('#csc_snipBody').data('baselineBlocks');
    if (baseline) {
        const diskContent = await readMemoriesForCharacter(target.avatar, target.fileName);
        // diskContent === null means transient read error — can't verify, so
        // proceed cautiously (the write itself is atomic-ish via upload-first).
        if (diskContent !== null) {
            const diskBlocks = parseMemories(diskContent || '');
            if (fingerprintBlocks(diskBlocks) !== fingerprintBlocks(baseline)) {
                toastr.warning('Memory file was modified by another process. Reloading snippets.', 'CharSummaryception');
                logActivity(`Skipped snippet save for ${target.name} — concurrent modification detected`, 'warning');
                await renderSnippetBrowserBody(target);
                return;
            }
        }
    }
    const serialized = serializeMemories(blocks);
    await writeMemoriesForCharacter(serialized, target.avatar, target.fileName);
    // Update the baseline so subsequent edits in the same session compare
    // against the most recent persisted state.
    $('#csc_snipBody').data('baselineBlocks', JSON.parse(JSON.stringify(blocks)));
    logActivity(`Edited snippets for ${target.name} (${blocks.length} blocks)`, 'success');
}

async function onPromoteSelectedClick() {
    if (isSummarizing || isPromoting) {
        toastr.info('Summarization is running — cannot promote bullets.', 'CharSummaryception');
        return;
    }
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    const target = $body.data('target');
    if (!blocks || !target) return;

    const selected = [];
    $body.find('.csc_snipBlock').each(function () {
        const blockIdx = Number($(this).attr('data-block'));
        const block = blocks[blockIdx];
        if (!block) return;
        let bulletIdx = 0;
        $(this).find('.csc_snipBullet').each(function () {
            const $cb = $(this).find('input.csc_snipBulletCheck');
            if ($cb.length && $cb.prop('checked') && bulletIdx < block.bullets.length) {
                selected.push({ blockIdx, bulletIdx, text: block.bullets[bulletIdx] });
            }
            bulletIdx++;
        });
    });

    if (selected.length < 2) {
        toastr.warning('Select at least 2 bullets to promote.', 'CharSummaryception');
        return;
    }
    if (selected.some(s => !s.text)) {
        toastr.error('Could not read one or more selected bullets.', 'CharSummaryception');
        return;
    }

    const s = extension_settings[MODULE_NAME];
    const maxLayers = s.maxLayers || 5;
    const deepestLayer = selected.reduce((m, sel) => {
        return Math.max(m, blocks[sel.blockIdx].layer || 0);
    }, 0);
    const targetLayer = Math.max(1, Math.min(deepestLayer + 1, maxLayers - 1));

    if (targetLayer <= deepestLayer) {
            toastr.warning(`Can't merge higher — that would exceed Max layers. Raise it in Settings → Compression.`, 'CharSummaryception');
        return;
    }

    const storyText = selected.map(sel => `- ${sel.text}`).join('\n');
    const priorContext = buildPriorContext(blocks, 10);
    const { systemPrompt: sysTemplate, userPrompt: userTemplate } = resolveLayerPrompt(s, targetLayer);
    const systemPrompt = substitutePromptTemplate(sysTemplate, { charName: target.name });
    const userPrompt = substitutePromptTemplate(userTemplate, {
        charName: target.name,
        priorContext,
        passage: storyText,
    });

    const confirm = await callGenericPopup(
        `Promote <strong>${selected.length}</strong> bullets to <strong>Layer ${targetLayer}</strong>?<br><br>The selected bullets will be removed from their source blocks and replaced with a single compressed block at Layer ${targetLayer}.`,
        POPUP_TYPE.CONFIRM,
    );
    if (!confirm) return;

    toastr.info(`Promoting ${selected.length} bullets to Layer ${targetLayer}…`, 'CharSummaryception', { timeOut: 3000 });
    logActivity(`Starting selective promotion: ${selected.length} bullets → Layer ${targetLayer} for ${target.name}`);

    // Hold a busy flag for the whole operation so an auto-triggered
    // summarization run can't race the final write.
    isPromoting = true;
    try {
        const deps = makeDeps();
        const result = await callSummarizer(deps, systemPrompt, userPrompt);
        if (!result || !result.trim()) {
            toastr.error('Promotion produced no output.', 'CharSummaryception');
            return;
        }

        const newBullets = result.split('\n')
            .map(l => l.trim())
            .filter(l => l.startsWith('- '))
            .map(l => l.replace(/^-\s+/, '').trim())
            .filter(Boolean);

        if (newBullets.length === 0) {
            toastr.error('No valid bullets in promotion output.', 'CharSummaryception');
            return;
        }

        const byBlock = new Map();
        for (const sel of selected) {
            if (!byBlock.has(sel.blockIdx)) byBlock.set(sel.blockIdx, []);
            byBlock.get(sel.blockIdx).push(sel.bulletIdx);
        }
        for (const [blockIdx, idxs] of byBlock) {
            idxs.sort((a, b) => b - a);
            for (const idx of idxs) {
                blocks[blockIdx].bullets.splice(idx, 1);
            }
        }
        for (let i = blocks.length - 1; i >= 0; i--) {
            if (blocks[i].bullets.length === 0) {
                blocks.splice(i, 1);
            }
        }

        const promotedBlock = {
            chat: target.name,
            date: getTimestamp(),
            bullets: newBullets,
            layer: targetLayer,
            promoted: false,
        };
        blocks.push(promotedBlock);

        // Check whether the target layer now exceeds the promotion threshold.
        // Manual promotion cascades into promoteLayer to mirror the automatic pipeline.
        const snippetsPerLayer = s.snippetsPerLayer || 30;
        const layerBlocks = blocks.filter(b => (b.layer || 0) === targetLayer && !b.promoted && !b.blockPromotion);
        if (layerBlocks.length > snippetsPerLayer && targetLayer < maxLayers - 1) {
            logActivity(`Layer ${targetLayer} exceeded threshold (${layerBlocks.length} > ${snippetsPerLayer}) after manual promotion — cascading`, 'info');
            const cascaded = await promoteLayer(deps, target, blocks, targetLayer);
            if (!cascaded) {
                logActivity(`Cascade promotion at Layer ${targetLayer} did not complete — layer may be over threshold`, 'warning');
            }
        }

        // Route through saveSnippetEdits so the concurrent-modification
        // fingerprint guard applies to the promotion write too.
        await saveSnippetEdits(target, blocks);
        logActivity(`Promoted ${selected.length} bullets → ${newBullets.length} bullets at Layer ${targetLayer} for ${target.name}`, 'success');
        toastr.success(`Promoted: ${selected.length} → ${newBullets.length} bullets (Layer ${targetLayer})`, 'CharSummaryception');
        updateStatusDisplay();

        await renderSnippetBrowserBody(target, blocks);
    } catch (err) {
        console.error(LOG_PREFIX, 'Selective promotion failed:', err);
        logActivity(`Selective promotion failed: ${err.message}`, 'error');
        toastr.error(`Promotion failed: ${err.message}`, 'CharSummaryception');
    } finally {
        isPromoting = false;
    }
}

$(document).on('change', '#csc_snipPicker', async function () {
    const idx = Number($(this).val()) || 0;
    const targets = getMemoryTargets();
    const target = targets[idx];
    if (!target) return;
    $('.csc_snipFile').text(target.fileName);
    await renderSnippetBrowserBody(target);
});

$(document).on('change', '#csc_snipBody input.csc_snipBulletCheck', function (e) {
    e.stopPropagation();
    updateSnipSelectionState(this);
});

$(document).on('change', '.csc_snipSelectAllCheck', function (e) {
    e.stopPropagation();
    const checked = $(this).prop('checked');
    $(this).closest('.csc_snipLayer').find('input.csc_snipBulletCheck').prop('checked', checked);
    updateSnipSelectionState($(this).closest('.csc_snipLayer'));
});

$(document).on('click', '.csc_snipLayerToggle', function (e) {
    e.stopPropagation();
    const $layer = $(this).closest('.csc_snipLayer');
    const layer = Number($layer.attr('data-layer'));
    const collapsed = !$layer.hasClass('collapsed');
    $layer.toggleClass('collapsed', collapsed);
    csc_snipCollapsedLayers[collapsed ? 'add' : 'delete'](layer);
});

$(document).on('change', '.csc_snipSelectBlockCheck', function (e) {
    e.stopPropagation();
    const checked = $(this).prop('checked');
    $(this).closest('.csc_snipBlock').find('input.csc_snipBulletCheck').prop('checked', checked);
    updateSnipSelectionState($(this).closest('.csc_snipBlock'));
});

$(document).on('click', '#csc_snipPromoteSelected', onPromoteSelectedClick);

$(document).on('mousedown', '.csc_snipBulletText', function () {
    if (this.getAttribute('contenteditable') !== 'plaintext-only') {
        this.setAttribute('contenteditable', 'plaintext-only');
        this.focus();
    }
});

$(document).on('blur', '.csc_snipBulletText', async function () {
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    const target = $body.data('target');
    const newText = $(this).text().trim();
    this.removeAttribute('contenteditable');
    if (!blocks || !target) return;
    const $bullet = $(this).closest('.csc_snipBullet');
    const $block = $bullet.closest('.csc_snipBlock');
    const blockIdx = Number($block.data('block'));
    // Use live DOM position, not stored data-bullet — siblings keep stale
    // indices after a splice.
    const bulletIdx = $block.find('.csc_snipBullet').index($bullet);
    if (!blocks[blockIdx]) return;
    if (bulletIdx < 0 || bulletIdx >= blocks[blockIdx].bullets.length) {
        // New bullet appended at end (via the "+" button). Drop the orphan
        // <li> if there's no text so the UI doesn't show a stale empty row.
        if (newText) {
            blocks[blockIdx].bullets.push(newText);
            await saveSnippetEdits(target, blocks);
        } else {
            $bullet.remove();
        }
    } else if (!newText) {
        // Don't delete an existing bullet on accidental clear — restore the
        // original text. Use the dedicated trash button (csc_snipBulletDelete)
        // for intentional deletion.
        const original = blocks[blockIdx].bullets[bulletIdx];
        $(this).text(original);
        toastr.info('Bullet cleared — restored. Use the trash button to delete it.', 'CharSummaryception', { timeOut: 2000 });
        return;
    } else {
        blocks[blockIdx].bullets[bulletIdx] = newText;
    }
    await saveSnippetEdits(target, blocks);
});

$(document).on('click', '.csc_snipBulletDelete', async function (e) {
    e.stopPropagation();
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    const target = $body.data('target');
    if (!blocks || !target) return;
    const $bullet = $(this).closest('.csc_snipBullet');
    const $block = $bullet.closest('.csc_snipBlock');
    const blockIdx = Number($block.data('block'));
    const bulletIdx = $block.find('.csc_snipBullet').index($bullet);
    if (!blocks[blockIdx] || bulletIdx < 0 || bulletIdx >= blocks[blockIdx].bullets.length) return;
    blocks[blockIdx].bullets.splice(bulletIdx, 1);
    $bullet.remove();
    // Drop the block too if it's now empty, so serializeMemories doesn't write
    // a phantom <memory> tag that parseMemories silently discards.
    if (blocks[blockIdx].bullets.length === 0) {
        blocks.splice(blockIdx, 1);
        await saveSnippetEdits(target, blocks);
        await renderSnippetBrowserBody(target, blocks);
        return;
    }
    await saveSnippetEdits(target, blocks);
});

$(document).on('click', '.csc_snipAddBullet', function () {
    const blockIdx = Number($(this).data('block'));
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    if (!blocks || !blocks[blockIdx]) return;
    const $block = $body.find(`.csc_snipBlock[data-block="${blockIdx}"]`);
    const $ul = $block.find('.csc_snipBullets');
    const $newLi = $(`<li class="csc_snipBullet" data-block="${blockIdx}"><input type="checkbox" class="csc_snipBulletCheck" title="Select for merging into a higher layer" /><span class="csc_snipBulletText" contenteditable="plaintext-only"></span></li>`);
    $ul.append($newLi);
    const editor = $newLi.find('.csc_snipBulletText')[0];
    if (editor) {
        editor.focus();
        try {
            const range = document.createRange();
            range.selectNodeContents(editor);
            range.collapse(false);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        } catch { /* focus alone is enough for an empty node */ }
    }
});

$(document).on('click', '.csc_snipAddBlock', async function () {
    const layer = Number($(this).data('layer'));
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    const target = $body.data('target');
    if (!blocks || !target) return;
    const newBlock = {
        chat: 'manual',
        date: getTimestamp(new Date()),
        bullets: [''],
        layer: Number.isFinite(layer) ? layer : 0,
    };
    blocks.push(newBlock);
    // Render from the in-memory array (NOT a disk re-read): parseMemories
    // would discard the empty placeholder bullet.
    await renderSnippetBrowserBody(target, blocks);
    const newIdx = blocks.length - 1;
    const newBullet = $body.find(`.csc_snipBlock[data-block="${newIdx}"] .csc_snipBulletText`)[0];
    if (newBullet) {
        newBullet.setAttribute('contenteditable', 'plaintext-only');
        newBullet.focus();
    }
});

$(document).on('click', '.csc_snipDeleteBlock', async function () {
    const blockIdx = Number($(this).data('block'));
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    const target = $body.data('target');
    if (!blocks || !target || !blocks[blockIdx]) return;
    const confirm = await callGenericPopup('Delete this memory block and all its bullets?', POPUP_TYPE.CONFIRM);
    if (!confirm) return;
    blocks.splice(blockIdx, 1);
    await saveSnippetEdits(target, blocks);
    await renderSnippetBrowserBody(target, blocks);
});

$(document).on('click', '.csc_snipProtectBlock', async function () {
    const blockIdx = Number($(this).data('block'));
    const $body = $('#csc_snipBody');
    const blocks = $body.data('blocks');
    const target = $body.data('target');
    if (!blocks || !target || !blocks[blockIdx]) return;
    const block = blocks[blockIdx];
    const next = !block.blockPromotion;
    if (next) {
        block.blockPromotion = true;
    } else {
        delete block.blockPromotion;
    }
    await saveSnippetEdits(target, blocks);
    logActivity(`${next ? 'Protected' : 'Unprotected'} block ${blockIdx} for ${target.name}`);
    await renderSnippetBrowserBody(target, blocks);
});

/**
 * Reset ALL messages in the current chat to default perception (everyone
 * perceives everything). Strips every message's extra.perception map so
 * canPerceive() falls back to its "no entry = everyone sees it" default.
 */
async function resetChatPerception() {
    const chat = getContext().chat;
    if (!Array.isArray(chat) || chat.length === 0) {
        toastr.info('No messages to reset.', 'CharSummaryception');
        return;
    }

    const confirm = await callGenericPopup(
        'Reset perception for <b>ALL messages</b> in this chat?<br><br>Every message will return to the default state where everyone perceives everything. Custom perception settings on all messages will be permanently lost.',
        POPUP_TYPE.CONFIRM,
    );
    if (!confirm) return;

    let cleared = 0;
    for (const msg of chat) {
        if (!msg) continue;
        if (msg.extra?.perception && Object.keys(msg.extra.perception).length > 0) {
            delete msg.extra.perception;
            cleared++;
        }
    }

    if (cleared === 0) {
        toastr.info('No custom perception settings found — chat is already at default.', 'CharSummaryception');
        return;
    }

    saveChatDebounced();

    log('[PERC] resetChatPerception', { cleared });

    // Refresh all per-message perception buttons to drop the "custom" indicator.
    injectPerceptionButtonsAll();

    // Re-render the drawer body if it's currently open so the table resets.
    if ($('#charSummary_sceneDrawer').hasClass('open')) {
        renderSceneDrawerBody();
    }

    logActivity(`Reset perception for ${cleared} message(s) to default (everyone perceives everything)`, 'success');
    toastr.success(`Reset ${cleared} message(s) to default perception.`, 'CharSummaryception');
}

/**
 * Reset ONLY the message currently bound to the perception drawer.
 */
async function resetMessagePerception() {
    if (!Number.isInteger(drawerMessageId)) {
        toastr.info('No message selected.', 'CharSummaryception');
        return;
    }
    const chat = getContext().chat;
    const msg = chat?.[drawerMessageId];
    if (!msg) {
        toastr.info('No message selected.', 'CharSummaryception');
        return;
    }

    const hadPerception = msg.extra?.perception && Object.keys(msg.extra.perception).length > 0;

    const confirm = await callGenericPopup(
        `Reset perception for message <b>#${drawerMessageId + 1}</b>?<br><br>This message will return to the default state where everyone perceives everything.`,
        POPUP_TYPE.CONFIRM,
    );
    if (!confirm) return;

    if (!hadPerception) {
        toastr.info('This message is already at default perception.', 'CharSummaryception');
        return;
    }

    delete msg.extra.perception;
    saveChatDebounced();

    log('[PERC] resetMessagePerception', { messageId: drawerMessageId });

    injectPerceptionButton(drawerMessageId);

    if ($('#charSummary_sceneDrawer').hasClass('open')) {
        renderSceneDrawerBody();
    }

    logActivity(`Reset perception for message #${drawerMessageId + 1} to default (everyone perceives everything)`, 'success');
    toastr.success(`Reset message #${drawerMessageId + 1} to default perception.`, 'CharSummaryception');
}

function onAutoPillClick() {
    const s = extension_settings[MODULE_NAME];
    s.enabled = !s.enabled;
    saveSettingsDebounced();
    updateStatusDisplay();
    toastr.info(`Auto-summarization ${s.enabled ? 'enabled' : 'disabled'}`, 'CharSummaryception');
}

jQuery(async function () {
    // Derive the extension key from this module's URL so the settings
    // template still resolves if the install folder was renamed.
    const extensionKey = decodeURIComponent(new URL('.', import.meta.url).pathname)
        .split('/').filter(Boolean).slice(-2).join('/');
    const settingsHtml = await renderExtensionTemplateAsync(extensionKey, 'settings');
    $('#extensions_settings2').append(settingsHtml);

    $('body').append(`
        <div id="charSummary_sceneDrawer" class="charSummary_sceneDrawer">
            <div class="charSummary_drawerHeader">
                <i class="fa-solid fa-users" style="font-size:0.85em;opacity:0.7;"></i>
                <span class="charSummary_drawerTitle">Scene State</span>
                <div class="charSummary_drawerClose" id="charSummary_sceneClose" title="Close"><i class="fa-solid fa-xmark"></i></div>
            </div>
            <div class="charSummary_sceneToolbar">
                <small class="charSummary_sceneIntro">Choose who can see and hear the speaker. Unchecked characters won't get this message in their memory.</small>
                <div class="charSummary_sceneActions">
                    <button type="button" class="menu_button" id="charSummary_resetPerception" title="Reset this message so everyone perceives it again">
                        <i class="fa-solid fa-rotate-left"></i> Reset
                    </button>
                </div>
            </div>
            <div class="charSummary_sceneBody" id="charSummary_sceneBody">
                <div class="charSummary_diagEmpty">Click the group icon on any message to edit who perceived it.</div>
            </div>
        </div>
    `);

    loadSettings();

    eventSource.on(event_types.CHARACTER_MESSAGE_RENDERED, onCharacterMessageRendered);
    eventSource.on(event_types.USER_MESSAGE_RENDERED, onUserMessageRendered);
    eventSource.on(event_types.MESSAGE_DELETED, onMessageDeleted);
    eventSource.on(event_types.CHAT_CHANGED, onChatChanged);

    // Inject the perception button into all messages (new + loaded) via DOM
    // observation, since message-render events don't reliably fire on load.
    startPerceptionButtonObserver();

    $('#charSummary_openSettings').on('click', function (e) {
        e.stopPropagation();
        showSettingsModal();
    });

    $('#charSummary_openSceneState').on('click', function (e) {
        e.stopPropagation();
        const chat = getContext().chat;
        if (!chat || chat.length === 0) {
            toastr.info('No messages to edit yet.', 'CharSummaryception');
            return;
        }
        openPerceptionForMessage(chat.length - 1);
    });

    $('#charSummary_sceneClose').on('click', () => toggleSceneDrawer());
    $('#charSummary_resetPerception').on('click', resetMessagePerception);

    $('#charSummary_summarizeNow').on('click', onSummarizeNowClick);
    $('#charSummary_stopSummarize').on('click', function () {
        abortSummarization();
        toastr.warning('Summarization stopped.', 'CharSummaryception');
    });
    $('#charSummary_autoPill').on('click', onAutoPillClick);
    $('#charSummary_dataBankBtn').on('click', showDataBankBrowser);
    $('#charSummary_snippetBtn').on('click', showSnippetBrowser);

    // Settings modal is created/destroyed by callGenericPopup, so its Reset
    // Perception button needs a delegated handler.
    $(document).on('click', '#csc_modal_resetPerception', resetChatPerception);

    $(document).on('change', '#charSummary_sceneDrawer input[type="checkbox"]', function () {
        onSceneCheckboxChange(this);
    });

    $(document).on('change', '#csc_modal_connectionSource', async function () {
        const source = $(this).val();
        const s = extension_settings[MODULE_NAME];
        $('#csc_modal_connectionDetails').html(renderConnectionDetailsHTML(source, s));
        await hydrateConnectionDetails(source, s).catch(() => { });
    });

    // Settings modal tab switching. Panels stay mounted so all form inputs
    // remain capturable regardless of which tab is active on Save.
    $(document).on('click', '.csc_tab', function () {
        const tab = $(this).attr('data-tab');
        if (!tab) return;
        const $root = $(this).closest('.csc_tabs');
        $root.find('.csc_tab').removeClass('active');
        $(this).addClass('active');
        $root.find('.csc_tabPanel').removeClass('active');
        $root.find(`.csc_tabPanel[data-panel="${tab}"]`).addClass('active');
    });

    // Slider ↔ number-counter sync. ST core doesn't wire these up for
    // third-party extensions, so dragging the slider wouldn't update the
    // visible value.
    $(document).on('input', '.neo-range-slider', function () {
        const sliderId = $(this).attr('id');
        if (!sliderId) return;
        const $counter = $(`[data-for="${sliderId}"]`);
        if ($counter.length) $counter.val($(this).val());
    });
    $(document).on('input', '.neo-range-input', function () {
        const sliderId = $(this).attr('data-for');
        if (!sliderId) return;
        const $slider = $(`#${sliderId}`);
        if ($slider.length) {
            // Clamp to the slider's min/max so out-of-range typing doesn't desync.
            const min = Number($slider.attr('min'));
            const max = Number($slider.attr('max'));
            let v = Number($(this).val());
            if (Number.isFinite(min)) v = Math.max(min, v);
            if (Number.isFinite(max)) v = Math.min(max, v);
            $slider.val(v);
        }
    });

    $(document).on('change', '#csc_modal_promptLayer', function () {
        // Flush the just-edited layer to the buffer BEFORE switching.
        syncCurrentLayerToBuffer();
        const layer = Number($(this).val()) || 0;
        loadPromptLayerIntoForm(layer);
        const s = extension_settings[MODULE_NAME];
        rebuildPromptLayerOptions(layer, s.maxLayers, promptEditBuffer);
    });

    $(document).on('click', '#csc_modal_createOverride', function () {
        if (!promptEditBuffer) return;
        const layer = promptEditLayer;
        if (layer < 2) return;
        // Seed the override with Layer 1's prompts so the user starts from
        // a known-good base rather than a blank textarea.
        const base = resolveLayerPrompt({ layerPrompts: promptEditBuffer }, 1);
        promptEditBuffer[layer] = {
            systemPrompt: base.systemPrompt,
            userPrompt: base.userPrompt,
        };
        loadPromptLayerIntoForm(layer);
        const s = extension_settings[MODULE_NAME];
        rebuildPromptLayerOptions(layer, s.maxLayers, promptEditBuffer);
        // Mark dirty so the modal knows this is now a Custom configuration.
        // Modal-local only — committed to settings on Save.
        promptEditPreset = 'custom';
        $('#csc_modal_promptPreset').val('custom');
        toastr.info(`Created Layer ${layer} override.`, 'CharSummaryception');
    });

    $(document).on('click', '#csc_modal_removeOverride', function () {
        if (!promptEditBuffer) return;
        const layer = promptEditLayer;
        if (layer < 2) return;
        delete promptEditBuffer[layer];
        loadPromptLayerIntoForm(layer);
        const s = extension_settings[MODULE_NAME];
        rebuildPromptLayerOptions(layer, s.maxLayers, promptEditBuffer);
        promptEditPreset = 'custom';
        $('#csc_modal_promptPreset').val('custom');
        toastr.info(`Removed Layer ${layer} override (inherits Layer 1).`, 'CharSummaryception');
    });

    $(document).on('click', '#csc_modal_resetLayer', function () {
        if (!promptEditBuffer) return;
        const layer = promptEditLayer;
        const defaults = DEFAULT_LAYER_PROMPTS[layer] || DEFAULT_LAYER_PROMPTS[1];
        promptEditBuffer[layer] = {
            systemPrompt: defaults.systemPrompt,
            userPrompt: defaults.userPrompt,
        };
        loadPromptLayerIntoForm(layer);
        const s = extension_settings[MODULE_NAME];
        rebuildPromptLayerOptions(layer, s.maxLayers, promptEditBuffer);
        // Reset ≠ the active preset anymore.
        promptEditPreset = 'custom';
        $('#csc_modal_promptPreset').val('custom');
    });

    // Preset dropdown: swap the buffer to the preset's per-layer prompts
    // (or restore the user's last Custom buffer).
    $(document).on('change', '#csc_modal_promptPreset', function () {
        const s = extension_settings[MODULE_NAME];
        const prev = promptEditPreset;
        const next = $(this).val();
        if (prev === 'custom') {
            syncCurrentLayerToBuffer();
            promptEditStash = cloneLayerPrompts(promptEditBuffer);
        }
        let newBuffer;
        if (next === 'custom') {
            newBuffer = promptEditStash || s.lastCustomPrompt || cloneLayerPrompts(PROMPT_PRESETS.narrative);
        } else {
            newBuffer = cloneLayerPrompts(PROMPT_PRESETS[next] || PROMPT_PRESETS.narrative);
        }
        promptEditBuffer = newBuffer;
        promptEditPreset = next;
        rebuildPromptLayerOptions(0, s.maxLayers, promptEditBuffer);
        loadPromptLayerIntoForm(0);
    });

    $(document).on('input', '#csc_modal_userPrompt, #csc_modal_systemPrompt', function () {
        if (promptEditBuffer && promptEditPreset !== 'custom') {
            promptEditPreset = 'custom';
            $('#csc_modal_promptPreset').val('custom');
        }
    });

    $(document).on('click', '#csc_promptSaveSlot', function () {
        const s = extension_settings[MODULE_NAME];
        const name = ($('#csc_modal_promptSlotName').val() || '').trim();
        if (!name) {
            toastr.warning('Enter a slot name first.', 'CharSummaryception');
            return;
        }
        if (!s.savedCustomPrompts) s.savedCustomPrompts = {};
        syncCurrentLayerToBuffer();
        s.savedCustomPrompts[name] = cloneLayerPrompts(promptEditBuffer);
        saveSettingsDebounced();
        refreshPromptSlotDropdown(name);
        toastr.success(`Saved prompt slot "${name}"`, 'CharSummaryception');
    });

    $(document).on('click', '#csc_promptLoadSlot', function () {
        const s = extension_settings[MODULE_NAME];
        const name = $('#csc_modal_promptSlots').val();
        if (!name) {
            toastr.warning('No slot selected.', 'CharSummaryception');
            return;
        }
        const slotValue = s.savedCustomPrompts?.[name];
        if (slotValue == null) {
            toastr.error(`Slot "${name}" not found.`, 'CharSummaryception');
            return;
        }
        promptEditBuffer = cloneLayerPrompts(slotValue);
        promptEditPreset = 'custom';
        $('#csc_modal_promptPreset').val('custom');
        $('#csc_modal_promptSlotName').val(name);
        rebuildPromptLayerOptions(0, s.maxLayers, promptEditBuffer);
        loadPromptLayerIntoForm(0);
        toastr.success(`Loaded prompt slot "${name}"`, 'CharSummaryception');
    });

    $(document).on('click', '#csc_promptDeleteSlot', async function () {
        const s = extension_settings[MODULE_NAME];
        const name = $('#csc_modal_promptSlots').val();
        if (!name) return;
        const confirm = await callGenericPopup(`Delete prompt slot "${name}"?`, POPUP_TYPE.CONFIRM);
        if (!confirm) return;
        if (s.savedCustomPrompts) delete s.savedCustomPrompts[name];
        saveSettingsDebounced();
        refreshPromptSlotDropdown();
        toastr.success(`Deleted prompt slot "${name}"`, 'CharSummaryception');
    });

    $(document).on('click', '#csc_promptExport', function () {
        syncCurrentLayerToBuffer();
        const data = {
            format: 'charSummaryception-per-layer-prompts',
            version: 1,
            layerPrompts: cloneLayerPrompts(promptEditBuffer),
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `prompts-${(extension_settings[MODULE_NAME].promptPreset || 'custom')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toastr.success('Per-layer prompts exported.', 'CharSummaryception');
    });

    $(document).on('click', '#csc_promptImport', function () {
        $('#csc_promptImportInput').click();
    });

    $(document).on('change', '#csc_promptImportInput', async function () {
        const file = this.files?.[0];
        if (!file) return;
        try {
            const text = await file.text();
            let parsed;
            try {
                parsed = JSON.parse(text);
            } catch {
                // Legacy plain-text import: treat as Layer 0 user prompt.
                parsed = { layerPrompts: { 0: { userPrompt: text } } };
            }
            const incoming = parsed.layerPrompts || parsed;
            promptEditBuffer = cloneLayerPrompts(incoming);
            const s = extension_settings[MODULE_NAME];
            promptEditPreset = 'custom';
            $('#csc_modal_promptPreset').val('custom');
            rebuildPromptLayerOptions(0, s.maxLayers, promptEditBuffer);
            loadPromptLayerIntoForm(0);
            toastr.success(`Imported "${file.name}"`, 'CharSummaryception');
        } catch (err) {
            toastr.error(`Import failed: ${err.message}`, 'CharSummaryception');
        } finally {
            this.value = '';
        }
    });

    function refreshPromptSlotDropdown(selected) {
        const s = extension_settings[MODULE_NAME];
        const keys = Object.keys(s.savedCustomPrompts || {});
        const opts = keys.length > 0
            ? keys.map(k => `<option value="${escapeAttr(k)}" ${k === selected ? 'selected' : ''}>${escapeHtml(k)}</option>`).join('')
            : '<option value="">(no saved slots)</option>';
        $('#csc_modal_promptSlots').html(opts);
    }

    updateStatusDisplay();
    updateActivityLogDisplay();
});
