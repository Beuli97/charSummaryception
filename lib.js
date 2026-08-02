/**
 * CharSummaryception — Pure utility functions.
 *
 * Side-effect-free helpers: escaping, memory parsing/serialization, passage
 * building (perception-aware), perception map helpers. Nothing here touches
 * the DOM, SillyTavern globals, or network.
 */

export function escapeAttr(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

export function unescapeAttr(text) {
    return String(text)
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&gt;/g, '>')
        .replace(/&lt;/g, '<')
        .replace(/&amp;/g, '&');
}

export function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

export function stripNonDiegetic(text) {
    if (!text) return '';
    return text
        .replace(/```[\s\S]*?```/g, '')
        .replace(/<details[\s\S]*?<\/details>/gi, '')
        .replace(/^[ \t]*\|.*\|[ \t]*$/gm, '')
        .replace(/<[^>]*>/g, '')
        .replace(/\n{3,}/g, '\n\n');
}

/**
 * Parse <memory> blocks from raw markdown content.
 * Supports optional layer, range, promoted, date, chat attributes.
 *
 * @param {string} content Raw file content.
 * @returns {Array<{chat: string, date: string, bullets: string[], layer?: number, range?: [number, number], promoted?: boolean, blockPromotion?: boolean}>}
 */
export function parseMemories(content) {
    if (!content || !content.trim()) return [];

    const blocks = [];
    const regex = /<memory\b([^>]*)>([\s\S]*?)<\/memory>/gi;
    let match;

    while ((match = regex.exec(content)) !== null) {
        const attrs = match[1];
        const body = match[2];

        const chatMatch = attrs.match(/chat="([^"]*)"/);
        const dateMatch = attrs.match(/date="([^"]*)"/);
        const layerMatch = attrs.match(/layer="([^"]*)"/);
        const rangeMatch = attrs.match(/range="([^"]*)"/);
        const promotedMatch = attrs.match(/promoted="([^"]*)"/);
        const seedFromLayerMatch = attrs.match(/seedFromLayer="([^"]*)"/);
        const blockPromotionMatch = attrs.match(/blockPromotion="([^"]*)"/);

        const chat = chatMatch ? unescapeAttr(chatMatch[1]) : 'unknown';
        const date = dateMatch ? unescapeAttr(dateMatch[1]) : '';
        const layer = layerMatch ? Number(layerMatch[1]) : 0;
        const promoted = promotedMatch ? unescapeAttr(promotedMatch[1]) === 'true' : false;
        const blockPromotion = blockPromotionMatch ? unescapeAttr(blockPromotionMatch[1]) === 'true' : false;
        const seedFromLayer = seedFromLayerMatch ? Number(seedFromLayerMatch[1]) : undefined;

        let range = undefined;
        if (rangeMatch) {
            const [s, e] = rangeMatch[1].split('-').map(Number);
            if (Number.isFinite(s) && Number.isFinite(e)) range = [s, e];
        }

        const bullets = body.split('\n')
            .map(line => line.trim())
            .filter(line => line.startsWith('- ') || /^\[.*?\]\s*-\s/.test(line))
            .map(line => {
                const metaMatch = line.match(/^\[.*?\]\s*-\s+(.+)/);
                if (metaMatch) return metaMatch[1].trim();
                return line.slice(2).trim();
            })
            .filter(Boolean);

        if (bullets.length > 0) {
            const block = { chat, date, bullets, layer, promoted };
            if (blockPromotion) block.blockPromotion = true;
            if (range) block.range = range;
            if (seedFromLayer !== undefined) block.seedFromLayer = seedFromLayer;
            blocks.push(block);
        }
    }

    return blocks;
}

/**
 * Count total individual memories (bullets) across all blocks.
 */
export function countMemories(blocks) {
    return blocks.reduce((sum, b) => sum + b.bullets.length, 0);
}

/**
 * Build the attribute string for a <memory> tag.
 * Always emits chat + date. Emits layer/range/promoted when present.
 */
function buildMemoryAttrs(b) {
    let attrs = `chat="${escapeAttr(b.chat)}" date="${escapeAttr(b.date)}"`;
    if (Number.isFinite(b.layer)) attrs += ` layer="${b.layer}"`;
    if (b.range && b.range.length === 2) attrs += ` range="${b.range[0]}-${b.range[1]}"`;
    if (b.promoted) attrs += ` promoted="true"`;
    if (b.blockPromotion) attrs += ` blockPromotion="true"`;
    if (Number.isFinite(b.seedFromLayer)) attrs += ` seedFromLayer="${b.seedFromLayer}"`;
    return attrs;
}

/**
 * Sanitize a bullet for one-line serialization: collapse embedded newlines
 * and strip anything resembling a <memory> tag so a stray bullet (LLM output
 * or manual entry) can't corrupt the file format.
 */
function sanitizeBullet(bullet) {
    return String(bullet ?? '')
        .replace(/<\/?memory\b[^>]*>/gi, '')
        .replace(/\s*\r?\n\s*/g, ' ')
        .trim();
}

/**
 * Serialize an array of memory blocks back to <memory> tag format.
 * Promoted blocks are omitted: their content already lives in a higher-layer
 * block; writing them would bloat the file and double-index in Vector Storage.
 * Blocks with no serializable bullets are dropped — parseMemories would
 * discard them on the next read anyway.
 */
export function serializeMemories(blocks) {
    if (!blocks || blocks.length === 0) return '';
    return blocks
        .filter(b => !b.promoted)
        .map(b => {
            const bulletsText = (b.bullets || [])
                .map(sanitizeBullet)
                .filter(Boolean)
                .map(bullet => `- ${bullet}`)
                .join('\n');
            if (!bulletsText) return null;
            return `<memory ${buildMemoryAttrs(b)}>\n${bulletsText}\n</memory>`;
        })
        .filter(Boolean)
        .join('\n\n');
}

/**
 * Build a passage of chat messages for a SPECIFIC witness character.
 * Filters to only messages the witness perceived (missing perception map =
 * everyone perceives). For partial-perception witnesses (see-only/hear-only),
 * the matching modality slice from `splits` is used instead of raw text.
 *
 * @param {Array} chatArray SillyTavern chat array.
 * @param {number} startIdx Start index (inclusive).
 * @param {number} endIdx End index (inclusive).
 * @param {string} witnessAvatar The avatar of the perceiving character.
 * @param {function} getSpeakerAvatarFn Callback (msg) => avatar string.
 * @param {string} userName The user's display name.
 * @param {Map<number, {visual: string, audio: string}>} [splits]
 *        Optional modality-pure slices keyed by message index (built by
 *        summarizer.extractModalitySplit). Messages absent fall back to raw text.
 * @returns {{ text: string, messageCount: number }}
 */
export function buildPassageForWitness(chatArray, startIdx, endIdx, witnessAvatar, getSpeakerAvatarFn, userName, splits) {
    if (!chatArray || chatArray.length === 0) return { text: '', messageCount: 0 };

    const safeStart = Math.max(0, startIdx);
    const safeEnd = Math.min(chatArray.length - 1, endIdx);
    if (safeStart > safeEnd) return { text: '', messageCount: 0 };

    const lines = [];

    for (let i = safeStart; i <= safeEnd; i++) {
        const msg = chatArray[i];
        if (!msg || !msg.mes) continue;
        if (msg.is_system && !msg.is_user && !msg.name) continue;

        const speakerAvatar = getSpeakerAvatarFn(msg);
        if (!speakerAvatar) continue;

        // Use this message's own perception map. Missing extra.perception
        // means everyone perceives the message (the default).
        const msgPerception = msg.extra?.perception;
        const channel = getPerceptionChannel(witnessAvatar, speakerAvatar, msgPerception);
        if (channel === 'none') continue;

        const speakerName = msg.is_user ? (userName || 'User') : (msg.name || 'Character');

        let text;
        if ((channel === 'see' || channel === 'hear') && splits && splits.has(i)) {
            const slice = channel === 'see' ? splits.get(i).visual : splits.get(i).audio;
            text = (slice || '').trim();
        } else {
            text = stripNonDiegetic(msg.mes).trim();
        }
        if (!text) continue;
        lines.push(`${speakerName}: ${text}`);
    }

    return {
        text: lines.join('\n\n'),
        messageCount: lines.length,
    };
}

/**
 * Substitute template variables in a prompt string.
 */
export function substitutePromptTemplate(template, vars) {
    let result = template;
    for (const [key, value] of Object.entries(vars)) {
        if (value != null) {
            result = result.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), String(value));
        }
    }
    return result;
}

export function getTimestamp(date) {
    const now = date || new Date();
    return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}-${String(now.getUTCDate()).padStart(2, '0')} ${String(now.getUTCHours()).padStart(2, '0')}:${String(now.getUTCMinutes()).padStart(2, '0')} UTC`;
}

/**
 * Strip reasoning tags, thinking blocks, and other model artifacts.
 * Structured block patterns run BEFORE user-defined strip patterns so
 * <output> extraction fires before its delimiters are stripped.
 */
export function cleanSummarizerOutput(raw, stripPatterns = []) {
    if (!raw) return '';
    let text = raw;

    const blockPatterns = [
        { re: /<\|channel>thought[\s\S]*?<channel\|>/gi, keep: null },
        { re: /<thinking>[\s\S]*?<\/thinking>/gi, keep: null },
        { re: /<output>([\s\S]*?)<\/output>/gi, keep: '$1' },
        { re: /<reasoning>[\s\S]*?<\/reasoning>/gi, keep: null },
        { re: /<thought>[\s\S]*?<\/thought>/gi, keep: null },
        { re: /<reflect>[\s\S]*?<\/reflect>/gi, keep: null },
        { re: /<inner_monologue>[\s\S]*?<\/inner_monologue>/gi, keep: null },
    ];

    for (const { re, keep } of blockPatterns) {
        text = keep !== null ? text.replace(re, keep) : text.replace(re, '');
    }

    for (const pattern of stripPatterns) {
        if (!pattern) continue;
        text = text.split(pattern).join('');
    }

    text = text.replace(/\n{3,}/g, '\n').trim();
    return text;
}

/**
 * Detect if the LLM output is stuck in a repetition loop.
 * Returns true if the same line repeats 3+ times consecutively.
 */
export function isRepetitiveGarbage(text) {
    if (!text) return false;
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    let repeatCount = 1;
    for (let i = 1; i < lines.length; i++) {
        if (lines[i] === lines[i - 1]) {
            repeatCount++;
            if (repeatCount >= 3) return true;
        } else {
            repeatCount = 1;
        }
    }
    return false;
}

export const USER_AVATAR = 'user';

/**
 * Resolve WHICH channel a perceiver has for a given speaker's message.
 *
 * @param {string} perceiverAvatar
 * @param {string} speakerAvatar
 * @param {object} perception msg.extra.perception — keyed by speaker avatar.
 * @returns {'none' | 'see' | 'hear' | 'both'}
 */
export function getPerceptionChannel(perceiverAvatar, speakerAvatar, perception) {
    if (perceiverAvatar === speakerAvatar) return 'both';
    const entry = perception?.[speakerAvatar];
    if (!entry) return 'both';
    const seenBy = Array.isArray(entry.seenBy) ? entry.seenBy : [];
    const heardBy = Array.isArray(entry.heardBy) ? entry.heardBy : [];
    const sees = seenBy.includes(perceiverAvatar);
    const hears = heardBy.includes(perceiverAvatar);
    if (sees && hears) return 'both';
    if (sees) return 'see';
    if (hears) return 'hear';
    return 'none';
}

export function canPerceive(perceiverAvatar, speakerAvatar, perception) {
    return getPerceptionChannel(perceiverAvatar, speakerAvatar, perception) !== 'none';
}

/**
 * Does this message have ANY non-speaker target with partial (see-only or
 * hear-only) perception? If yes, the summarizer must compute a modality split.
 *
 * @param {object} msg SillyTavern message.
 * @param {Array<{avatar: string}>} targets All potential perceivers.
 * @param {function} getSpeakerAvatarFn (msg) => avatar.
 * @returns {boolean}
 */
export function hasPartialPerception(msg, targets, getSpeakerAvatarFn) {
    if (!msg || !targets || targets.length === 0) return false;
    const perception = msg.extra?.perception;
    if (!perception) return false;
    const speakerAvatar = getSpeakerAvatarFn(msg);
    if (!speakerAvatar) return false;
    const entry = perception[speakerAvatar];
    if (!entry) return false;
    const seenBy = Array.isArray(entry.seenBy) ? entry.seenBy : [];
    const heardBy = Array.isArray(entry.heardBy) ? entry.heardBy : [];
    for (const t of targets) {
        if (t.avatar === speakerAvatar) continue;
        const sees = seenBy.includes(t.avatar);
        const hears = heardBy.includes(t.avatar);
        if (sees !== hears) return true; // exactly one channel → partial
    }
    return false;
}
