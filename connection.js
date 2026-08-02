/**
 * CharSummaryception — Connection Utility
 *
 * Routes summarization requests through one of three backends:
 *   - default:     SillyTavern's generateRaw() (active connection)
 *   - openai:      OpenAI-compatible endpoint (streaming supported)
 *   - openrouter:  OpenRouter endpoint (streaming supported)
 *
 * Adapted from Summaryception's connectionutil.js (AGPL-3.0, by Lodactio).
 */

const LOG_PREFIX = '[CharSummaryception][Connection]';

class ConnectionError extends Error {
    constructor(message, { retryable = false, status = null } = {}) {
        super(message);
        this.name = 'ConnectionError';
        this.retryable = retryable;
        this.status = status;
    }
}

export { ConnectionError };

function escapeHtmlLocal(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Look up the OpenRouter model metadata for the given id and update the
 * capabilities hint + reasoning dropdown in the connection modal.
 * Rebuilds the reasoning <select> from the model's `reasoning.supported_efforts`.
 *
 * @param {string} modelId
 * @param {string} [storedReasoning] Persisted setting (used on initial hydrate
 *        so the saved value is honoured). When omitted, the live select value
 *        is used so user picks aren't overwritten on every keystroke.
 */
function updateOpenRouterModelCapabilities(modelId, storedReasoning) {
    const $caps = $('#csc_conn_openrouterCaps');
    const $reasoning = $('#csc_conn_openrouterReasoning');
    if (!$caps.length && !$reasoning.length) return;

    const id = String(modelId || '').trim();
    const preserve = storedReasoning !== undefined
        ? String(storedReasoning || '')
        : ($reasoning.val() || '');

    if (!id) {
        $caps.html('Pick or type a model ID to see its capabilities.');
        populateReasoningSelect($reasoning, ['minimal', 'low', 'medium', 'high'], false, '', preserve);
        $reasoning.prop('disabled', false);
        return;
    }

    const info = _openRouterModelCache.get(id);
    if (!info) {
        $caps.html('<b style="opacity:0.9;">Unknown model</b> — not in the fetched list. Capabilities not verified; reasoning may not work. Click the refresh button next to Model, or check the ID for typos.');
        populateReasoningSelect($reasoning, ['minimal', 'low', 'medium', 'high'], false, '', preserve);
        $reasoning.prop('disabled', false);
        return;
    }

    const params = Array.isArray(info.supported_parameters) ? info.supported_parameters : [];
    const supportsReasoning = params.includes('reasoning') || params.includes('include_reasoning');
    const reasoningInfo = info.reasoning || {};
    const mandatory = reasoningInfo.mandatory === true;
    const efforts = Array.isArray(reasoningInfo.supported_efforts) ? reasoningInfo.supported_efforts : [];
    const defaultEffort = reasoningInfo.default_effort || '';
    const defaultEnabled = reasoningInfo.default_enabled === true;
    const supportsMaxTokensFlag = reasoningInfo.supports_max_tokens === true;
    const supportsTemperature = params.includes('temperature');
    const maxOut = info?.top_provider?.max_completion_tokens;

    const bits = [];
    if (supportsReasoning) {
        bits.push('<b style="color:#2ecc71;">✓ Reasoning</b>');
    } else {
        bits.push('<b style="color:#e74c3c;">✗ No reasoning</b>');
    }
    if (mandatory) bits.push('<i>(mandatory)</i>');
    else if (defaultEnabled) bits.push('<i>(on by default)</i>');
    if (defaultEffort) bits.push(`default: ${escapeHtmlLocal(defaultEffort)}`);
    if (maxOut) bits.push(`Max output: ${Number(maxOut).toLocaleString()} tokens`);
    if (!supportsTemperature) bits.push('<b style="color:#f39c12;">⚠ ignores temperature</b>');
    if (supportsMaxTokensFlag && efforts.length === 0) {
        bits.push('<i>depth scales with Max tokens</i>');
    }
    $caps.html(bits.join(' · '));

    if (!supportsReasoning) {
        populateReasoningSelect($reasoning, [], false, '', preserve);
        $reasoning.prop('disabled', true);
    } else if (mandatory) {
        // Locked: no Off option, value forced to default_effort.
        populateReasoningSelect($reasoning, efforts, true, defaultEffort, preserve);
        $reasoning.prop('disabled', true);
    } else if (supportsMaxTokensFlag && efforts.length === 0) {
        populateReasoningSelect($reasoning, [], false, '', preserve);
        $reasoning.prop('disabled', true);
    } else if (efforts.length === 0) {
        // Only enabled/disabled is meaningful. Use a synthetic "on" effort so
        // the dropdown offers Off + On. sendViaOpenRouter() treats "on" as enabled:true.
        populateReasoningSelect($reasoning, ['on'], false, 'on', preserve);
        $reasoning.prop('disabled', false);
    } else {
        populateReasoningSelect($reasoning, efforts, false, defaultEffort, preserve);
        $reasoning.prop('disabled', false);
    }
}

/**
 * Rebuild the reasoning <select> from a list of effort IDs.
 * An Off option is prepended unless `mandatory` is true, OR the model
 * already lists "none" as an effort (in which case "none" IS the off value).
 *
 * @param {jQuery} $select
 * @param {string[]} efforts
 * @param {boolean} mandatory
 * @param {string} defaultEffort
 * @param {string} preserve
 */
function populateReasoningSelect($select, efforts, mandatory, defaultEffort, preserve) {
    const safePreserve = preserve == null ? '' : String(preserve);
    const hasNone = efforts.includes('none');
    const options = [];

    if (!mandatory && !hasNone) {
        options.push({ value: '', label: 'Off' });
    }
    for (const eff of efforts) {
        const label = eff.charAt(0).toUpperCase() + eff.slice(1);
        options.push({ value: eff, label });
    }

    $select.html(
        options.map(o =>
            `<option value="${escapeHtmlLocal(o.value)}">${escapeHtmlLocal(o.label)}</option>`
        ).join('')
    );

    const values = options.map(o => o.value);
    let pick;
    // Empty string ("Off") is a valid value — test membership directly.
    if (values.includes(safePreserve)) {
        pick = safePreserve;
    } else if (defaultEffort && values.includes(defaultEffort)) {
        pick = defaultEffort;
    } else {
        pick = values[0] ?? '';
    }
    $select.val(pick);
}

let _delegatedHandlersBound = false;

// Per-session cache of OpenRouter model metadata (id -> full /models object).
// Populated by fetchOpenRouterModels(); used to auto-gate UI controls and to
// decide which params to actually send in sendViaOpenRouter().
let _openRouterModelCache = new Map();

// Debounce handle for the #csc_conn_openrouterModel input handler.
let _openRouterModelInputTimer = null;
function bindDelegatedHandlers() {
    if (_delegatedHandlersBound) return;
    _delegatedHandlersBound = true;

    $(document).on('click', '#csc_conn_openrouterRefresh', async function () {
        const apiKey = $('#csc_conn_openrouterKey').val();
        const $btn = $(this);
        $btn.prop('disabled', true);
        try {
            const models = await fetchOpenRouterModels(apiKey);
            const $datalist = $('#csc_conn_openrouterModels');
            const opts = models.map(m =>
                `<option value="${escapeHtmlLocal(m.id)}">${escapeHtmlLocal(m.name || m.id)}</option>`
            ).join('');
            $datalist.html(opts);
            if (models.length > 0) toastr.success(`Found ${models.length} OpenRouter model(s)`, 'CharSummaryception');
            updateOpenRouterModelCapabilities($('#csc_conn_openrouterModel').val());
        } catch (err) {
            toastr.error(`Failed to fetch models: ${err.message}`, 'CharSummaryception');
        } finally {
            $btn.prop('disabled', false);
        }
    });

    // Debounced live capabilities hint when the user types/picks a model ID.
    $(document).on('input', '#csc_conn_openrouterModel', function () {
        clearTimeout(_openRouterModelInputTimer);
        const val = $(this).val();
        _openRouterModelInputTimer = setTimeout(() => updateOpenRouterModelCapabilities(val), 200);
    });

    $(document).on('click', '#csc_conn_testBtn', async function () {
        const source = $('#csc_modal_connectionSource').val();
        const $btn = $(this);
        const $status = $('#csc_conn_testStatus');
        $btn.prop('disabled', true);
        $status.text('Testing…').css('color', '');
        try {
            let result;
            if (source === 'openai') {
                result = await testOpenAIConnection($('#csc_conn_openaiUrl').val(), $('#csc_conn_openaiKey').val(), $('#csc_conn_openaiModel').val());
            } else if (source === 'openrouter') {
                result = await testOpenRouterConnection($('#csc_conn_openrouterKey').val(), $('#csc_conn_openrouterModel').val());
            } else {
                result = { success: false, message: 'No test available for this connection source.' };
            }
            if (result.success) {
                $status.html(`&check; ${escapeHtmlLocal(result.message)}`).css('color', '#2ecc71');
            } else {
                $status.html(`&cross; ${escapeHtmlLocal(result.message)}`).css('color', '#e74c3c');
            }
        } catch (err) {
            $status.html(`&cross; ${escapeHtmlLocal(err.message)}`).css('color', '#e74c3c');
        } finally {
            $btn.prop('disabled', false);
        }
    });
}

function proxiedUrl(url, useProxy = true) {
    if (!useProxy) return url;
    return `/proxy/${url}`;
}

/**
 * Resolve SillyTavern's request headers (auth, CSRF, etc.). ST's
 * getRequestHeaders is synchronous in current versions, but some forks
 * return a Promise; we await either case.
 */
async function getProxyHeaders() {
    try {
        const ctx = SillyTavern.getContext();
        if (typeof ctx.getRequestHeaders === 'function') {
            const headers = ctx.getRequestHeaders();
            const resolved = headers && typeof headers.then === 'function'
                ? await headers
                : headers;
            if (resolved && typeof resolved === 'object') return resolved;
        }
    } catch (e) { /* fallback below */ }
    return { 'Content-Type': 'application/json' };
}

/**
 * Send a summarization request using the configured connection.
 * @param {object} settings Extension settings with connection config.
 * @param {string} systemPrompt
 * @param {string} userPrompt
 * @param {AbortSignal} [signal] Optional abort signal propagated into fetch().
 *        Only the openai / openrouter backends honour it; the default
 *        (generateRaw) backend has no abort API exposed.
 * @param {function} [onProgress] Streaming-only activity callback.
 * @returns {Promise<string>}
 */
export async function sendSummarizerRequest(settings, systemPrompt, userPrompt, signal, onProgress) {
    const source = settings.connectionSource || 'default';

    switch (source) {
        case 'openai':
            return await sendViaOpenAI(settings.openaiUrl, settings.openaiKey, settings.openaiModel, systemPrompt, userPrompt, settings.openaiMaxTokens, signal, onProgress);
        case 'openrouter':
            return await sendViaOpenRouter(settings.openrouterKey, settings.openrouterModel, systemPrompt, userPrompt, settings.openrouterMaxTokens, settings.openrouterReasoning, signal, onProgress);
        case 'default':
        default:
            return await sendViaDefault(systemPrompt, userPrompt);
    }
}

async function sendViaDefault(systemPrompt, userPrompt) {
    const { generateRaw } = SillyTavern.getContext();

    if (!generateRaw) {
        throw new ConnectionError('generateRaw is not available.', { retryable: false });
    }

    // Pass object form — the positional-arg shim maps the 2nd arg to `api`, not `systemPrompt`.
    const options = { prompt: userPrompt, systemPrompt };
    const result = await generateRaw(options);

    if (!result || typeof result !== 'string') {
        throw new ConnectionError('generateRaw returned an empty or invalid response.', { retryable: true });
    }

    return result;
}

async function sendViaOpenAI(url, apiKey, model, systemPrompt, userPrompt, maxTokens, signal, onProgress) {
    if (!url) throw new ConnectionError('OpenAI URL not configured.', { retryable: false });
    if (!model) throw new ConnectionError('OpenAI model not set.', { retryable: false });

    const baseUrl = url.replace(/\/+$/, '');
    let endpoint = baseUrl;
    if (!endpoint.endsWith('/chat/completions')) {
        if (endpoint.endsWith('/v1')) endpoint += '/chat/completions';
        else if (!endpoint.includes('/chat/completions')) endpoint += '/v1/chat/completions';
    }

    const isLocal = /^https?:\/\/(localhost|127\.0\.0\.1|\[?::1\]?|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|0\.0\.0\.0)(:\d+)?/i.test(endpoint);

    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    const tokenLimit = maxTokens && maxTokens > 0 ? maxTokens : undefined;

    const requestBody = {
        model,
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
        ],
        temperature: 0.3,
        stream: true,
    };

    if (tokenLimit) requestBody.max_tokens = tokenLimit;

    const body = JSON.stringify(requestBody);

    const proxyHeaders = await getProxyHeaders();
    const localFetchOpts = (h) => ({ method: 'POST', headers: h, body, signal });

    let response;
    if (isLocal) {
        // Local endpoints: try the ST proxy first (avoids CORS), then direct.
        try {
            response = await fetch(proxiedUrl(endpoint), localFetchOpts({ ...proxyHeaders, ...headers }));
        } catch (proxyError) {
            if (signal?.aborted) throw proxyError;
            try {
                response = await fetch(endpoint, localFetchOpts(headers));
            } catch (directError) {
                if (signal?.aborted) throw directError;
                throw new ConnectionError(`Failed to connect to ${baseUrl}. Proxy: ${proxyError.message}. Direct: ${directError.message}.`, { retryable: true });
            }
        }
    } else {
        // Remote endpoints: direct first. On network/CORS failure, retry via
        // the ST proxy — many OpenAI-compatible endpoints don't send CORS
        // headers, so a direct browser fetch can never reach them.
        try {
            response = await fetch(endpoint, { method: 'POST', headers, body, signal });
        } catch (fetchError) {
            if (signal?.aborted) throw fetchError;
            try {
                response = await fetch(proxiedUrl(endpoint), localFetchOpts({ ...proxyHeaders, ...headers }));
            } catch (proxyError) {
                if (signal?.aborted) throw proxyError;
                throw new ConnectionError(`Failed to connect to ${baseUrl}. Direct: ${fetchError.message}. Proxy: ${proxyError.message}.`, { retryable: true });
            }
        }
    }

    if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        if (response.status === 401) throw new ConnectionError('OpenAI endpoint returned 401 Unauthorized. Check your API key.', { retryable: false, status: 401 });
        if (response.status === 403) throw new ConnectionError(`OpenAI endpoint returned 403 Forbidden: ${errorText}`, { retryable: false, status: 403 });
        throw new ConnectionError(`OpenAI request failed (${response.status}): ${errorText}`, { retryable: response.status >= 500 || response.status === 429, status: response.status });
    }

    const fullContent = await consumeSSEStream(response, 'OpenAI endpoint', onProgress);
    if (!fullContent.trim()) {
        throw new ConnectionError('OpenAI endpoint returned an empty response (streaming).', { retryable: true });
    }

    return fullContent;
}

export async function testOpenAIConnection(url, apiKey, model) {
    try {
        const result = await sendViaOpenAI(url, apiKey, model || 'test', 'You are a test assistant.', 'Respond with exactly: CONNECTION_OK', 100);
        return { success: true, message: `Connection successful! Response: "${result.substring(0, 100)}"` };
    } catch (error) {
        return { success: false, message: `Connection failed: ${error.message}` };
    }
}

const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

/**
 * Consume an OpenAI-compatible SSE (Server-Sent Events) response stream,
 * returning the concatenated `delta.content` payloads. Falls back to a
 * single text read if the body is non-streaming.
 *
 * @param {Response} response
 * @param {string} label readable backend name for error messages.
 * @param {function} [onProgress] Activity callback, invoked per chunk.
 * @returns {Promise<string>}
 */
async function consumeSSEStream(response, label, onProgress) {
    if (!response.body || typeof response.body.getReader !== 'function') {
        try {
            const text = await response.text();
            try { onProgress?.(); } catch { /* never let progress throw */ }
            try {
                const parsed = JSON.parse(text);
                const content = parsed?.choices?.[0]?.message?.content
                    ?? parsed?.choices?.[0]?.delta?.content;
                if (content) return String(content);
            } catch { /* fall through */ }
            return text;
        } catch (e) {
            throw new ConnectionError(`${label} returned a response with no readable body.`, { retryable: true });
        }
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = '';
    let buffer = '';

    const processLine = (line) => {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) return;
        const data = trimmed.slice(5).trim();
        if (data === '[DONE]') return;
        try {
            const parsed = JSON.parse(data);
            const choice = parsed.choices?.[0];
            if (!choice) return;
            // delta.content (streaming), message.content (non-streaming), or text (legacy).
            const delta = choice.delta?.content
                ?? choice.message?.content
                ?? choice.text;
            if (delta) fullContent += delta;
        } catch { /* skip unparseable chunks */ }
    };

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            try { onProgress?.(); } catch { /* never let progress throw */ }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) processLine(line);
        }
        // Flush any trailing content left in the buffer when the stream ends
        // without a final newline.
        if (buffer.trim()) processLine(buffer);
    } finally {
        try { reader.releaseLock(); } catch { /* already released */ }
    }

    return fullContent;
}

/**
 * Send a summarization request via OpenRouter.
 * OpenRouter is OpenAI-compatible but requires attribution headers
 * (HTTP-Referer, X-Title) and uses a fixed base URL.
 *
 * @param {string} apiKey
 * @param {string} model
 * @param {string} systemPrompt
 * @param {string} userPrompt
 * @param {number} [maxTokens] Response cap (0 / undefined = unlimited).
 * @param {string} [reasoning] Reasoning effort ID
 * @param {AbortSignal} [signal]
 * @param {function} [onProgress] Streaming activity callback.
 * @returns {Promise<string>}
 */
async function sendViaOpenRouter(apiKey, model, systemPrompt, userPrompt, maxTokens, reasoning, signal, onProgress) {
    if (!model) throw new ConnectionError('OpenRouter model not set.', { retryable: false });

    const endpoint = `${OPENROUTER_BASE_URL}/chat/completions`;

    // OpenRouter recommends both X-Title and HTTP-Referer attribution headers.
    const headers = {
        'Content-Type': 'application/json',
        'X-Title': 'CharSummaryception',
        'HTTP-Referer': (typeof window !== 'undefined' && window.location?.origin) || 'https://sillytavern.app',
    };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    const tokenLimit = maxTokens && maxTokens > 0 ? maxTokens : undefined;

    // Drop parameters the model doesn't support. Falls through conservatively
    // (sends temperature) when the cache misses.
    const modelInfo = _openRouterModelCache.get(model);
    const supportedParams = modelInfo && Array.isArray(modelInfo.supported_parameters)
        ? modelInfo.supported_parameters
        : null;
    const supportsTemperature = !supportedParams || supportedParams.includes('temperature');
    const supportsReasoning = !supportedParams
        || supportedParams.includes('reasoning')
        || supportedParams.includes('include_reasoning');
    const reasoningInfo = modelInfo?.reasoning || {};
    const reasoningEfforts = Array.isArray(reasoningInfo.supported_efforts) ? reasoningInfo.supported_efforts : [];
    const reasoningDefaultEffort = reasoningInfo.default_effort || '';
    const reasoningMandatory = reasoningInfo.mandatory === true;

    const requestBody = {
        model,
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
        ],
        stream: true,
    };

    if (supportsTemperature) requestBody.temperature = 0.3;

    if (tokenLimit) requestBody.max_tokens = tokenLimit;

    // Extended thinking. Effort IDs come from the cached supported_efforts;
    // invalid stored values fall back to the model's declared default_effort.
    // OpenRouter returns reasoning on delta.reasoning, which consumeSSEStream()
    // ignores, so the returned summary stays clean.
    if (supportsReasoning) {
        const userVal = String(reasoning || '').trim();

        let sendEnabled, sendEffort;

        if (reasoningMandatory) {
            sendEnabled = true;
            sendEffort = reasoningDefaultEffort
                || (reasoningEfforts.includes('high') ? 'high' : (reasoningEfforts[0] || ''));
        } else if (userVal === '' || userVal === 'none') {
            if (reasoningEfforts.includes('none')) {
                sendEnabled = true;
                sendEffort = 'none';
            } else {
                sendEnabled = false;
                sendEffort = undefined;
            }
        } else if (reasoningEfforts.length === 0) {
            sendEnabled = true;
            sendEffort = undefined;
        } else if (reasoningEfforts.includes(userVal)) {
            sendEnabled = true;
            sendEffort = userVal;
        } else if (reasoningDefaultEffort && reasoningEfforts.includes(reasoningDefaultEffort)) {
            sendEnabled = true;
            sendEffort = reasoningDefaultEffort;
        } else {
            sendEnabled = true;
            sendEffort = undefined;
        }

        if (sendEffort) {
            requestBody.reasoning = { enabled: true, effort: sendEffort };
        } else {
            requestBody.reasoning = { enabled: sendEnabled };
        }
    }

    const body = JSON.stringify(requestBody);

    let response;
    try {
        response = await fetch(endpoint, { method: 'POST', headers, body, signal });
    } catch (fetchError) {
        if (signal?.aborted) throw fetchError;
        throw new ConnectionError(`Failed to connect to OpenRouter: ${fetchError.message}`, { retryable: true });
    }

    if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        if (response.status === 401) throw new ConnectionError('OpenRouter returned 401 Unauthorized. Check your API key.', { retryable: false, status: 401 });
        if (response.status === 403) throw new ConnectionError(`OpenRouter returned 403 Forbidden: ${errorText}`, { retryable: false, status: 403 });
        throw new ConnectionError(`OpenRouter request failed (${response.status}): ${errorText}`, { retryable: response.status >= 500 || response.status === 429, status: response.status });
    }

    const fullContent = await consumeSSEStream(response, 'OpenRouter', onProgress);
    if (!fullContent.trim()) {
        throw new ConnectionError('OpenRouter returned an empty response (streaming).', { retryable: true });
    }

    return fullContent;
}

/**
 * Fetch available models from OpenRouter's /models endpoint.
 * Returns array of { id, name, ... } (OpenRouter uses `data` envelope, not `models`).
 */
export async function fetchOpenRouterModels(apiKey) {
    const endpoint = `${OPENROUTER_BASE_URL}/models`;
    const headers = {
        'X-Title': 'CharSummaryception',
        'HTTP-Referer': (typeof window !== 'undefined' && window.location?.origin) || 'https://sillytavern.app',
    };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    let response;
    try {
        response = await fetch(endpoint, { method: 'GET', headers });
    } catch (fetchError) {
        throw new Error(`Failed to connect to OpenRouter: ${fetchError.message}`);
    }

    if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(`Failed to fetch OpenRouter models (${response.status}): ${errorText}`);
    }

    let data;
    try {
        data = await response.json();
    } catch (e) {
        throw new Error(`OpenRouter /models returned a non-JSON response: ${e.message}`);
    }
    if (!data?.data || !Array.isArray(data.data)) {
        throw new Error('Unexpected response format from OpenRouter /models.');
    }

    // Cache full metadata so the UI can auto-gate controls and
    // sendViaOpenRouter() can drop unsupported params.
    _openRouterModelCache = new Map();
    for (const m of data.data) {
        if (m && m.id) _openRouterModelCache.set(m.id, m);
    }

    return data.data;
}

export async function testOpenRouterConnection(apiKey, model) {
    try {
        if (model) {
            const result = await sendViaOpenRouter(apiKey, model, 'You are a test assistant.', 'Respond with exactly: CONNECTION_OK', 100);
            return { success: true, message: `Connection successful! Response: "${result.substring(0, 100)}"` };
        }
        const models = await fetchOpenRouterModels(apiKey);
        return { success: true, message: `Connection successful! ${models.length} model(s) available.` };
    } catch (error) {
        return { success: false, message: `Connection failed: ${error.message}` };
    }
}

export function getConnectionDisplayName(settings) {
    switch (settings.connectionSource) {
        case 'default': return 'Default (Main API)';
        case 'openai': return `OpenAI: ${settings.openaiModel || '(no model)'}`;
        case 'openrouter': return `OpenRouter: ${settings.openrouterModel || '(no model)'}`;
        default: return 'Default (Main API)';
    }
}

/**
 * Render connection-mode-specific fields into #csc_modal_connectionDetails.
 * Wired up in index.js after the settings modal template is rendered.
 *
 * @param {string} source One of: default | openai | openrouter
 * @param {object} settings Extension settings for this module.
 * @returns {string} HTML to inject into the details container.
 */
export function renderConnectionDetailsHTML(source, settings) {
    bindDelegatedHandlers();

    const s = settings || {};
    if (!source || source === 'default') {
        return '<small class="charSummary_helperText">Uses your active SillyTavern connection — nothing else to configure.</small>';
    }

    if (source === 'openai') {
        return `
            <div class="charSummary_modalFieldGroup">
                <label><small>API URL</small></label>
                <small class="charSummary_helperText">Any OpenAI-compatible endpoint.</small>
                <input id="csc_conn_openaiUrl" class="text_pole" type="text" value="${escapeHtmlLocal(s.openaiUrl || '')}" placeholder="https://api.openai.com/v1" />
            </div>
            <div class="charSummary_modalFieldGroup">
                <label><small>API key</small></label>
                <small class="charSummary_helperText">Optional for local servers.</small>
                <input id="csc_conn_openaiKey" class="text_pole" type="password" value="${escapeHtmlLocal(s.openaiKey || '')}" placeholder="sk-…" />
            </div>
            <div class="charSummary_modalFieldGroup">
                <label><small>Model</small></label>
                <small class="charSummary_helperText">Model ID, e.g. <code>gpt-4o-mini</code>.</small>
                <input id="csc_conn_openaiModel" class="text_pole" type="text" value="${escapeHtmlLocal(s.openaiModel || '')}" placeholder="gpt-4o-mini" />
            </div>
            <div class="charSummary_modalFieldGroup">
                <label><small>Max tokens (0 = unlimited)</small></label>
                <small class="charSummary_helperText">Response cap per call. 0 = no limit.</small>
                <input id="csc_conn_openaiMaxTokens" class="text_pole" type="number" min="0" value="${Number(s.openaiMaxTokens) || 0}" />
            </div>
            <div class="charSummary_modalFieldGroup">
                <button type="button" class="menu_button" id="csc_conn_testBtn"><i class="fa-solid fa-plug"></i> Test Connection</button>
                <span id="csc_conn_testStatus"></span>
            </div>
        `;
    }

    if (source === 'openrouter') {
        const currentModel = escapeHtmlLocal(s.openrouterModel || '');
        return `
            <div class="charSummary_modalFieldGroup">
                <label><small>OpenRouter API key</small></label>
                <small class="charSummary_helperText">Get one at <a href="https://openrouter.ai/keys" target="_blank" rel="noopener">openrouter.ai/keys</a> — free models are available.</small>
                <input id="csc_conn_openrouterKey" class="text_pole" type="password" value="${escapeHtmlLocal(s.openrouterKey || '')}" placeholder="sk-or-v1-…" />
            </div>
            <div class="charSummary_modalFieldGroup">
                <label><small>Model</small></label>
                <small class="charSummary_helperText">Type an ID or click refresh to populate the list. Free models end with <code>:free</code>.</small>
                <div class="charSummary_inlineRow">
                    <input id="csc_conn_openrouterModel" class="text_pole" type="text" list="csc_conn_openrouterModels" value="${currentModel}" placeholder="anthropic/claude-3.5-sonnet" />
                    <button type="button" class="menu_button" id="csc_conn_openrouterRefresh"><i class="fa-solid fa-arrows-rotate"></i></button>
                </div>
                <datalist id="csc_conn_openrouterModels"></datalist>
                <small id="csc_conn_openrouterCaps" class="charSummary_helperText">Pick or type a model ID to see its capabilities.</small>
            </div>
            <div class="charSummary_modalFieldGroup">
                <label><small>Max tokens (0 = unlimited)</small></label>
                <small class="charSummary_helperText">Response cap per call. 0 = no limit.</small>
                <input id="csc_conn_openrouterMaxTokens" class="text_pole" type="number" min="0" value="${Number(s.openrouterMaxTokens) || 0}" />
            </div>
            <div class="charSummary_modalFieldGroup">
                <label><small>Reasoning (thinking)</small></label>
                <small class="charSummary_helperText">Extended thinking for models that support it. Effort options populate automatically from each model's declared capabilities.</small>
                <select id="csc_conn_openrouterReasoning" class="text_pole">
                    <option value="">Off</option>
                </select>
            </div>
            <div class="charSummary_modalFieldGroup">
                <button type="button" class="menu_button" id="csc_conn_testBtn"><i class="fa-solid fa-plug"></i> Test Connection</button>
                <span id="csc_conn_testStatus"></span>
            </div>
        `;
    }

    return '';
}

/**
 * After renderConnectionDetailsHTML() is injected into the DOM, call this to
 * wire up dynamic parts (initial OpenRouter model fetch).
 */
export async function hydrateConnectionDetails(source, settings) {
    if (source === 'openrouter') {
        if (settings.openrouterKey) {
            try {
                const models = await fetchOpenRouterModels(settings.openrouterKey);
                const $datalist = $('#csc_conn_openrouterModels');
                const opts = models.map(m =>
                    `<option value="${escapeHtmlLocal(m.id)}">${escapeHtmlLocal(m.name || m.id)}</option>`
                ).join('');
                $datalist.html(opts);
            } catch (e) {
                console.warn(LOG_PREFIX, 'Initial OpenRouter model fetch failed — user can retry via Refresh button:', e);
            }
        }
        // Sync the caps hint + reasoning gating to the current model.
        updateOpenRouterModelCapabilities(settings.openrouterModel, settings.openrouterReasoning);
    }
}
