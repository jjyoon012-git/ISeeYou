// ============================================
// ISY Background Service Worker
// ============================================

const API_BASE = "http://localhost:8000";
const FETCH_TIMEOUT_MS = 18000;
const VIDEO_FETCH_TIMEOUT_MS = 180000;
const RETRY_COUNT = 1;
const RETRY_BACKOFF_MS = 1000;

// 네트워크/일시 장애 대비.
// - 외부 STOP은 즉시 throw (사용자 의도). 자체 timeout은 1회 retry (서버 큐 적체 흡수).
// - 5xx 응답도 retry 대상 (fetch는 5xx에서 reject하지 않으므로 별도 분기).
// - timeout은 시도마다 새로 시작 (시도당 독립적 시간 보장).
async function fetchWithRetry(url, options, externalSignal, timeoutMs = FETCH_TIMEOUT_MS) {
  let lastErr;
  for (let i = 0; i <= RETRY_COUNT; i++) {
    if (externalSignal?.aborted) throw new DOMException('aborted', 'AbortError');

    const timeoutCtl = new AbortController();
    const timer = setTimeout(() => timeoutCtl.abort(), timeoutMs);
    const signal = externalSignal
      ? AbortSignal.any([externalSignal, timeoutCtl.signal])
      : timeoutCtl.signal;

    try {
      const response = await fetch(url, { ...options, signal });
      if (response.status >= 500 && response.status < 600 && i < RETRY_COUNT) {
        try { response.body?.cancel(); } catch {}
        await new Promise(r => setTimeout(r, RETRY_BACKOFF_MS));
        continue;
      }
      return response;
    } catch (err) {
      lastErr = err;
      if (err.name === 'AbortError') {
        if (externalSignal?.aborted) throw err;
        if (i === RETRY_COUNT) throw err;
        await new Promise(r => setTimeout(r, RETRY_BACKOFF_MS * 5));
        continue;
      }
      if (i === RETRY_COUNT) throw err;
      await new Promise(r => setTimeout(r, RETRY_BACKOFF_MS));
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastErr;
}

const memCache = new Map();
const MAX_CACHE_SIZE = 200;
const CACHE_TTL_MS = 30 * 60 * 1000;

// 진행 중인 분석 요청의 AbortController. STOP_BG_ANALYSIS로 일괄 중단.
const activeControllers = new Set();
function registerController() {
  const ctl = new AbortController();
  activeControllers.add(ctl);
  return ctl;
}
function releaseController(ctl) { activeControllers.delete(ctl); }
function abortAllActive() {
  for (const ctl of activeControllers) {
    try { ctl.abort(); } catch {}
  }
  activeControllers.clear();
}

// server.py의 resolve_image_url 정규화 로직을 미러링.
// sqp만 다른 동일 썸네일이나 dthumb 변형이 캐시 적중하도록 URL을 통일.
function normalizeUrlForCache(url) {
  if (!url) return url;
  try {
    const u = new URL(url);
    if (u.hostname.includes('ytimg.com')) {
      u.search = '';
      u.hash = '';
      return u.toString();
    }
    if (u.hostname.includes('pstatic.net') && u.pathname.includes('dthumb')) {
      const src = u.searchParams.get('src');
      if (src) {
        try {
          let decoded = src;
          for (let i = 0; i < 3; i++) {
            const next = decodeURIComponent(decoded);
            if (next === decoded) break;
            decoded = next;
          }
          return decoded.replace(/^"|"$/g, '');
        } catch {}
      }
    }
    return url;
  } catch {
    return url;
  }
}

// ============================================
// 이벤트 리스너 — 반드시 top-level에 등록 (SW reboot 대비)
// ============================================

chrome.runtime.onInstalled.addListener(details => {
  if (details.reason === 'install' || details.reason === 'update') setupContextMenus();
});
chrome.contextMenus.onClicked.addListener(handleContextMenuClick);
chrome.runtime.onMessage.addListener(handleMessage);

function setupContextMenus() {
  chrome.contextMenus.removeAll(() => {
    if (chrome.runtime.lastError) {
      console.warn('[ISY-BG] removeAll error:', chrome.runtime.lastError.message);
    }
    chrome.contextMenus.create({
      id: "isy-analyze",
      title: "I See You로 검증하기",
      contexts: ["image", "video"]
    }, () => {
      if (chrome.runtime.lastError) {
        console.error('[ISY-BG] create menu error:', chrome.runtime.lastError.message);
      }
    });
  });
}

async function handleContextMenuClick(info, tab) {
  if (info.menuItemId !== "isy-analyze" || !tab?.id) return;
  const { srcUrl: mediaUrl, mediaType } = info;
  if (!mediaUrl) return;

  await safeSendToTab(tab.id, { type: "ANALYSIS_START", url: mediaUrl });
  try {
    const result = await analyzeContent(mediaUrl, mediaType, null, null, tab.url);
    await safeSendToTab(tab.id, { type: "ANALYSIS_RESULT", url: mediaUrl, result });
  } catch (err) {
    console.error('[ISY-BG] Context menu analysis failed:', err.message);
    await safeSendToTab(tab.id, { type: "ANALYSIS_ERROR", url: mediaUrl, error: err.message });
  }
}

async function safeSendToTab(tabId, message) {
  try {
    return await chrome.tabs.sendMessage(tabId, message);
  } catch (err) {
    if (!err.message?.includes('Receiving end does not exist')) {
      console.error('[ISY-BG] sendMessage error:', err.message);
    }
    return null;
  }
}

function handleMessage(message, sender, sendResponse) {
  if (!message || typeof message.type !== 'string') return false;

  if (message.type === "ANALYZE_REQUEST") {
    analyzeContent(message.url, message.mediaType, message.text, message.platformMeta, sender?.tab?.url, message.itemMeta)
      .then(result => sendResponse({ success: true, result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }

  if (message.type === "ANALYZE_BATCH") {
    analyzeBatch(message.items)
      .then(results => sendResponse({ success: true, results }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }

  if (message.type === "STOP_BG_ANALYSIS") {
    abortAllActive();
    sendResponse({ ok: true });
    return false;
  }

  return false;
}

// ============================================
// 공통 캐시+fetch 패턴
// 세 가지 분석 타입(이미지/영상/텍스트)에서 공유되는 흐름:
// 캐시 조회 → POST 요청 → 결과 캐시 저장.
// ============================================
async function fetchCachedAnalysis(cacheKey, endpoint, body, timeoutMs = FETCH_TIMEOUT_MS) {
  const cached = await getCached(cacheKey);
  if (cached) return { ...cached, fromCache: true };

  const ctl = registerController();
  try {
    const response = await fetchWithRetry(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }, ctl.signal, timeoutMs);

    if (!response.ok) throw new Error(await formatApiError(response));
    const result = await response.json();
    setCached(cacheKey, stripLargeFields(result));
    return result;
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Analysis aborted or timed out');
    throw err;
  } finally {
    releaseController(ctl);
  }
}

async function analyzeContent(url, mediaType, text, platformMeta, pageUrl, itemMeta) {
  if (mediaType === 'text') return analyzeText(text);
  if (mediaType === 'video') return analyzeVideo(url, platformMeta);
  if (!url) throw new Error('URL required');

  const cacheKey = normalizeUrlForCache(url);
  const result = await fetchCachedAnalysis(cacheKey, '/api/analyze/image', {
    url,
    media_type: 'image',
    page_url: pageUrl || null
  });
  if (itemMeta) result.item_meta = itemMeta;
  return result;
}

// 계약:
//   url         - 직접 접근 가능한 영상 URL. blob: 이면 null.
//   platformMeta - { platform: 'youtube'|'instagram'|null, videoId: string|null }
async function analyzeVideo(url, platformMeta) {
  const videoId = platformMeta?.videoId ?? platformMeta?.video_id ?? null;
  const platform = platformMeta?.platform ?? null;
  const cacheKey = url
    ? normalizeUrlForCache(url)
    : `video:${platform ?? 'unknown'}:${videoId ?? 'unknown'}`;

  return fetchCachedAnalysis(cacheKey, '/api/analyze/video', {
    url: url || null,
    platform_meta: { platform, video_id: videoId }
  }, VIDEO_FETCH_TIMEOUT_MS);
}

async function analyzeText(text) {
  if (!text) throw new Error('Text required');
  const cacheKey = 'text:' + text.slice(0, 120).replace(/\s+/g, ' ');
  return fetchCachedAnalysis(cacheKey, '/api/analyze/text', { text });
}

async function analyzeBatch(items) {
  const BATCH_SIZE = 5;
  const results = [];
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const chunk = items.slice(i, i + BATCH_SIZE);
    const chunkResults = await Promise.all(
      chunk.map(item =>
        analyzeContent(item.url, item.mediaType, item.text, item.platformMeta)
          .catch(err => ({ url: item.url, error: err.message }))
      )
    );
    results.push(...chunkResults);
  }
  return results;
}

// ============================================
// 메모리 캐시 (LRU) + chrome.storage.session 영속화
// SW가 30초 idle로 종료되어도 캐시가 살아남는다.
// ============================================
const SESSION_CACHE_KEY = 'isyCache';
const FLUSH_DEBOUNCE_MS = 500;

const cacheReady = (async () => {
  try {
    const data = await chrome.storage.session.get(SESSION_CACHE_KEY);
    const entries = data?.[SESSION_CACHE_KEY]?.entries;
    if (Array.isArray(entries)) {
      const now = Date.now();
      for (const [key, value] of entries) {
        if (value && typeof value.timestamp === 'number'
            && (now - value.timestamp) <= CACHE_TTL_MS) {
          memCache.set(key, value);
        }
      }
    }
  } catch (err) {
    console.warn('[ISY-BG] cache hydrate failed:', err.message);
  }
})();

let flushTimer = null;
function scheduleFlush() {
  if (flushTimer) clearTimeout(flushTimer);
  flushTimer = setTimeout(flushCache, FLUSH_DEBOUNCE_MS);
}

async function flushCache() {
  flushTimer = null;
  try {
    await chrome.storage.session.set({
      [SESSION_CACHE_KEY]: { entries: Array.from(memCache.entries()) }
    });
  } catch (err) {
    console.warn('[ISY-BG] cache flush failed:', err.message);
  }
}

async function getCached(url) {
  await cacheReady;
  const entry = memCache.get(url);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
    memCache.delete(url);
    scheduleFlush();
    return null;
  }
  // LRU touch
  memCache.delete(url);
  memCache.set(url, entry);
  return entry.result;
}

async function setCached(url, result) {
  await cacheReady;
  while (memCache.size >= MAX_CACHE_SIZE) {
    memCache.delete(memCache.keys().next().value);
  }
  memCache.set(url, { result, timestamp: Date.now() });
  scheduleFlush();
}

function stripLargeFields(result) {
  const slim = { ...result };
  delete slim.heatmap;
  delete slim.heatmap_base64;
  delete slim.gradcam;
  return slim;
}

async function formatApiError(response) {
  let detail = response.statusText;
  try {
    const body = await response.json();
    detail = body.detail || detail;
  } catch {}
  return `API ${response.status}: ${detail}`;
}
