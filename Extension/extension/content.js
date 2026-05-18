// ============================================
// ISY Content Script
// ============================================

(function() {
  if (!window.ISY) {
    console.error('[ISY] Namespace not loaded - check manifest js order');
    return;
  }

  const ISY = window.ISY;
  const adapter = ISY.state.currentAdapter;

  console.log(`[ISY] Ready on ${window.location.hostname} (${adapter.name})`);

  const TEXT_ANALYSIS_ENABLED = false;
  const LOCAL_API_BASE = 'http://localhost:8000';
  const ANALYZE_CONCURRENCY = 4;
  const progress = { pending: 0, done: 0, failed: 0 };
  const focusCursors = { high: 0, low: 0, failed: 0 };
  const analyzeQueue = [];
  let activeAnalyzeCount = 0;
  let analysisStarted = false;
  let followupObserverStarted = false;
  // URL 변경 감지 fallback — history hook / polling이 모두 실패해도 popup이
  // GET_STATE를 호출할 때마다 이 값과 location.href를 비교해 reset.
  let lastSeenUrl = location.href;

  function getItemKey(item) {
    if (item.mediaType === 'text') {
      return 'text:' + item.text.slice(0, 120).replace(/\s+/g, ' ');
    }
    if (item.mediaType === 'video' && item.platformMeta?.videoId) {
      return `video:${item.platformMeta.platform || 'unknown'}:${item.platformMeta.videoId}`;
    }
    return item.url;
  }

  function getCurrentItems() {
    const mediaItems = ISY.extractMedia();
    const textItems = TEXT_ANALYSIS_ENABLED ? ISY.extractText() : [];
    return { mediaItems, textItems, allItems: [...mediaItems, ...textItems] };
  }

  function isVideoFile(file) {
    return !!file && (
      file.type?.startsWith('video/')
      || /\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(file.name || '')
    );
  }

  function getUploadBadgeTarget(input) {
    return input.closest?.('ytcp-uploads-dialog, tp-yt-paper-dialog, ytcp-dialog')
      || document.querySelector('ytcp-uploads-dialog, tp-yt-paper-dialog, ytcp-dialog')
      || document.body;
  }

  function showPlatformDisclosureStatus(target, state, payloadOrMessage) {
    if (!target || !target.isConnected) return;
    let panel = target.querySelector?.('.isy-platform-disclosure');
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'isy-platform-disclosure';
      target.appendChild(panel);
    }

    panel.classList.toggle('isy-platform-disclosure-error', state === 'error');
    panel.classList.toggle('isy-platform-disclosure-done', state === 'done');

    if (state === 'loading') {
      panel.innerHTML = '';
      const title = document.createElement('strong');
      title.textContent = 'ISY 플랫폼 공개 라벨 등록 중';
      const body = document.createElement('span');
      body.textContent = '영상 분석 후 시청자에게 보일 공개 라벨을 등록합니다.';
      panel.append(title, body);
      return;
    }

    if (state === 'error') {
      panel.innerHTML = '';
      const title = document.createElement('strong');
      title.textContent = '플랫폼 공개 라벨 등록 실패';
      const body = document.createElement('span');
      body.textContent = String(payloadOrMessage || '서버 연결 또는 분석 중 오류가 발생했습니다.');
      panel.append(title, body);
      return;
    }

    const payload = payloadOrMessage || {};
    const disclosure = payload.disclosure || {};
    panel.innerHTML = '';
    const title = document.createElement('strong');
    title.textContent = '플랫폼 공개 라벨 등록 완료';
    const body = document.createElement('span');
    body.textContent = `${disclosure.viewer_title || 'AI 공개 라벨'} · ${disclosure.percent ?? 0}%`;
    const link = document.createElement('a');
    link.href = `${LOCAL_API_BASE}${payload.watch_url || `/demo/watch/${payload.video_id || ''}`}`;
    link.target = '_blank';
    link.rel = 'noreferrer';
    link.textContent = '시청자 화면 열기';
    panel.append(title, body, link);
  }

  async function postUploadFile(endpoint, file) {
    const form = new FormData();
    form.append('file', file, file.name || 'upload.mp4');
    const response = await fetch(`${LOCAL_API_BASE}${endpoint}`, {
      method: 'POST',
      body: form
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch {}
      const err = new Error(`API ${response.status}: ${detail}`);
      err.status = response.status;
      throw err;
    }
    return response.json();
  }

  async function analyzeUploadFile(file, target) {
    const key = `upload:${file.name}:${file.size}:${file.lastModified}`;
    if (ISY.state.analyzedUrls.has(key)) return;
    ISY.state.analyzedUrls.add(key);
    progress.pending += 1;

    if (target && target.isConnected) {
      ISY.ui.showLoadingBadge(target, key);
      showPlatformDisclosureStatus(target, 'loading');
    }

    try {
      let payload;
      let platformRegistered = true;
      try {
        payload = await postUploadFile('/api/platform/demo-upload', file);
      } catch (err) {
        if (err.status !== 404) throw err;
        platformRegistered = false;
        payload = await postUploadFile('/api/analyze/video-file', file);
      }
      const result = payload.disclosure || payload;
      ISY.ui.showResultBadge(target, result, key);
      if (platformRegistered) {
        showPlatformDisclosureStatus(target, 'done', payload);
      } else {
        showPlatformDisclosureStatus(target, 'error', 'Platform demo endpoint is not running. Restart python server.py to enable viewer disclosure links.');
      }
    } catch (err) {
      console.error('[ISY] Upload video analysis failed:', err);
      showPlatformDisclosureStatus(target, 'error', err.message);
      recordFailure(key, 'video', err.message, target);
    } finally {
      progress.done += 1;
      if (ISY.badges.refreshVisibility) ISY.badges.refreshVisibility();
    }
  }

  function startYouTubeStudioUploadWatcher() {
    if (location.hostname !== 'studio.youtube.com') return;
    document.addEventListener('change', event => {
      const input = event.target;
      if (!input || input.tagName !== 'INPUT' || input.type !== 'file') return;
      const file = Array.from(input.files || []).find(isVideoFile);
      if (!file) return;
      analyzeUploadFile(file, getUploadBadgeTarget(input));
    }, true);
  }

  function recordFailure(key, mediaType, error, element) {
    progress.failed += 1;
    const reason = error || '분석 실패';
    ISY.state.results.set(key, {
      isFake: false,
      fakeProb: 0,
      level: 'failed',
      mediaType,
      error: reason,
      element
    });
    // 페이지에 실패 배지 표시 (미디어만 — 텍스트는 high만 띄우는 기존 정책 유지)
    if (element && element.isConnected && mediaType !== 'text') {
      ISY.ui.showResultBadge(element, {
        level: 'failed',
        media_type: mediaType,
        error: reason
      }, key);
    }
  }

  async function analyzeItem(item, options) {
    const opts = options || {};
    const key = getItemKey(item);

    if (ISY.state.analyzedUrls.has(key) && !opts.force) return;
    ISY.state.analyzedUrls.add(key);

    progress.pending += 1;

    // 미디어 항목은 결과 도착 전까지 "분석 중" 배지로 상태를 보여줌.
    // 텍스트는 high 결과에만 배지가 붙는 기존 정책을 유지.
    if (item.mediaType !== 'text' && item.element && item.element.isConnected) {
      ISY.ui.showLoadingBadge(item.element, key);
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: 'ANALYZE_REQUEST',
        url: item.url || null,
        mediaType: item.mediaType,
        text: item.text || null,
        platformMeta: item.platformMeta || null,
        itemMeta: {
          isThumbnail: !!item.isThumbnail,
          source: item.source || null
        }
      });

      if (response && response.success) {
        if (item.mediaType === 'text') {
          ISY.ui.showTextBadge(item.element, response.result, key);
        } else {
          ISY.ui.showResultBadge(item.element, response.result, key);
        }
      } else {
        const error = !response ? 'Service Worker 응답 없음' : (response.error || '분석 실패');
        console.warn('[ISY] Analysis failed:', error);
        recordFailure(key, item.mediaType, error, item.element);
      }
    } catch (err) {
      console.error('[ISY] Message failed:', err);
      recordFailure(key, item.mediaType, err.message, item.element);
    } finally {
      progress.done += 1;
      if (ISY.badges.refreshVisibility) ISY.badges.refreshVisibility();
    }
  }

  // 동시 요청을 ANALYZE_CONCURRENCY개로 제한.
  // 서버 모델 추론이 sequential이라 폭주시키면 후순위 요청들이 클라이언트 25s timeout에 걸린다.
  function analyzeItems(items, options) {
    items.forEach(item => analyzeQueue.push({ item, options }));
    pumpAnalyzeQueue();
  }

  // 실패 항목 재분석. specificKeys가 주어지면 그 중 실패 상태인 것만, 아니면 전체 실패.
  // 팝업의 "실패 재시도" 버튼과 페이지 오버레이의 "재시도" 버튼이 공통 사용.
  function retryFailedItems(specificKeys) {
    const allFailed = Array.from(ISY.state.results.entries())
      .filter(([, r]) => r.level === 'failed')
      .map(([key]) => key);
    const targets = Array.isArray(specificKeys) && specificKeys.length
      ? allFailed.filter(k => specificKeys.includes(k))
      : allFailed;

    targets.forEach(key => {
      const stored = ISY.state.results.get(key);
      if (stored && stored.element && stored.element.dataset
          && stored.element.dataset.isyBadged === 'true') {
        ISY.badges.detach(stored.element);
      }
      ISY.state.analyzedUrls.delete(key);
      ISY.state.results.delete(key);
    });
    progress.failed = Math.max(0, progress.failed - targets.length);

    const itemsToRetry = getCurrentItems().allItems
      .filter(item => targets.includes(getItemKey(item)));
    analyzeItems(itemsToRetry, { force: true });

    return { ok: true, retried: itemsToRetry.length, requested: targets.length };
  }
  ISY.retryFailedItems = retryFailedItems;

  function pumpAnalyzeQueue() {
    while (activeAnalyzeCount < ANALYZE_CONCURRENCY && analyzeQueue.length > 0) {
      const { item, options } = analyzeQueue.shift();
      activeAnalyzeCount += 1;
      analyzeItem(item, options).finally(() => {
        activeAnalyzeCount -= 1;
        pumpAnalyzeQueue();
      });
    }
  }

  function getResultsByLevel(level) {
    return Array.from(ISY.state.results.values())
      .filter(result => {
        if (level === 'low') return result.level === 'low' || result.level === 'uncertain';
        return result.level === level;
      })
      .filter(result => result.element && result.element.isConnected);
  }

  function focusResult(level) {
    const results = getResultsByLevel(level);
    if (results.length === 0) {
      return { ok: false, error: '해당 분류의 콘텐츠를 찾지 못했습니다.' };
    }

    const index = focusCursors[level] % results.length;
    focusCursors[level] = (focusCursors[level] + 1) % results.length;
    const target = results[index].element;

    target.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
    ISY.ui.highlightElement(target);

    return {
      ok: true,
      level,
      index: index + 1,
      total: results.length
    };
  }

  function resetPageState() {
    progress.pending = 0;
    progress.done = 0;
    progress.failed = 0;
    focusCursors.high = 0;
    focusCursors.low = 0;
    focusCursors.failed = 0;
    ISY.resetAnalyzed();
    ISY.badges.removeAll();
    ISY.ui.removeAllTextBadges();
    ISY.state.results.clear();
  }

  function startFollowupScanning() {
    if (followupObserverStarted) return;
    followupObserverStarted = true;

    ISY.observer.startMutationObserver(({ mediaItems, textItems }) => {
      if (!analysisStarted) return;
      const enabledTextItems = TEXT_ANALYSIS_ENABLED ? textItems : [];
      analyzeItems([...mediaItems, ...enabledTextItems]);
    });

    // URL 변경 감지는 startFollowupScanning과 분리해 page-level 초기화에서 한 번만 등록
    // (아래 코드 참조). 여기서는 등록하지 않음 — 중복 등록 시 콜백이 여러 번 실행됨.
  }

  function scanAfterUserInteraction() {
    if (!analysisStarted) return;
    setTimeout(() => {
      restoreResultBadges();
      if (ISY.badges.refreshVisibility) ISY.badges.refreshVisibility();
      analyzeItems(getCurrentItems().allItems);
    }, 450);
  }

  function restoreResultBadges() {
    const items = getCurrentItems().allItems;
    for (const item of items) {
      if (item.mediaType === 'text') continue;
      const key = getItemKey(item);
      const stored = ISY.state.results.get(key);
      if (!stored || !item.element || !item.element.isConnected) continue;
      if (item.element.dataset?.isyBadged) continue;

      const result = stored.level === 'failed'
        ? {
          level: 'failed',
          media_type: stored.mediaType || item.mediaType,
          error: stored.error || '분석 실패'
        }
        : {
          fake_probability: stored.fakeProb || 0,
          media_type: stored.mediaType || item.mediaType
        };

      ISY.ui.showResultBadge(item.element, result, key);
    }
  }

  // ── 분석 상태 초기화 헬퍼 ────────────────────────────────
  // URL 변경 감지 경로와 사용자 명시 중단 경로에서 공통 사용.
  function sendStopBg() {
    try { chrome.runtime.sendMessage({ type: 'STOP_BG_ANALYSIS' }).catch(() => {}); } catch {}
  }

  // 사용자가 명시적으로 분석을 중단/초기화할 때 사용.
  // URL 변경 경로에서는 사용하지 않음 — observer가 계속 URL 변화를 감지해야 함.
  function stopAllAnalysis() {
    stopPickMode();
    analysisStarted = false;
    followupObserverStarted = false;
    analyzeQueue.length = 0;
    ISY.observer?.stopAll?.();
    sendStopBg();
  }

  let pickModeCleanup = null;

  function getVisibleRect(element) {
    if (!element || !element.isConnected) return null;
    const rect = element.getBoundingClientRect();
    const media = element.querySelector?.('img, video');
    const mediaRect = media?.getBoundingClientRect();
    const sourceRect = mediaRect && mediaRect.width > 1 && mediaRect.height > 1 ? mediaRect : rect;

    let left = Math.max(0, sourceRect.left);
    let top = Math.max(0, sourceRect.top);
    let right = Math.min(window.innerWidth, sourceRect.right);
    let bottom = Math.min(window.innerHeight, sourceRect.bottom);

    let parent = (media || element).parentElement;
    while (parent && parent !== document.body && parent !== document.documentElement) {
      const style = window.getComputedStyle(parent);
      const clipsX = style.overflowX !== 'visible' && style.overflowX !== 'clip-visible';
      const clipsY = style.overflowY !== 'visible' && style.overflowY !== 'clip-visible';
      if (clipsX || clipsY) {
        const parentRect = parent.getBoundingClientRect();
        if (clipsX) {
          left = Math.max(left, parentRect.left);
          right = Math.min(right, parentRect.right);
        }
        if (clipsY) {
          top = Math.max(top, parentRect.top);
          bottom = Math.min(bottom, parentRect.bottom);
        }
      }
      parent = parent.parentElement;
    }

    const width = right - left;
    const height = bottom - top;

    if (width <= 1 || height <= 1) return null;
    return { left, top, right, bottom, width, height };
  }

  function analyzeCurrentVideo() {
    const candidates = getCurrentItems().mediaItems
      .filter(item => item.mediaType === 'video' && item.element && item.element.isConnected)
      .map(item => ({ item, rect: getVisibleRect(item.element) }))
      .filter(candidate => candidate.rect);

    if (candidates.length === 0) {
      return { ok: false, error: '분석할 현재 영상을 찾지 못했습니다.' };
    }

    candidates.sort((a, b) => (b.rect.width * b.rect.height) - (a.rect.width * a.rect.height));
    const item = candidates[0].item;
    const key = getItemKey(item);
    if (item.element?.dataset?.isyBadged) {
      ISY.badges.detach(item.element);
    }
    ISY.state.analyzedUrls.delete(key);
    ISY.state.results.delete(key);
    analyzeItems([item], { force: true });
    return { ok: true, count: 1 };
  }

  function getCandidateForPoint(candidates, clientX, clientY) {
    let best = null;
    let bestArea = Infinity;

    for (const candidate of candidates) {
      const rect = candidate.rect;
      if (!rect) continue;
      if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
        continue;
      }
      const area = Math.max(1, rect.width * rect.height);
      if (area < bestArea) {
        best = candidate;
        bestArea = area;
      }
    }

    return best;
  }

  function showPickToast(text) {
    let toast = document.querySelector('.isy-pick-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'isy-pick-toast';
      toast.setAttribute('role', 'status');
      document.body.appendChild(toast);
    }
    toast.textContent = text;
  }

  function stopPickMode() {
    if (pickModeCleanup) {
      pickModeCleanup();
      pickModeCleanup = null;
    }
  }

  function startPickAnalysis() {
    stopPickMode();

    const initialItems = getCurrentItems().allItems;
    if (initialItems.length === 0) {
      return { ok: false, error: '분석할 이미지나 영상을 찾지 못했습니다.' };
    }

    const overlay = document.createElement('div');
    overlay.className = 'isy-pick-overlay';
    document.documentElement.appendChild(overlay);

    const candidates = [];
    const candidatesByKey = new Map();

    function createCandidate(item) {
      const key = getItemKey(item);
      const existing = candidatesByKey.get(key);
      if (existing) {
        existing.item = item;
        return existing;
      }

      const marker = document.createElement('div');
      marker.className = 'isy-pick-candidate';
      overlay.appendChild(marker);

      const candidate = { key, item, rect: null, marker };
      candidates.push(candidate);
      candidatesByKey.set(key, candidate);
      return candidate;
    }

    function syncCandidates() {
      getCurrentItems().allItems.forEach(createCandidate);
    }

    function updateCandidateMarker(candidate) {
      const rect = getVisibleRect(candidate.item.element);
      candidate.rect = rect;
      if (!rect) {
        candidate.marker.hidden = true;
        return;
      }
      candidate.marker.hidden = false;
      candidate.marker.style.left = `${rect.left}px`;
      candidate.marker.style.top = `${rect.top}px`;
      candidate.marker.style.width = `${rect.width}px`;
      candidate.marker.style.height = `${rect.height}px`;
    }

    function updateCandidateMarkers() {
      syncCandidates();
      candidates.forEach(updateCandidateMarker);
    }

    initialItems.forEach(createCandidate);
    updateCandidateMarkers();

    let hovered = null;
    showPickToast('파란 박스를 계속 클릭해 원하는 항목만 분석하세요. Esc로 종료합니다.');
    document.body.classList.add('isy-pick-mode');

    function setHovered(item) {
      const next = item || null;
      if (hovered === next) return;
      if (hovered) {
        hovered.marker?.classList.remove('isy-pick-candidate-active');
        if (hovered.item.element?.isConnected) hovered.item.element.classList.remove('isy-pick-target');
      }
      hovered = next;
      const hasTarget = hovered && hovered.item.element?.isConnected;
      overlay.classList.toggle('isy-pick-overlay-ready', !!hasTarget);
      overlay.classList.toggle('isy-pick-overlay-empty', !hasTarget);
      if (hasTarget) {
        hovered.marker?.classList.add('isy-pick-candidate-active');
        hovered.item.element.classList.add('isy-pick-target');
        showPickToast('선택 가능: 클릭하면 이 항목을 분석합니다. 계속 선택할 수 있고 Esc로 종료합니다.');
      } else {
        showPickToast('파란 박스가 있는 이미지나 영상 위로 마우스를 이동하세요. Esc로 취소할 수 있습니다.');
      }
    }

    function cleanup() {
      document.removeEventListener('mousemove', onMouseMove, true);
      window.removeEventListener('scroll', onViewportChange, true);
      window.removeEventListener('resize', onViewportChange, true);
      document.removeEventListener('pointerdown', onPointerBlock, true);
      document.removeEventListener('mousedown', onPointerBlock, true);
      document.removeEventListener('mouseup', onPointerBlock, true);
      document.removeEventListener('pointerup', onPointerUp, true);
      document.removeEventListener('click', onClick, true);
      document.removeEventListener('keydown', onKeyDown, true);
      document.body.classList.remove('isy-pick-mode');
      if (hovered && hovered.item.element?.isConnected) hovered.item.element.classList.remove('isy-pick-target');
      if (overlay.isConnected) overlay.remove();
      document.querySelectorAll('.isy-pick-toast').forEach(el => el.remove());
    }

    function onMouseMove(event) {
      updateCandidateMarkers();
      setHovered(getCandidateForPoint(candidates, event.clientX, event.clientY));
    }

    function onViewportChange() {
      updateCandidateMarkers();
      restoreResultBadges();
      if (hovered && !hovered.rect) setHovered(null);
    }

    function blockEvent(event) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }

    function onPointerBlock(event) {
      blockEvent(event);
    }

    function analyzePickedTarget(event) {
      const candidate = getCandidateForPoint(candidates, event.clientX, event.clientY);
      if (!candidate) return;
      blockEvent(event);
      const item = candidate.item;

      const key = getItemKey(item);
      if (item.element?.dataset?.isyBadged) {
        ISY.badges.detach(item.element);
      }
      ISY.state.analyzedUrls.delete(key);
      ISY.state.results.delete(key);
      analyzeItems([item], { force: true });
      showPickToast('분석을 시작했습니다. 다른 파란 박스도 계속 선택할 수 있습니다. Esc로 종료합니다.');
    }

    function onPointerUp(event) {
      analyzePickedTarget(event);
    }

    function onClick(event) {
      blockEvent(event);
    }

    function onKeyDown(event) {
      if (event.key === 'Escape') stopPickMode();
    }

    document.addEventListener('mousemove', onMouseMove, true);
    window.addEventListener('scroll', onViewportChange, true);
    window.addEventListener('resize', onViewportChange, true);
    document.addEventListener('pointerdown', onPointerBlock, true);
    document.addEventListener('mousedown', onPointerBlock, true);
    document.addEventListener('mouseup', onPointerBlock, true);
    document.addEventListener('pointerup', onPointerUp, true);
    document.addEventListener('click', onClick, true);
    document.addEventListener('keydown', onKeyDown, true);
    pickModeCleanup = cleanup;

    return { ok: true };
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || typeof message.type !== 'string') {
      return false;
    }

    switch (message.type) {
      case 'PING':
        sendResponse({ ok: true, adapter: adapter.name });
        return false;

      case 'GET_STATE': {
        // popup이 syncState 호출 시점에 URL 변경을 자체 감지해 reset.
        // history hook / polling이 모두 작동 안 해도 이 경로로 catch됨.
        if (location.href !== lastSeenUrl) {
          console.log(`[ISY] GET_STATE: URL 변경 감지 ${lastSeenUrl} → ${location.href} — 결과 초기화`);
          lastSeenUrl = location.href;
          if (analysisStarted || ISY.state.results.size > 0 || analyzeQueue.length > 0) {
            analysisStarted = false;
            analyzeQueue.length = 0;
            resetPageState();
            sendStopBg();
          }
        }
        const allResults = Array.from(ISY.state.results.values());
        const highCount = allResults.filter(r => r.level === 'high').length;
        // uncertain(애매)은 사용자 관점에서 "확실히 의심 아님"이므로 low로 묶어 카운트.
        const lowCount = allResults.filter(r => r.level === 'low' || r.level === 'uncertain').length;
        const failedCount = allResults.filter(r => r.level === 'failed').length;
        sendResponse({
          autoMode: false,
          analysisStarted,
          adapter: adapter.name,
          analyzedCount: ISY.state.analyzedUrls.size,
          pendingCount: Math.max(0, progress.pending - progress.done),
          badgeCount: ISY.badges.getCount(),
          highCount,
          lowCount,
          failedCount
        });
        return false;
      }

      case 'ANALYZE_ALL': {
        stopPickMode();
        analysisStarted = true;
        startFollowupScanning();
        const { mediaItems, textItems, allItems } = getCurrentItems();
        analyzeItems(allItems, { force: true });
        sendResponse({ count: mediaItems.length + textItems.length });
        return false;
      }

      case 'START_PICK_ANALYSIS':
        sendResponse(startPickAnalysis());
        return false;

      case 'ANALYZE_CURRENT_VIDEO':
        // 픽 모드가 켜진 상태에서 호출되면 overlay/이벤트 리스너가 살아있어
        // 후속 클릭/스크롤이 차단된다 — 분석 시작 전에 안전하게 정리.
        stopPickMode();
        // followup scanning은 켜지 않는다 — Shorts 스크롤 시 새 영상이 자동으로
        // 큐에 들어가 pending이 줄지 않게 되어 진행 상태가 영원히 '분석 중'으로 남음.
        // 현재 영상 분석은 단일 영상 1회 분석으로 명확히 끝나야 한다.
        sendResponse(analyzeCurrentVideo());
        return false;

      case 'FOCUS_RESULT':
        sendResponse(focusResult(message.level));
        return false;

      case 'RETRY_FAILED': {
        const result = retryFailedItems(message.failedKeys);
        sendResponse(result);
        return false;
      }

      case 'STOP_ANALYSIS':
        stopAllAnalysis();
        sendResponse({ ok: true });
        return false;

      case 'CLEAR_RESULTS':
        stopAllAnalysis();
        resetPageState();
        sendResponse({ ok: true });
        return false;

      case 'ANALYSIS_RESULT': {
        const targetEl = Array.from(document.querySelectorAll('img, video'))
          .find(el => ISY.getUrlFromElement(el) === message.url);
        if (targetEl) ISY.ui.showResultBadge(targetEl, message.result, message.url);
        sendResponse({ ok: !!targetEl });
        return false;
      }

      default:
        return false;
    }
  });

  document.addEventListener('click', scanAfterUserInteraction, true);
  document.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      scanAfterUserInteraction();
    }
  }, true);

  // URL 변경 감지 — 분석 종류(ANALYZE_ALL / PICK / CURRENT_VIDEO)와 무관하게 한 번 등록.
  // SPA 페이지 이동 시 이전 페이지의 누적 결과는 의미를 잃으므로 깨끗하게 정리한다.
  ISY.observer.startUrlChangeDetector(() => {
    lastSeenUrl = location.href;  // GET_STATE fallback과 중복 reset 방지
    const hasResults = ISY.state.results.size > 0;
    const hasQueue = analyzeQueue.length > 0;
    if (!analysisStarted && !hasResults && !hasQueue) return;

    console.log('[ISY] URL changed - resetting');
    analysisStarted = false;
    analyzeQueue.length = 0;
    resetPageState();
    sendStopBg();
  });

  let restoreTimer = null;
  function scheduleRestoreResultBadges() {
    if (ISY.state.results.size === 0 || restoreTimer) return;
    restoreTimer = setTimeout(() => {
      restoreTimer = null;
      restoreResultBadges();
    }, 120);
  }

  window.addEventListener('scroll', scheduleRestoreResultBadges, true);
  window.addEventListener('resize', scheduleRestoreResultBadges, true);
  startYouTubeStudioUploadWatcher();
})();
