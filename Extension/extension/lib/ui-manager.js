// ============================================
// UI Manager
// Renders result badges, detail overlays, and auto-mode status.
// ============================================

(function() {
  if (!window.ISY) return;

  const activeOverlays = new WeakMap();
  // 0.4~0.6 구간은 모델 신뢰도가 낮은 "애매" 영역으로 별도 표기.
  const HIGH_THRESHOLD = 0.6;
  const LOW_THRESHOLD = 0.4;

  const MEDIA_TYPE_LABELS = { image: '이미지', video: '영상', text: '텍스트' };

  function isDebugDetailsEnabled() {
    try {
      return window.ISY_DEBUG_DETAILS === true
        || window.localStorage?.getItem('isyDebugDetails') === 'true';
    } catch {
      return window.ISY_DEBUG_DETAILS === true;
    }
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderDebugDetails(result, adapterName) {
    if (!isDebugDetailsEnabled()) return '';

    const mediaType = MEDIA_TYPE_LABELS[result.media_type] || result.media_type || 'content';
    const rows = [
      ['콘텐츠 유형', mediaType],
      ['얼굴 크롭', result.crop_status],
      ['모델', result.model || adapterName]
    ].filter(([, value]) => value);

    if (rows.length === 0) return '';

    return `
      <div class="isy-debug-details" aria-label="개발자 디버그 정보">
        ${rows.map(([label, value]) => `
          <div class="isy-metric">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
          </div>
        `).join('')}
      </div>
    `;
  }

  function toUserMessage(raw) {
    if (!raw) return '알 수 없는 오류';
    const s = String(raw).toLowerCase();
    if (s.includes('timeout') || s.includes('timed out')) return '서버 응답 시간이 초과됐습니다';
    if (s.includes('download') || s.includes('502')) return '이미지를 가져오지 못했습니다';
    if (s.includes('aborted')) return '분석이 중단됐습니다';
    if (s.includes('service worker') || s.includes('응답 없음')) return '확장 프로그램을 새로고침 해주세요';
    return String(raw);
  }

  function addClickAndKeydown(element, handler) {
    element.addEventListener('click', handler);
    element.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') handler(e);
    });
  }

  function getFakeProbability(result) {
    return result.fake_probability != null ? result.fake_probability : 0;
  }

  function getResultMeta(result) {
    if (result && result.level === 'failed') {
      return {
        fakeProb: 0,
        isHigh: false,
        isFailed: true,
        level: 'failed',
        percent: 0,
        title: '분석 실패',
        badgeText: '실패',
        ariaText: '분석 실패, 상세 보기'
      };
    }
    const fakeProb = getFakeProbability(result);
    const percent = Math.round(fakeProb * 100);
    const isHigh = fakeProb >= HIGH_THRESHOLD;
    const isUncertain = !isHigh && fakeProb >= LOW_THRESHOLD;

    let level, title, badgeText, ariaText;
    if (isHigh) {
      level = 'high';
      title = 'AI 가능성 높음';
      badgeText = `의심 ${percent}%`;
      ariaText = `AI 생성 가능성 ${percent}%, 상세 보기`;
    } else if (isUncertain) {
      level = 'uncertain';
      title = 'AI 가능성 애매';
      badgeText = `애매 ${percent}%`;
      ariaText = `AI 생성 가능성 ${percent}%로 판정 애매, 상세 보기`;
    } else {
      level = 'low';
      title = 'AI 가능성 낮음';
      badgeText = '낮음';
      ariaText = `AI 생성 가능성 낮음, ${percent}%, 상세 보기`;
    }

    return {
      fakeProb,
      isHigh,
      isUncertain,
      isFailed: false,
      level,
      percent,
      title,
      badgeText,
      ariaText
    };
  }

  function recordResult(itemKey, result, mediaType, element) {
    if (!itemKey) return;
    const meta = getResultMeta(result);
    window.ISY.state.results.set(itemKey, {
      isFake: meta.isHigh,
      fakeProb: meta.fakeProb,
      level: meta.level,
      mediaType: mediaType || result.media_type || 'unknown',
      element
    });
  }

  function showLoadingBadge(targetElement, itemKey) {
    if (!targetElement || !targetElement.isConnected) return;
    if (targetElement.dataset.isyBadged) return;
    targetElement.dataset.isyBadged = 'loading';

    const badge = document.createElement('div');
    badge.className = 'isy-badge isy-loading-badge';
    badge.textContent = '분석 중';
    badge.title = '분석 중';
    badge.setAttribute('role', 'status');
    badge.setAttribute('aria-label', '분석 중');
    badge.dataset.isyLevel = 'loading';
    if (itemKey) badge.dataset.isyItemKey = itemKey;

    const ok = window.ISY.badges.attach(targetElement, badge);
    if (!ok) delete targetElement.dataset.isyBadged;
  }

  function showResultBadge(targetElement, result, itemKey) {
    if (!targetElement || !targetElement.isConnected) return;
    const ds = targetElement.dataset;
    if (ds.isyBadged === 'true') return;
    // loading 배지가 붙어있으면 결과 배지로 교체
    if (ds.isyBadged === 'loading') {
      window.ISY.badges.detach(targetElement);
    }
    ds.isyBadged = 'true';

    const meta = getResultMeta(result);
    if (itemKey) {
      window.ISY.state.results.set(itemKey, {
        isFake: meta.isHigh,
        fakeProb: meta.fakeProb,
        level: meta.level,
        mediaType: result.media_type || 'media',
        element: targetElement
      });
    }

    const badge = document.createElement('div');
    let levelClass;
    if (meta.isFailed) levelClass = 'isy-failed';
    else if (meta.isHigh) levelClass = 'isy-high';
    else if (meta.isUncertain) levelClass = 'isy-uncertain';
    else levelClass = 'isy-low';
    badge.className = `isy-badge ${levelClass}`;
    badge.textContent = meta.badgeText;
    badge.title = meta.ariaText;
    badge.setAttribute('role', 'button');
    badge.setAttribute('tabindex', '0');
    badge.setAttribute('aria-label', meta.ariaText);
    badge.dataset.isyLevel = meta.level;
    if (itemKey) badge.dataset.isyItemKey = itemKey;

    function open(e) {
      e.preventDefault();
      e.stopPropagation();
      showDetailOverlay(targetElement, result, itemKey);
    }
    addClickAndKeydown(badge, open);

    const ok = window.ISY.badges.attach(targetElement, badge);
    if (!ok) delete targetElement.dataset.isyBadged;
  }

  function showTextBadge(element, result, itemKey) {
    if (!element || !element.isConnected) return;
    if (element.dataset.isyTextBadged === 'true') return;

    const meta = getResultMeta(result);
    recordResult(itemKey, result, 'text', element);
    if (!meta.isHigh) return;

    element.dataset.isyTextBadged = 'true';
    element.classList.add('isy-text-highlighted');

    const cs = window.getComputedStyle(element);
    if (cs.position === 'static') {
      element.dataset.isyOriginalPosition = 'static';
      element.style.position = 'relative';
    }

    const badge = document.createElement('span');
    badge.className = 'isy-text-badge';
    badge.textContent = `의심 ${meta.percent}%`;
    badge.setAttribute('aria-label', meta.ariaText);
    badge.setAttribute('role', 'button');
    badge.setAttribute('tabindex', '0');
    badge.title = meta.ariaText;
    badge.dataset.isyLevel = meta.level;

    function openText(e) {
      e.preventDefault();
      e.stopPropagation();
      showDetailOverlay(element, result, itemKey);
    }
    addClickAndKeydown(badge, openText);

    element.appendChild(badge);
  }

  function showDetailOverlay(targetElement, result, itemKey) {
    const existing = activeOverlays.get(targetElement);
    if (existing && existing.isConnected) {
      existing.remove();
      activeOverlays.delete(targetElement);
      return;
    }

    document.querySelectorAll('.isy-detail-overlay').forEach(el => el.remove());

    const overlay = document.createElement('div');
    overlay.className = 'isy-detail-overlay';

    const meta = getResultMeta(result);
    const adapterName = window.ISY.state.currentAdapter.name;

    if (meta.isFailed) {
      const safeReason = toUserMessage(result?.error)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      overlay.innerHTML = `
        <div class="isy-detail-card failed" role="dialog" aria-label="분석 실패">
          <button class="isy-close" aria-label="닫기">x</button>
          <h3>
            <span class="isy-detail-icon" aria-hidden="true">!</span>
            분석 실패
          </h3>
          <p class="isy-explain">사유: ${safeReason}</p>
          <button type="button" class="isy-retry-btn" aria-label="이 항목 재시도">재시도</button>
        </div>
      `;
    } else {
      const pctText = meta.fakeProb * 100;
      const cardLevel = meta.level;
      const iconChar = meta.isHigh ? '!' : (meta.isUncertain ? '?' : 'i');

      overlay.innerHTML = `
        <div class="isy-detail-card ${cardLevel}" role="dialog" aria-label="AI 생성 가능성 분석 결과">
          <button class="isy-close" aria-label="닫기">x</button>
          <h3>
            <span class="isy-detail-icon" aria-hidden="true">${iconChar}</span>
            ${meta.title}
          </h3>
          <p class="isy-explain">
            이 결과는 확정 판정이 아니라 모델이 계산한 AI 생성 가능성입니다.
          </p>
          <div class="isy-gauge-wrap">
            <div class="isy-gauge-header">
              <span>AI 생성 가능성</span>
              <strong class="isy-gauge-value">${pctText.toFixed(1)}%</strong>
            </div>
            <div class="isy-gauge-track" role="progressbar" aria-valuenow="${pctText.toFixed(1)}" aria-valuemin="0" aria-valuemax="100" aria-valuetext="AI 생성 가능성 ${pctText.toFixed(1)}%">
              <div class="isy-gauge-fill ${cardLevel}" style="width: ${pctText.toFixed(2)}%"></div>
              <div class="isy-gauge-threshold-low" aria-hidden="true"></div>
              <div class="isy-gauge-threshold-high" aria-hidden="true"></div>
            </div>
            <div class="isy-gauge-scale" aria-hidden="true">
              <span>0%</span><span>40%</span><span>60%</span><span>100%</span>
            </div>
          </div>
          ${renderDebugDetails(result, adapterName)}
        </div>
      `;
    }

    overlay.style.position = 'fixed';
    overlay.style.top = '-9999px';
    overlay.style.left = '-9999px';
    overlay.style.zIndex = String(window.ISY.CONSTANTS.OVERLAY_Z_INDEX);
    document.body.appendChild(overlay);

    positionOverlay(overlay, targetElement);

    function closeOverlay() {
      overlay.remove();
      activeOverlays.delete(targetElement);
      document.removeEventListener('keydown', onEscKey, true);
      document.removeEventListener('click', onOutsideClick, true);
    }
    function onEscKey(e) {
      if (e.key === 'Escape') closeOverlay();
    }
    function onOutsideClick(e) {
      if (overlay.contains(e.target)) return;
      if (e.target.closest('.isy-badge, .isy-text-badge')) return;
      closeOverlay();
    }

    overlay.querySelector('.isy-close').addEventListener('click', closeOverlay);
    const retryBtn = overlay.querySelector('.isy-retry-btn');
    if (retryBtn && itemKey) {
      retryBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        retryBtn.disabled = true;
        retryBtn.textContent = '재시도 중…';
        if (typeof window.ISY.retryFailedItems === 'function') {
          window.ISY.retryFailedItems([itemKey]);
        }
        closeOverlay();
      });
    }
    document.addEventListener('keydown', onEscKey, true);
    setTimeout(() => document.addEventListener('click', onOutsideClick, true), 0);
    activeOverlays.set(targetElement, overlay);
  }

  function positionOverlay(overlay, targetElement) {
    const card = overlay.querySelector('.isy-detail-card');
    const cardRect = card.getBoundingClientRect();
    const cardW = cardRect.width;
    const cardH = cardRect.height;
    const rect = targetElement.getBoundingClientRect();
    const margin = 12;

    if (window.innerWidth <= 520) {
      overlay.style.left = '12px';
      overlay.style.right = '12px';
      overlay.style.bottom = '12px';
      overlay.style.top = 'auto';
      return;
    }

    let left = rect.right + margin;
    if (left + cardW > window.innerWidth - margin) {
      left = rect.left - cardW - margin;
    }
    if (left < margin) {
      left = Math.max(margin, Math.min(rect.left, window.innerWidth - cardW - margin));
    }

    let top = rect.top;
    if (top + cardH > window.innerHeight - margin) {
      top = window.innerHeight - cardH - margin;
    }
    if (top < margin) top = margin;

    overlay.style.top = `${top}px`;
    overlay.style.left = `${left}px`;
  }

  function highlightElement(element) {
    if (!element || !element.isConnected) return false;
    element.classList.add('isy-focus-target');
    setTimeout(() => {
      if (element.isConnected) element.classList.remove('isy-focus-target');
    }, 1800);
    return true;
  }

  function removeAllTextBadges() {
    document.querySelectorAll('[data-isy-text-badged="true"]').forEach(el => {
      const badge = el.querySelector('.isy-text-badge');
      if (badge) badge.remove();
      el.classList.remove('isy-text-highlighted');
      if (el.dataset.isyOriginalPosition === 'static') {
        el.style.position = '';
        delete el.dataset.isyOriginalPosition;
      }
      delete el.dataset.isyTextBadged;
    });
  }

  let autoIndicatorEl = null;
  let autoIndicatorOnDisable = null;

  function showAutoIndicator(onDisable) {
    if (autoIndicatorEl && autoIndicatorEl.isConnected) {
      autoIndicatorOnDisable = onDisable;
      return;
    }
    autoIndicatorOnDisable = onDisable;

    const el = document.createElement('div');
    el.className = 'isy-auto-indicator';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.innerHTML = `
      <span class="isy-auto-dot" aria-hidden="true"></span>
      <span class="isy-auto-text">자동 분석 대기 중</span>
      <button class="isy-auto-stop" aria-label="자동 모드 끄기" title="자동 모드 끄기">x</button>
    `;
    el.querySelector('.isy-auto-stop').addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      if (autoIndicatorOnDisable) autoIndicatorOnDisable();
    });

    document.body.appendChild(el);
    autoIndicatorEl = el;
  }

  function updateAutoIndicator(text) {
    if (!autoIndicatorEl || !autoIndicatorEl.isConnected) return;
    const t = autoIndicatorEl.querySelector('.isy-auto-text');
    if (t) t.textContent = text;
  }

  function hideAutoIndicator() {
    if (autoIndicatorEl && autoIndicatorEl.isConnected) autoIndicatorEl.remove();
    autoIndicatorEl = null;
    autoIndicatorOnDisable = null;
  }

  window.ISY.ui = {
    showResultBadge,
    showLoadingBadge,
    showTextBadge,
    removeAllTextBadges,
    showAutoIndicator,
    updateAutoIndicator,
    hideAutoIndicator,
    highlightElement
  };
})();
