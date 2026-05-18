// ============================================
// Badge Manager
// Attaches result badges without changing media DOM structure.
// On Instagram carousels and YouTube Shorts, only the badge for the visible media is shown.
// ============================================

(function() {
  if (!window.ISY) return;

  const activeBadges = new Map();
  const containerBadges = new WeakMap();
  let carouselMutationObserver = null;

  function isInstagram() {
    return window.ISY.state.currentAdapter?.name === 'Instagram';
  }

  function isYouTubeShorts() {
    return window.ISY.state.currentAdapter?.name === 'YouTube'
      && location.pathname.startsWith('/shorts/');
  }

  function needsScopedVisibility() {
    return isInstagram() || isYouTubeShorts();
  }

  function getCurrentYouTubeShortsKey() {
    const match = location.pathname.match(/^\/shorts\/([^/?#]+)/);
    return match ? `video:youtube:${match[1]}` : null;
  }

  function findContainer(target) {
    const adapter = window.ISY.state.currentAdapter;
    if (typeof adapter.containerFinder === 'function') {
      const found = adapter.containerFinder(target);
      if (found) return found;
    }
    return target.parentElement || document.body;
  }

  function getContainerStack(container) {
    const stack = containerBadges.get(container) || [];
    const liveStack = stack.filter(entry => entry.target?.isConnected && entry.badge?.isConnected);
    containerBadges.set(container, liveStack);
    return liveStack;
  }

  // target에서 container까지 올라가면서 aria-hidden="true"인 조상이 있으면 숨겨진 슬라이드
  function isSlideHidden(target, container) {
    let el = target.parentElement;
    while (el && el !== container) {
      if (el.getAttribute('aria-hidden') === 'true') return true;
      el = el.parentElement;
    }
    return false;
  }

  function getCarouselMutationObserver() {
    if (carouselMutationObserver) return carouselMutationObserver;
    carouselMutationObserver = new MutationObserver(() => {
      requestAnimationFrame(updateScopedVisibility);
    });
    return carouselMutationObserver;
  }

  // 작은 썸네일에서 배지가 콘텐츠를 가리지 않도록 컨테이너 너비에 맞춰 컴팩트 모드 적용.
  const COMPACT_WIDTH_THRESHOLD = 160;

  function attachBadge(target, badge) {
    if (!target || !target.isConnected) return false;
    installVisibilityListeners();

    const container = findContainer(target);
    if (!container) return false;

    // static 컨테이너만 relative로 변경하고 마킹 — detach 시 원복.
    const cs = window.getComputedStyle(container);
    if (cs.position === 'static' && !container.dataset.isyOriginalPosition) {
      container.dataset.isyOriginalPosition = 'static';
      container.style.position = 'relative';
    }

    const stack = getContainerStack(container);
    const index = needsScopedVisibility() ? 0 : stack.length;

    badge.style.position = 'absolute';
    badge.style.top = `${8 + index * 30}px`;
    badge.style.right = '8px';
    badge.style.left = 'auto';
    badge.style.zIndex = String(window.ISY.CONSTANTS.BADGE_Z_INDEX);

    const targetRect = target.getBoundingClientRect();
    if (targetRect.width > 0 && targetRect.width < COMPACT_WIDTH_THRESHOLD) {
      badge.classList.add('isy-badge-compact');
    }

    // Carousels/Shorts can keep off-screen DOM alive, so start hidden until visibility is recalculated.
    if (needsScopedVisibility()) badge.classList.add('isy-badge-hidden');

    container.appendChild(badge);
    stack.push({ target, badge });
    containerBadges.set(container, stack);
    activeBadges.set(target, { badge, container });

    if (isInstagram()) {
      // 캐러셀 슬라이드 전환 시 aria-hidden 변화를 즉시 감지
      getCarouselMutationObserver().observe(container, {
        subtree: true,
        attributeFilter: ['aria-hidden']
      });
      requestAnimationFrame(updateScopedVisibility);
    } else if (isYouTubeShorts()) {
      requestAnimationFrame(updateScopedVisibility);
    }

    // YouTube 등 가상화 환경에서 같은 <img>가 다른 콘텐츠로 재활용될 때, observer가 src
    // 변경을 감지해 이 스냅샷과 비교 후 배지를 떼낼 수 있도록 시점 src를 저장한다.
    if (target.tagName === 'IMG' || target.tagName === 'VIDEO') {
      target.dataset.isyBadgedSrc = target.currentSrc || target.src || '';
    }

    return true;
  }

  function restoreContainerPosition(container) {
    if (!container || !container.dataset || container.dataset.isyOriginalPosition !== 'static') return;
    const remaining = containerBadges.get(container);
    if (remaining && remaining.length > 0) return;
    container.style.position = '';
    delete container.dataset.isyOriginalPosition;
  }

  function getViewportOverlapRatio(target) {
    const rect = target.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return 0;

    const viewportW = window.innerWidth || document.documentElement.clientWidth;
    const viewportH = window.innerHeight || document.documentElement.clientHeight;
    const overlapLeft = Math.max(rect.left, 0);
    const overlapTop = Math.max(rect.top, 0);
    const overlapRight = Math.min(rect.right, viewportW);
    const overlapBottom = Math.min(rect.bottom, viewportH);
    const overlapWidth = Math.max(0, overlapRight - overlapLeft);
    const overlapHeight = Math.max(0, overlapBottom - overlapTop);

    return (overlapWidth * overlapHeight) / (rect.width * rect.height);
  }

  function updateYouTubeShortsVisibility() {
    const currentKey = getCurrentYouTubeShortsKey();
    let bestTarget = null;
    let bestOverlap = 0;

    for (const [target, record] of activeBadges) {
      if (!target?.isConnected) continue;
      const itemKey = record.badge?.dataset?.isyItemKey;
      if (currentKey && itemKey?.startsWith('video:youtube:') && itemKey !== currentKey) continue;

      const overlap = getViewportOverlapRatio(target);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestTarget = target;
      }
    }

    for (const [target, record] of activeBadges) {
      const itemKey = record.badge?.dataset?.isyItemKey;
      const isWrongShort = currentKey && itemKey?.startsWith('video:youtube:') && itemKey !== currentKey;
      record.badge.classList.toggle('isy-badge-hidden', isWrongShort || target !== bestTarget || bestOverlap < 0.4);
    }
  }

  function updateInstagramVisibility() {

    // 컨테이너(article)별로 배지를 묶어서 처리
    const containerMap = new Map();
    for (const [target, record] of activeBadges) {
      const c = record.container;
      if (!c) continue;
      if (!containerMap.has(c)) containerMap.set(c, []);
      containerMap.get(c).push({ target, badge: record.badge });
    }

    for (const [container, entries] of containerMap) {
      if (entries.length === 1) {
        entries[0].badge.classList.remove('isy-badge-hidden');
        continue;
      }

      // 캐러셀: aria-hidden 속성으로 현재 슬라이드 판단
      const hasAriaHidden = entries.some(e => isSlideHidden(e.target, container));
      if (hasAriaHidden) {
        for (const entry of entries) {
          entry.badge.classList.toggle('isy-badge-hidden', isSlideHidden(entry.target, container));
        }
        continue;
      }

      // aria-hidden 미사용 시 rect 기반 폴백: 컨테이너와 가장 많이 겹치는 이미지만 표시
      const containerRect = container.getBoundingClientRect();
      let bestEntry = null;
      let bestOverlap = -1;
      for (const entry of entries) {
        const rect = entry.target.getBoundingClientRect();
        if (rect.width <= 0) continue;
        const overlapLeft = Math.max(rect.left, containerRect.left);
        const overlapRight = Math.min(rect.right, containerRect.right);
        const overlap = Math.max(0, overlapRight - overlapLeft) / rect.width;
        if (overlap > bestOverlap) {
          bestOverlap = overlap;
          bestEntry = entry;
        }
      }
      for (const entry of entries) {
        entry.badge.classList.toggle('isy-badge-hidden', entry !== bestEntry || bestOverlap < 0.4);
      }
    }
  }

  function updateScopedVisibility() {
    if (isInstagram()) {
      updateInstagramVisibility();
      return;
    }
    if (isYouTubeShorts()) {
      updateYouTubeShortsVisibility();
    }
  }

  function removeAllBadges() {
    const touchedContainers = new Set();
    for (const [target, { badge, container }] of activeBadges) {
      if (badge.isConnected) badge.remove();
      if (target.dataset) delete target.dataset.isyBadged;
      if (container) touchedContainers.add(container);
    }
    activeBadges.clear();
    // 모든 배지를 제거했으니 stack도 비워야 restoreContainerPosition이 동작.
    for (const container of touchedContainers) {
      containerBadges.set(container, []);
      restoreContainerPosition(container);
    }
    carouselMutationObserver?.disconnect();
    carouselMutationObserver = null;
    document.querySelectorAll('.isy-detail-overlay').forEach(el => el.remove());
    window.removeEventListener('scroll', onScroll, true);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('transitionend', onTransitionEnd, true);
    window.removeEventListener('popstate', onNavigation);
    document.removeEventListener('yt-navigate-finish', onNavigation, true);
    document.removeEventListener('yt-page-data-updated', onNavigation, true);
  }

  // 단일 타깃의 배지만 제거. loading→결과 교체 시 사용.
  function detachBadge(target) {
    const record = activeBadges.get(target);
    if (!record) return false;
    const { badge, container } = record;
    if (badge.isConnected) badge.remove();
    activeBadges.delete(target);
    const stack = containerBadges.get(container);
    if (stack) {
      const idx = stack.findIndex(entry => entry.target === target);
      if (idx >= 0) stack.splice(idx, 1);
    }
    if (target.dataset) delete target.dataset.isyBadged;
    restoreContainerPosition(container);
    return true;
  }

  function getBadgeCount() {
    return activeBadges.size;
  }

  const onScroll = () => requestAnimationFrame(updateScopedVisibility);
  const onResize = () => requestAnimationFrame(updateScopedVisibility);
  const onNavigation = () => requestAnimationFrame(updateScopedVisibility);
  // rect 폴백용: CSS 전환 애니메이션이 끝난 뒤 정확한 위치로 재계산
  const onTransitionEnd = () => requestAnimationFrame(updateScopedVisibility);

  function installVisibilityListeners() {
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    window.addEventListener('transitionend', onTransitionEnd, true);
    window.addEventListener('popstate', onNavigation);
    document.addEventListener('yt-navigate-finish', onNavigation, true);
    document.addEventListener('yt-page-data-updated', onNavigation, true);
  }

  window.ISY.badges = {
    attach: attachBadge,
    detach: detachBadge,
    removeAll: removeAllBadges,
    getCount: getBadgeCount,
    refreshVisibility: updateScopedVisibility
  };
})();
