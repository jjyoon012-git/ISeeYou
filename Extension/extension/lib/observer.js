// ============================================
// DOM 변경 감지기
// ============================================

(function() {
  if (!window.ISY) return;

  let mutationObserver = null;
  let onNewContentCallback = null;
  let debounceTimer = null;
  const urlChangeListeners = [];
  let urlHookInstalled = false;
  let lastUrl = location.href;
  let origPush = null;
  let origReplace = null;
  // SPA가 history API를 우회하는 경우(예: YouTube 일부 경로)에도 URL 변경을 잡기 위한 폴링
  let urlPollTimer = null;

  // stopAll에서 removeEventListener에 같은 참조를 전달하려면 모듈 스코프에 있어야 함
  function checkUrlChange() {
    if (location.href !== lastUrl) {
      const oldUrl = lastUrl;
      lastUrl = location.href;
      console.log(`[ISY] URL: ${oldUrl} → ${lastUrl}`);
      urlChangeListeners.forEach(cb => {
        try { cb(lastUrl); } catch (e) { console.error(e); }
      });
    }
  }

  // callback({ mediaItems, textItems }) 형태로 호출
  function startMutationObserver(callback) {
    if (mutationObserver) {
      console.log('[ISY] Observer already running');
      return;
    }
    onNewContentCallback = callback;

    mutationObserver = new MutationObserver(mutations => {
      let hasNewMedia = false;
      let hasNewText = false;

      for (const mutation of mutations) {
        if (mutation.type === 'attributes') {
          const target = mutation.target;
          if (target && target.nodeType === Node.ELEMENT_NODE && !window.ISY.isExtensionElement(target)) {
            if (target.tagName === 'IMG' || target.tagName === 'VIDEO' || target.tagName === 'SOURCE') {
              // 가상화로 재활용된 <img>의 src가 새 콘텐츠로 바뀌면, 이전 결과 배지를 떼서
              // 재분석된 결과가 화면에 새로 그려지게 한다. 같은 URL로 잠깐 돌아오는
              // 깜빡임은 isyBadgedSrc 비교로 무시.
              if (window.ISY.badges && target.dataset && target.dataset.isyBadged) {
                const prevSrc = target.dataset.isyBadgedSrc || '';
                const nextSrc = target.currentSrc || target.src || '';
                if (prevSrc && nextSrc && prevSrc !== nextSrc) {
                  window.ISY.badges.detach(target);
                }
              }
              hasNewMedia = true;
            }
          }
          continue;
        }

        for (const node of mutation.addedNodes) {
          if (node.nodeType !== Node.ELEMENT_NODE) continue;
          if (window.ISY.isExtensionElement(node)) continue;

          const tag = node.tagName;

          if (tag === 'IMG' || tag === 'VIDEO'
              || (node.querySelector && node.querySelector('img, video'))) {
            hasNewMedia = true;
          }

          // 텍스트 컨테이너 추가 감지 (Naver 기사 동적 로드 등)
          if (tag === 'P' || tag === 'ARTICLE' || tag === 'SECTION'
              || (tag === 'DIV' && node.querySelector && node.querySelector('p'))
              || tag === 'SPAN') {
            hasNewText = true;
          }
        }
        if (hasNewMedia && hasNewText) break;
      }

      if (hasNewMedia || hasNewText) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          const mediaItems = window.ISY.extractMedia();
          const textItems = window.ISY.extractText();
          if ((mediaItems.length > 0 || textItems.length > 0) && onNewContentCallback) {
            onNewContentCallback({ mediaItems, textItems });
          }
        }, window.ISY.CONSTANTS.DEBOUNCE_MS);
      }
    });

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'poster', 'data-src', 'data-srcset']
    });

    console.log('[ISY] MutationObserver started');
  }

  function startUrlChangeDetector(callback) {
    urlChangeListeners.push(callback);

    if (urlHookInstalled) return;
    urlHookInstalled = true;

    origPush = history.pushState.bind(history);
    origReplace = history.replaceState.bind(history);

    history.pushState = function(...args) {
      const result = origPush(...args);
      checkUrlChange();
      return result;
    };
    history.replaceState = function(...args) {
      const result = origReplace(...args);
      checkUrlChange();
      return result;
    };

    window.addEventListener('popstate', checkUrlChange);

    // 백업 폴링 — history API가 우회된 SPA navigation도 1초 이내 감지
    urlPollTimer = setInterval(checkUrlChange, 1000);
  }

  function stopAll() {
    mutationObserver?.disconnect();
    mutationObserver = null;
    clearTimeout(debounceTimer);
    onNewContentCallback = null;

    if (urlHookInstalled) {
      history.pushState = origPush;
      history.replaceState = origReplace;
      window.removeEventListener('popstate', checkUrlChange);
      origPush = null;
      origReplace = null;
      urlHookInstalled = false;
      urlChangeListeners.length = 0;
    }
    if (urlPollTimer) {
      clearInterval(urlPollTimer);
      urlPollTimer = null;
    }

    console.log('[ISY] Observer stopped');
  }

  window.ISY.observer = {
    startMutationObserver,
    startUrlChangeDetector,
    stopAll
  };
})();
