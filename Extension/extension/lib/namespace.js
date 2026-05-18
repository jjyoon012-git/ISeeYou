// ============================================
// ISY 네임스페이스 — 다른 모든 모듈의 진입점
// 반드시 가장 먼저 로드돼야 함
// ============================================

(function() {
  if (window.ISY) return;  // 중복 초기화 방지

  window.ISY = {
    adapters: {},
    getCurrentAdapter: null,

    extractMedia: null,
    extractText: null,
    extractImageUrl: null,
    extractVideoUrl: null,
    getUrlFromElement: null,

    observer: null,
    badges: null,
    ui: null,

    state: {
      autoMode: false,
      analyzedUrls: new Set(),
      currentAdapter: null,
      results: new Map()   // url/key → { isFake, fakeProb }
    },

    CONSTANTS: {
      DEBOUNCE_MS: 500,
      MIN_IMAGE_SIZE: 150,
      EXTENSION_CLASS_PREFIX: 'isy-',
      BADGE_Z_INDEX: 999998,
      OVERLAY_Z_INDEX: 999999
    },

    isExtensionElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      if (typeof node.className !== 'string') return false;
      return node.className.startsWith(this.CONSTANTS.EXTENSION_CLASS_PREFIX)
          || node.className.includes(' ' + this.CONSTANTS.EXTENSION_CLASS_PREFIX);
    },

    resetAnalyzed() {
      this.state.analyzedUrls.clear();
    }
  };

  console.log('[ISY] Namespace initialized');
})();