// ============================================
// ISY Popup Script
// ============================================

const analyzeAllBtn = document.getElementById('analyze-all-btn');
const analyzeSelectedBtn = document.getElementById('analyze-selected-btn');
const analyzeCurrentVideoBtn = document.getElementById('analyze-current-video-btn');
const analyzeShortsVideoBtn = document.getElementById('analyze-shorts-video-btn');
const startDefault = document.getElementById('start-default');
const startShorts = document.getElementById('start-shorts');

const statusEl = document.getElementById('status');
const adapterNameEl = document.getElementById('adapter-name');

const startCard = document.getElementById('start-card');
const progressCard = document.getElementById('progress-card');
const resultsCard = document.getElementById('results-card');

const progressLabel = document.getElementById('progress-label');
const progressCount = document.getElementById('progress-count');
const progressBar = progressCard.querySelector('.progress-bar');
const progressFill = document.getElementById('progress-fill');

const highCountEl = document.getElementById('high-count');
const lowCountEl = document.getElementById('low-count');
const failedCountEl = document.getElementById('failed-count');
const focusHighBtn = document.getElementById('focus-high-btn');
const focusLowBtn = document.getElementById('focus-low-btn');
const focusFailedBtn = document.getElementById('focus-failed-btn');
const retryFailedBtn = document.getElementById('retry-failed-btn');
const clearResultsBtn = document.getElementById('clear-results-btn');
const stopAnalysisBtn = document.getElementById('stop-analysis-btn');

// 분석 시작 명시 플래그 — content 측 state 갱신이 늦어도 즉시 running 카드로 전환
let intentRunning = false;
// 이번 분석 작업 시작 시점의 누적 완료 수 — 페이지에 이미 있는 결과를 진행률에서 제외
let baselineCompleted = 0;
// 직전 모드 — 'running' → 'done' 전환 시 status를 자동으로 '분석 완료'로 갱신
let lastMode = null;
// 현재 페이지가 YouTube Shorts인지 — Shorts에선 누적 결과 카드를 숨긴다
// (해당 콘텐츠로 이동이 의미 없고, 다른 페이지에서 본 썸네일 결과가 섞여 보여 혼란을 줌)
let isShorts = false;

function setMode(mode) {
  // 'done' 모드에서는 결과 요약 + 시작 카드를 함께 표시 — 사용자가 결과를 본 채로
  // 다른 분석 모드(전체/선택/현재 영상)로 바로 전환할 수 있게 한다.
  startCard.classList.toggle('hidden', mode === 'running');
  progressCard.classList.toggle('hidden', mode !== 'running');
  // resultsCard는 applyState가 누적 결과 유무에 따라 직접 토글한다 (running 중에도 노출).
  // idle일 때만 강제로 숨김.
  if (mode === 'idle') resultsCard.classList.add('hidden');

  // done 모드로 처음 진입 시 status를 명시적으로 '분석 완료'로 갱신.
  // popup을 닫았다 다시 연 경우(lastMode가 null)에도 결과 카드와 함께 메시지가 뜸.
  if (lastMode !== 'done' && mode === 'done') {
    setStatus('분석 완료', 'success');
  }
  lastMode = mode;
}

function updateResultsCard(high, low, failed) {
  highCountEl.textContent = high;
  lowCountEl.textContent = low;
  failedCountEl.textContent = failed;
  setFocusButtonState(focusHighBtn, high);
  setFocusButtonState(focusLowBtn, low);
  setFocusButtonState(focusFailedBtn, failed);
  retryFailedBtn.classList.toggle('hidden', failed <= 0);
  retryFailedBtn.disabled = failed <= 0;
}

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.className = 'status' + (type ? ' status-' + type : '');
}

function setFocusButtonState(button, count) {
  button.disabled = count <= 0;
}

function updateProgress(completed, total) {
  const safeTotal = Math.max(total, 1);
  const pct = Math.min(100, Math.round((completed / safeTotal) * 100));
  progressCount.textContent = `${completed} / ${total}`;
  progressFill.style.width = `${pct}%`;
  progressBar.setAttribute('aria-valuenow', String(pct));
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function pingContentScript(tabId, retries = 1) {
  for (let i = 0; i <= retries; i++) {
    try {
      const response = await chrome.tabs.sendMessage(tabId, { type: 'PING' });
      if (response && response.ok) return response;
    } catch (err) {}
    if (i < retries) await new Promise(resolve => setTimeout(resolve, 200));
  }
  return null;
}

function isRestrictedUrl(url) {
  return url.startsWith('chrome://')
    || url.startsWith('chrome-extension://')
    || url.startsWith('about:');
}

// 분석 시작 직전의 누적 결과 수를 baseline으로 캡처.
// 이후 진행률이 "이번 작업"만 반영하게 됨.
async function captureBaseline() {
  const tab = await getActiveTab();
  if (!tab) return;
  try {
    const state = await chrome.tabs.sendMessage(tab.id, { type: 'GET_STATE' });
    if (state) {
      baselineCompleted = (state.highCount || 0) + (state.lowCount || 0) + (state.failedCount || 0);
    }
  } catch {}
}

async function safeSend(message) {
  const tab = await getActiveTab();
  if (!tab || !tab.url) {
    setStatus('탭 정보를 가져올 수 없습니다.', 'error');
    return null;
  }
  if (isRestrictedUrl(tab.url)) {
    setStatus('이 페이지에서는 사용할 수 없습니다.', 'error');
    return null;
  }
  const ping = await pingContentScript(tab.id);
  if (!ping) {
    setStatus('페이지를 새로고침한 뒤 다시 시도하세요.', 'error');
    return null;
  }
  try {
    return await chrome.tabs.sendMessage(tab.id, message);
  } catch (err) {
    setStatus(err.message, 'error');
    return null;
  }
}

function applyState(state) {
  const high = state.highCount || 0;
  const low = state.lowCount || 0;
  const failed = state.failedCount || 0;
  const pending = state.pendingCount || 0;
  const completed = high + low + failed;

  // 진행률은 "이번 작업 기준"으로 계산 — 페이지 전체 누적 결과가 아닌
  // 분석 시작 시점 이후의 증가분만 반영해야 1/1 같은 자연스러운 표시가 된다.
  const runCompleted = Math.max(0, completed - baselineCompleted);
  const runTotal = runCompleted + pending;

  const running = intentRunning || pending > 0 || (!!state.analysisStarted && completed === 0);

  // 결과 카드는 누적 결과가 있으면 running/done 모두에서 표시 — 분석 중에도
  // 이전까지의 결과를 계속 보면서 카운트 변화를 추적할 수 있게 한다.
  // 단 Shorts에서는 콘텐츠 이동이 무의미하고 누적 카운트가 혼란을 주므로 숨김.
  if (completed > 0 && !isShorts) {
    resultsCard.classList.remove('hidden');
    updateResultsCard(high, low, failed);
  } else {
    resultsCard.classList.add('hidden');
  }

  if (running) {
    setMode('running');
    progressLabel.textContent = pending > 0 ? '분석 중' : '시작 중';
    updateProgress(runCompleted, runTotal > 0 ? runTotal : 1);
    return;
  }

  if (completed > 0) {
    setMode('done');
    return;
  }

  setMode('idle');
}

async function syncState() {
  const tab = await getActiveTab();
  if (!tab || !tab.url) {
    adapterNameEl.textContent = '알 수 없음';
    return;
  }
  if (isRestrictedUrl(tab.url)) {
    adapterNameEl.textContent = '사용 불가';
    [analyzeAllBtn, analyzeSelectedBtn, analyzeCurrentVideoBtn].forEach(btn => {
      if (btn) btn.disabled = true;
    });
    return;
  }

  const ping = await pingContentScript(tab.id);
  if (!ping) {
    adapterNameEl.textContent = '새로고침 필요';
    return;
  }

  adapterNameEl.textContent = ping.adapter || '알 수 없음';

  // Shorts에서는 전체/선택 분석이 의미 없으므로 '현재 영상 분석' 단일 액션으로 단순화.
  isShorts = /youtube\.com\/shorts\//.test(tab.url || '');
  const isWatch  = /youtube\.com\/watch/.test(tab.url || '');
  startDefault.classList.toggle('hidden', isShorts);
  startShorts.classList.toggle('hidden', !isShorts);
  // 일반 watch 페이지에서만 ghost '현재 영상만' 보조 액션 노출 (Shorts는 별도 카드)
  analyzeCurrentVideoBtn.classList.toggle('hidden', !isWatch);

  try {
    const state = await chrome.tabs.sendMessage(tab.id, { type: 'GET_STATE' });
    if (state) applyState(state);
  } catch (err) {
    console.error(err);
  }
}

async function focusResult(level) {
  const labels = {
    high: '가능성 높은',
    low: '가능성 낮은',
    failed: '분석 실패'
  };
  const res = await safeSend({ type: 'FOCUS_RESULT', level });
  if (!res) return;
  if (res.ok) {
    setStatus(`${labels[level]} 콘텐츠 위치로 이동했습니다.`, 'success');
  } else {
    setStatus(res.error || '이동할 콘텐츠를 찾지 못했습니다.', 'error');
  }
}

focusHighBtn.addEventListener('click', () => focusResult('high'));
focusLowBtn.addEventListener('click', () => focusResult('low'));
focusFailedBtn.addEventListener('click', () => focusResult('failed'));

retryFailedBtn.addEventListener('click', async () => {
  await captureBaseline();
  setStatus('실패 항목을 재분석합니다', 'loading');
  const res = await safeSend({ type: 'RETRY_FAILED' });
  if (!res) return;
  if (!res.ok || res.retried === 0) {
    setStatus('재시도할 항목을 찾지 못했습니다.', 'error');
    return;
  }
  intentRunning = true;
  progressCard.classList.remove('compact');  // 재시도는 X/Y 표시
  setMode('running');
  setStatus(`${res.retried}개 항목 재분석 중`, 'loading');
  startPolling();
});

stopAnalysisBtn.addEventListener('click', async () => {
  const res = await safeSend({ type: 'STOP_ANALYSIS' });
  if (res && res.ok) {
    stopPolling();
    intentRunning = false;
    setStatus('자동 분석을 중단했습니다.', 'success');
    // 부분 결과가 있으면 done, 없으면 idle로
    syncState();
  }
});

clearResultsBtn.addEventListener('click', async () => {
  const res = await safeSend({ type: 'CLEAR_RESULTS' });
  if (!res || !res.ok) {
    setStatus('라벨을 숨기지 못했습니다.', 'error');
    return;
  }
  stopPolling();
  intentRunning = false;
  baselineCompleted = 0;
  setMode('idle');
  setStatus('화면의 분석 라벨을 숨겼습니다.', 'success');
});

let pollTimer = null;
let lastPollState = null;

function startPolling() {
  stopPolling();
  let lastSignature = '';
  let stableCount = 0;
  pollTimer = setInterval(async () => {
    const tab = await getActiveTab();
    if (!tab) return stopPolling();
    try {
      const state = await chrome.tabs.sendMessage(tab.id, { type: 'GET_STATE' });
      if (!state) return;
      lastPollState = state;

      // 한 번이라도 pending 또는 완료 결과를 본 시점부터는 intent 플래그 해제
      const hasReal = (state.pendingCount || 0) > 0
        || ((state.highCount || 0) + (state.lowCount || 0) + (state.failedCount || 0)) > 0;
      if (hasReal) intentRunning = false;

      applyState(state);

      const signature = [
        state.highCount || 0,
        state.lowCount || 0,
        state.failedCount || 0,
        state.pendingCount || 0
      ].join(':');

      if (signature === lastSignature) {
        // pending이 남아있는데 변화가 없다 = 백그라운드 분석이 아직 진행 중.
        // 이 상태에서 stable로 polling을 멈추면 결과가 와도 popup이 모름.
        if ((state.pendingCount || 0) > 0) {
          stableCount = 0;
        } else {
          stableCount += 1;
          if (stableCount >= 5) stopPolling({ reason: 'stable' });
        }
      } else {
        stableCount = 0;
        lastSignature = signature;
      }
    } catch (err) {
      stopPolling();
    }
  }, 700);
}

function stopPolling(opts) {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (opts && opts.reason === 'stable') {
    const pending = (lastPollState && lastPollState.pendingCount) || 0;
    if (pending === 0) {
      setStatus('분석 완료', 'success');
      intentRunning = false;
      if (lastPollState) applyState(lastPollState);
    }
  }
}

async function triggerAnalyzeAll() {
  await captureBaseline();
  setStatus('페이지의 콘텐츠를 분석하는 중', 'loading');
  intentRunning = true;
  progressCard.classList.remove('compact');  // 전체 분석은 X/Y · 진행률 막대 표시
  setMode('running');
  updateProgress(0, 1);
  const res = await safeSend({ type: 'ANALYZE_ALL' });
  if (!res) {
    intentRunning = false;
    syncState();
    return;
  }
  if (res.count === 0) {
    setStatus('분석할 콘텐츠를 찾지 못했습니다.', 'error');
    intentRunning = false;
    setMode('idle');
    return;
  }
  setStatus(`${res.count}개 항목 분석을 시작했습니다.`, 'loading');
  startPolling();
}

analyzeAllBtn.addEventListener('click', triggerAnalyzeAll);

analyzeSelectedBtn.addEventListener('click', async () => {
  setStatus('분석할 콘텐츠를 페이지에서 선택하세요.', 'loading');
  const res = await safeSend({ type: 'START_PICK_ANALYSIS' });
  if (!res) return;
  if (!res.ok) {
    setStatus(res.error || '선택 모드를 시작하지 못했습니다.', 'error');
    return;
  }
  setStatus('파란 박스를 클릭해 항목을 선택하세요. Esc로 종료합니다.', 'success');
});

async function triggerCurrentVideoAnalysis() {
  await captureBaseline();
  setStatus('현재 영상을 분석하는 중', 'loading');
  intentRunning = true;
  progressCard.classList.add('compact');  // 영상 1개 분석은 진행률·X/Y 숨김
  setMode('running');
  const res = await safeSend({ type: 'ANALYZE_CURRENT_VIDEO' });
  if (!res) {
    intentRunning = false;
    syncState();
    return;
  }
  if (!res.ok) {
    setStatus(res.error || '분석할 현재 영상을 찾지 못했습니다.', 'error');
    intentRunning = false;
    setMode('idle');
    return;
  }
  setStatus('현재 영상 분석을 시작했습니다.', 'loading');
  startPolling();
}

analyzeCurrentVideoBtn.addEventListener('click', triggerCurrentVideoAnalysis);
analyzeShortsVideoBtn.addEventListener('click', triggerCurrentVideoAnalysis);

window.addEventListener('unload', stopPolling);

syncState();
