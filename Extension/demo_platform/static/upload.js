const form = document.getElementById('upload-form');
const fileInput = document.getElementById('video-file');
const fileName = document.getElementById('file-name');
const submitBtn = document.getElementById('submit-btn');
const statusEl = document.getElementById('status');
const resultCard = document.getElementById('result-card');

const { levelClass } = window.ISY_DEMO;

function setStatus(text, mode) {
  statusEl.textContent = text;
  statusEl.className = `yt-status ${mode || 'idle'}`;
}

function resultTitle(disclosure) {
  if (disclosure.level === 'high') return 'AI 생성 가능성 높음';
  if (disclosure.level === 'uncertain') return 'AI 생성 여부 확인 필요';
  return 'AI 생성 가능성 낮음';
}

function renderResult(payload) {
  const disclosure = payload.disclosure;
  const title = resultTitle(disclosure);
  const cls = levelClass(disclosure.level);
  resultCard.className = `yt-result-card ${cls}`;
  resultCard.innerHTML = `
    <div class="yt-result-header">
      <span class="isy-dot"></span>
      ISY 검증 결과 · ${title}
    </div>
    <div class="yt-result-body">
      <div class="yt-score">
        <div>
          <div class="yt-score-pct">${disclosure.percent}%</div>
          <div class="yt-score-label">AI 생성 가능성</div>
        </div>
      </div>
      <div>
        <div class="yt-gauge-header"><span>0%</span><span>100%</span></div>
        <div class="yt-gauge-track">
          <div class="yt-gauge-fill" style="width:${disclosure.percent}%"></div>
        </div>
      </div>
      <p class="yt-result-note">DEMO 시청자에게 영상 시작 시 ISY 검증 라벨이 노출됩니다.</p>
      <a class="yt-watch-link" href="/demo/browse" target="_blank" rel="noreferrer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
        시청자 입장에서 보기
      </a>
    </div>
  `;
}

async function parseError(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (fileName) fileName.textContent = file ? file.name : '';
  if (file) uploadSelectedFile();
});

async function uploadSelectedFile() {
  const file = fileInput.files?.[0];
  if (!file) return;

  submitBtn.disabled = true;
  resultCard.className = 'yt-result-card hidden';
  setStatus('영상 분석 및 공개 라벨 등록 중', 'busy');

  const data = new FormData();
  data.append('file', file, file.name);

  try {
    const response = await fetch('/api/platform/demo-upload', { method: 'POST', body: data });
    if (!response.ok) throw new Error(await parseError(response));
    const payload = await response.json();
    setStatus('시청자 공개 라벨 등록 완료', 'done');
    renderResult(payload);
  } catch (err) {
    setStatus(err.message || '업로드 실패', 'error');
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  uploadSelectedFile();
});
