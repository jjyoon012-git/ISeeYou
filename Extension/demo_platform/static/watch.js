const videoId = location.pathname.split('/').filter(Boolean).pop();
const video = document.getElementById('video');
const playerLabel = document.getElementById('player-label');
const labelTextEl = document.getElementById('label-text');
const videoTitle = document.getElementById('video-title');
const channelAvatar = document.getElementById('channel-avatar');
const relatedContainer = document.getElementById('related-videos');

const { levelClass, labelText: levelLabel } = window.ISY_DEMO;

function updatePlayerLabelVisibility() {
  playerLabel.hidden = video.currentTime > 10;
}

function renderDisclosure(disclosure) {
  const cls = levelClass(disclosure.level);
  video.src = `/api/platform/videos/${encodeURIComponent(disclosure.video_id)}`;

  const title = disclosure.filename || '업로드된 영상';
  videoTitle.textContent = title;
  if (channelAvatar) channelAvatar.textContent = title[0]?.toUpperCase() || 'D';

  labelTextEl.textContent = levelLabel(disclosure);
  playerLabel.className = `yt-player-label ${cls}`;
}

function createRelatedCard(d) {
  const cls = levelClass(d.level);
  const src = `/api/platform/videos/${encodeURIComponent(d.video_id)}`;
  const title = d.filename || '업로드된 영상';

  const el = document.createElement('a');
  el.className = 'yt-related-card';
  el.href = `/demo/watch/${d.video_id}`;
  el.innerHTML = `
    <div class="yt-related-thumb">
      <video muted playsinline preload="metadata" src="${src}"></video>
    </div>
    <div class="yt-related-info">
      <div class="yt-related-title">${title}</div>
      <div class="yt-related-channel">DEMO Creator</div>
      <div class="yt-related-isy ${cls}">
        <span class="isy-dot"></span>
        ${levelLabel(d)}
      </div>
    </div>
  `;

  const v = el.querySelector('video');
  v.addEventListener('loadedmetadata', () => { v.currentTime = 1; });
  el.addEventListener('mouseenter', () => { v.play().catch(() => {}); });
  el.addEventListener('mouseleave', () => { v.pause(); v.currentTime = 1; });
  return el;
}

async function loadRelated() {
  if (!relatedContainer) return;
  try {
    const res = await fetch('/api/platform/disclosures');
    if (!res.ok) return;
    const all = await res.json();
    const others = all.filter(d => d.video_id !== videoId);
    others.forEach(d => relatedContainer.appendChild(createRelatedCard(d)));
  } catch { /* 실패 무시 */ }
}

async function loadDisclosure() {
  try {
    const res = await fetch(`/api/platform/disclosures/${encodeURIComponent(videoId)}`);
    if (!res.ok) throw new Error('등록된 AI 콘텐츠 안내를 찾지 못했습니다.');
    renderDisclosure(await res.json());
  } catch (err) {
    labelTextEl.textContent = err.message;
  }
}

video.addEventListener('timeupdate', updatePlayerLabelVisibility);
video.addEventListener('seeked', updatePlayerLabelVisibility);
video.addEventListener('play', updatePlayerLabelVisibility);
video.addEventListener('ended', () => { playerLabel.hidden = true; });

loadDisclosure();
loadRelated();
