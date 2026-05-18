const { levelClass, labelText } = window.ISY_DEMO;

function createCard(disclosure) {
  const cls = levelClass(disclosure.level);
  const videoSrc = `/api/platform/videos/${encodeURIComponent(disclosure.video_id)}`;
  const title = disclosure.filename || '업로드된 영상';
  const initial = title[0]?.toUpperCase() || 'D';

  const card = document.createElement('div');
  card.className = 'yt-card';
  card.innerHTML = `
    <div class="yt-card-thumb">
      <video muted playsinline preload="metadata" src="${videoSrc}"></video>
      <div class="yt-card-isy ${cls}">
        <span class="isy-dot"></span>
        <span class="isy-badge-text">ISY 검증</span>
        <span class="isy-divider">·</span>
        <span>${labelText(disclosure)}</span>
      </div>
    </div>
    <div class="yt-card-meta">
      <div class="yt-channel-avatar">${initial}</div>
      <div class="yt-card-info">
        <div class="yt-card-title">${title}</div>
        <div class="yt-card-channel">DEMO Creator</div>
        <div class="yt-card-stats">조회수 0회</div>
      </div>
    </div>
  `;

  const video = card.querySelector('video');
  video.addEventListener('loadedmetadata', () => { video.currentTime = 1; });
  card.addEventListener('mouseenter', () => { video.play().catch(() => {}); });
  card.addEventListener('mouseleave', () => { video.pause(); video.currentTime = 1; });
  card.addEventListener('click', () => { location.href = `/demo/watch/${disclosure.video_id}`; });

  return card;
}

async function loadVideos() {
  const grid = document.getElementById('video-grid');
  try {
    const res = await fetch('/api/platform/disclosures');
    if (!res.ok) throw new Error();
    const videos = await res.json();
    grid.innerHTML = '';
    if (!videos.length) {
      grid.innerHTML = '<p class="yt-empty">업로드된 영상이 없습니다. 먼저 영상을 업로드해주세요.</p>';
      return;
    }
    videos.forEach(d => grid.appendChild(createCard(d)));
  } catch {
    grid.innerHTML = '<p class="yt-empty">영상 목록을 불러오지 못했습니다.</p>';
  }
}

loadVideos();
