# ISY — I See You

웹 페이지의 이미지·영상·텍스트가 AI로 생성된 콘텐츠인지 브라우저에서 실시간으로 탐지하는 Chrome 확장 프로그램입니다.

확장 프로그램이 페이지의 미디어를 수집하면, 로컬 FastAPI 서버가 PyTorch 모델로 REAL/FAKE 확률을 계산하고, 결과 배지를 페이지 위에 바로 표시합니다.

```
브라우저 (content.js)
    └─► background.js (Service Worker)
            └─► localhost:8000 (FastAPI + PyTorch)
                    └─► REAL / FAKE 확률 반환
```

---

## 구현 상태

| 영역 | 상태 | 비고 |
|------|------|------|
| 이미지 분석 | ✅ 완성 | EfficientNet-B4 Late Fusion + FFT (versionv9) |
| 영상 분석 | ✅ 완성 | EfficientNet-B0 앙상블 n=7 Median TTA (video_inference.py) |
| 텍스트 분석 | 🔲 엔드포인트 준비 중 | `text_model/` 폴더 추가 시 활성화 |
| 확장 프로그램 UI | ✅ 완성 | 팝업 모드 카드, 배지, 페이지 스캔, 우클릭 분석 |
| Mock 서버 | ✅ 완성 | 모델 없이 UI 동작 확인 가능 |
| 데모 플랫폼 | ✅ 완성 | YouTube 다크 테마 기반 파트너 플랫폼 시연 화면 |

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/ryujihos0105/isy-extention.git
cd isy-extention
```

### 2. Python 환경 설정

Python 3.10 이상이 필요합니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. 패키지 설치

**PyTorch는 환경에 맞는 버전을 먼저 설치하세요.**

```bash
# CUDA 11.8 (GPU)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (GPU)
pip install torch==2.7.1+cu121 torchvision==0.22.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.7.1 torchvision==0.22.1
```

그다음 나머지 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. 모델 가중치 배치

`*.pt` 파일은 Git에 포함되지 않습니다. **[Hugging Face Hub](https://huggingface.co/ryujiho/isy-weights)** 에서 다운로드한 뒤 아래 경로에 넣어주세요.

**이미지 모델**

HF Hub에서 `image/best.pt` → 로컬 경로:
```
versionv9/weights/best.pt
```

**영상 모델** (앙상블 7개, 폴더는 이미 생성되어 있음)

HF Hub에서 `video/checkpoints_.../best.pt` → 로컬 경로:
```
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_frame/best.pt
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout_frame/best.pt
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_img320_frame/best.pt
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame/best.pt
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed7_frame/best.pt
video/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_df_holdout_frame/best.pt
```

### 5. 서버 실행

```bash
# 실제 모델로 실행
python server.py

# 모델 없이 UI만 테스트
python mock_server.py
```

서버가 실행되면 `http://localhost:8000` 에서 응답합니다.

### 6. 확장 프로그램 설치

1. Chrome에서 `chrome://extensions` 를 엽니다.
2. 오른쪽 위 **개발자 모드**를 켭니다.
3. **압축해제된 확장 프로그램을 로드합니다**를 클릭합니다.
4. 이 프로젝트의 `extension/` 폴더를 선택합니다.
5. 서버가 실행된 상태에서 아무 페이지나 열고, 팝업에서 **현재 페이지 분석**을 누릅니다.

---

## 프로젝트 구조

```
isy-extention/
├── extension/                   # Chrome 확장 프로그램
│   ├── manifest.json            # MV3 설정 (권한, 스크립트 목록)
│   ├── background.js            # Service Worker — API 호출, LRU 캐시(200개, TTL 30분)
│   ├── content.js               # 페이지 스캔, 배지 표시
│   ├── content.css
│   ├── lib/
│   │   ├── namespace.js         # ISY 전역 네임스페이스
│   │   ├── site-adapters.js     # 사이트별 DOM 어댑터 (YouTube, Instagram, Naver)
│   │   ├── media-extractor.js   # 이미지·영상·텍스트 후보 수집
│   │   ├── badge-manager.js     # 배지 DOM 부착/제거
│   │   ├── ui-manager.js        # 배지 렌더링, 상세 오버레이
│   │   └── observer.js          # DOM 변화·URL 변화 감지
│   └── popup/
│       ├── popup.html
│       ├── popup.css
│       └── popup.js
├── versionv9/                   # 이미지 판별 모델 (완성)
│   ├── model.py                 # EfficientNet-B4 Late Fusion 정의
│   ├── preprocess.py            # 얼굴 크롭 + FFT 방식 B 전처리
│   ├── config.py                # 경로·하이퍼파라미터 설정
│   └── weights/
│       └── best.pt              # 학습된 가중치 (Git 미포함)
├── video_inference.py           # 영상 판별 모델 추론 (완성)
│   └── EfficientNet-B0 앙상블 n=7, Median TTA
├── video/                       # 영상 모델 관련 파일
│   ├── builder.py               # 모델 빌더
│   ├── masking.py               # 마스킹 유틸리티
│   ├── MODEL_HANDOFF.md         # 영상 모델 인수인계 문서
│   └── checkpoints_*/best.pt   # 앙상블 체크포인트 (Git 미포함)
├── demo_platform/               # 시연용 데모 플랫폼 (YouTube 다크 테마)
│   ├── disclosures.json         # 업로드 영상별 AI 공개 라벨 저장
│   ├── uploads/                 # 업로드된 영상 파일
│   └── static/
│       ├── browse.html / .js    # 홈 (영상 그리드, 호버 프리뷰 + AI 라벨)
│       ├── upload.html / .js    # 크리에이터 업로드 화면
│       ├── watch.html / .js     # 시청자 화면 (영상 위 ISY 검증 라벨, 처음 10초)
│       ├── utils.js             # 공통 유틸리티
│       └── demo.css             # YouTube 다크 테마 공통 스타일
├── server.py                    # FastAPI 추론 서버
├── mock_server.py               # 테스트용 Mock 서버
└── requirements.txt             # Python 의존성
```

---

## API

### 이미지 분석 — `POST /api/analyze/image`

```bash
curl -X POST http://localhost:8000/api/analyze/image \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

```json
{
  "url": "https://example.com/image.jpg",
  "media_type": "image",
  "fake_probability": 0.1234,
  "real_probability": 0.8766,
  "label": "REAL",
  "consistency_score": 88,
  "crop_status": "성공",
  "model": "versionv9-fftB"
}
```

### 영상 분석 — `POST /api/analyze/video`

```bash
curl -X POST http://localhost:8000/api/analyze/video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "platform_meta": {"platform": "youtube", "videoId": "VIDEO_ID"}}'
```

```json
{
  "url": "...",
  "media_type": "video",
  "fake_probability": 0.87,
  "real_probability": 0.13,
  "label": "FAKE",
  "model": "efficientnet-b0-ensemble-n7-median-tta"
}
```

### 텍스트 분석 — `POST /api/analyze/text`

현재 엔드포인트만 준비된 상태입니다. `text_model/` 폴더를 추가하면 자동으로 활성화됩니다.

### 데모 플랫폼

플랫폼 협약을 가정한 시연 화면입니다. 업로드 페이지에서 영상을 분석하면 서버가 결과를 `Platform Disclosure API` 형태로 저장하고, 시청자 화면에서 ISY 검증 라벨로 표시합니다.

```bash
python server.py
```

| 화면 | URL |
|------|-----|
| 홈 (영상 그리드) | http://localhost:8000/demo/browse |
| 크리에이터 업로드 | http://localhost:8000/demo/upload |
| 시청자 화면 | http://localhost:8000/demo/watch/{video_id} |

---

## 새 모델 추가 방법

`versionv9/`와 동일한 구조로 폴더를 만들거나 `video_inference.py`처럼 별도 모듈로 작성합니다.

```
text_model/
├── model.py
├── preprocess.py
├── config.py
└── weights/best.pt
```

그다음 `server.py`의 `_load_text_model()` 함수를 구현하고 서버를 재시작합니다.

---

## 개발자 디버그 옵션

사용자용 결과 라벨에는 판정에 필요한 정보만 표시합니다. 내부 분석 정보(콘텐츠 유형, 얼굴 크롭 결과, 모델명 등)는 기본적으로 숨겨져 있습니다.

개발 중 상세 결과 오버레이에서 내부 정보를 보고 싶으면 분석 대상 페이지의 DevTools 콘솔에서 아래 값을 켭니다.

```js
localStorage.setItem('isyDebugDetails', 'true')
```

다시 사용자 기본 표시로 되돌리려면:

```js
localStorage.removeItem('isyDebugDetails')
```

---

## 주의사항

- 모델 가중치(`*.pt`)는 `.gitignore`로 Git에 포함되지 않습니다. [Hugging Face Hub](https://huggingface.co/ryujiho/isy-weights) 에서 다운로드하세요.
- 서버는 `localhost:8000`에서만 요청을 수신합니다. 외부에서는 접근할 수 없습니다.
- 확장 프로그램은 모든 URL에서 실행되므로 온라인 뱅킹 등 민감한 사이트에서 사용 시 주의하세요.
