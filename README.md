# ISeeYou

ISeeYou는 텍스트, 이미지, 비디오, 멀티모달 입력을 분석해 AI 생성물 가능성과 진위 판별 보조 결과를 제공하는 설명 가능한 AI(XAI) 분석 서비스입니다. 단순 확률만 보여주지 않고, 어떤 근거가 판정에 영향을 주었는지 사용자가 이해할 수 있도록 문장, 프레임, 모달리티, 모델별 신호를 함께 표시합니다.

## 전체 구조

```text
ISeeYou/
├── UI/                  # Vite + React 프론트엔드
├── AI/
│   ├── Image/           # 이미지 AI 생성물 탐지 코드/모델 연결
│   ├── Text/            # 한국어 LOG-AID, 영어 DeBERTa 텍스트 탐지 연결
│   ├── Video/           # 7개 EfficientNet-B0 비디오 앙상블 연결
│   └── Multimodal/      # 통합 추론 서버와 멀티모달 fusion/XAI
├── MODEL_ARTIFACTS.md   # Git에 올리지 않는 모델 파일 관리 메모
├── CLEANUP_NOTES.md     # 정리 기준과 legacy 후보 기록
├── .gitignore
└── README.md
```

## 실행 방법

### 1. 백엔드 실행

```powershell
cd C:\Users\jjeong\Desktop\ISeeYou
python AI\Multimodal\inference\multimodal_inference_server.py
```

기본 API 주소는 `http://127.0.0.1:8001`입니다.

### 2. 프론트엔드 실행

```powershell
cd C:\Users\jjeong\Desktop\ISeeYou\UI
npm install
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

프론트엔드는 `http://127.0.0.1:5174`에서 확인할 수 있습니다. `UI/vite.config.ts`의 proxy가 `/multimodal-api/*` 요청을 백엔드 `8001` 포트로 전달합니다.

## 주요 기능

| 영역 | 현재 구현된 기능 | XAI 표시 방식 |
|---|---|---|
| Text | 한국어 LOG-AID, 영어 DeBERTa 기반 텍스트 판별. 너무 짧은 입력은 분석 대신 더 긴 문장을 요청 | 문장 span, 표현 신호, 반복/길이/근거 표현, 표현 관계 그래프 |
| Image | 전체 장면/얼굴/주파수 단서를 활용한 이미지 AI 생성 가능성 판별 | 의심 영역, RGB/FFT 단서, 얼굴 재판독 여부, 판단 근거 카드 |
| Video | 7개 EfficientNet-B0 계열 모델이 대표 프레임을 앙상블 분석 | 프레임별 generated 확률, 모델 의견 매트릭스, 의심 프레임 설명 |
| Multimodal | Visual, Audio/Sync, Text, Frequency, Scene graph 등 6개 계열 신호를 융합 | 모달리티별 점수, gate/down-weight, fusion logic, 타임라인 근거 |

## 모델 파라미터 상태

- 2026-05-18 기준 멀티모달 서비스 연결은 `final5000_gpu_anchor_fusion_v4b` 기준 런타임 번들입니다.
- `final8000` 학습은 일부 베이스 모델 결과만 확인되었고 최종 fusion 결과는 아직 확인되지 않아 서비스 연결 대상으로 사용하지 않았습니다.
- 실제 `.pt`, `.pkl`, `.safetensors` 등 모델 파일은 GitHub에 업로드하지 않습니다. 자세한 로컬 모델 경로와 제외 기준은 `MODEL_ARTIFACTS.md`를 참고하세요.

## UI/UX 방향

분석 페이지는 파일 또는 URL 입력을 먼저 보여주고, 긴 설명은 `?` 도움말 안에 접어 두도록 정리했습니다. 결과 화면은 핵심 판정, 주요 근거, XAI 해석 안내를 먼저 보여주며, 세부 pipeline/timeline/fusion 설명은 펼침 영역으로 확인할 수 있습니다.

## 결과 해석 주의

ISeeYou의 결과는 보조적 판단 도구입니다. AI 생성물 탐지 점수와 XAI 근거는 의심 신호를 이해하기 위한 참고 자료이며, 최종 사실 확인은 원본 출처, 메타데이터, 추가 검증과 함께 판단해야 합니다.

## 개발 및 유지보수 참고

- UI 코드: `UI/src`
- 통합 API 서버: `AI/Multimodal/inference/multimodal_inference_server.py`
- API 모델 경로 상수는 통합 서버 상단에 정의되어 있습니다.
- 모델 가중치, 데이터셋, 로그, 환경 변수 파일은 Git에 포함하지 않습니다.
- 민감정보 가능성이 있는 `.env` 파일은 로컬에서만 관리하고, 공유 시에는 `.env.example` 형태만 사용하세요.
