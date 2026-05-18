# AI/Multimodal

## 목적
UI와 연결되는 통합 추론 API 서버를 제공합니다. Image, Text, Video, Multimodal 분석 요청을 한 서버에서 받아 각 모델 번들을 호출합니다.

## 폴더 구조
```text
Multimodal/
├── inference/multimodal_inference_server.py
├── inference/start_backend.ps1
├── models/_service_runtime_bundle_v3.pt
└── README.md
```

## 주요 파일
- `inference/multimodal_inference_server.py`: HTTP API 서버. `/health`, `/analyze-text`, `/analyze-image`, `/analyze-video`, `/analyze`, `/explain`, `/explain-text` 등을 제공합니다.
- `models/_service_runtime_bundle_v3.pt`: 최신 멀티모달 런타임 번들.
- `inference/start_backend.ps1`: 새 구조 기준 백엔드 실행 스크립트.

## 입력/출력
- 입력: 텍스트 JSON, 이미지/비디오 multipart 파일, 이미지/비디오 URL JSON.
- 출력: verdict, real/fake 퍼센트, metrics, reasoning, processingScope, xai, modalityJudgments, fusionSteps 등.

## 흐름
1. 요청 타입에 따라 Text/Image/Video/Multimodal 분기.
2. 해당 모델 번들 로드.
3. 전처리와 추론 수행.
4. UI가 이해할 수 있는 공통 분석 payload로 후처리.
5. 필요한 경우 LLM 기반 설명 문구를 추가 생성.

## 현재 구현됨
- Text/Image/Video/Multimodal 분석 API.
- 멀티모달 gate/down-weight 및 fusion 설명.
- Video 실제 7-model ensemble API.
- Image/Text 모델 번들 경로를 새 `AI/*/models` 구조로 참조.

## 향후 개선 예정
- 모달리티별 독립 서버/모듈 분리.
- 배포용 FastAPI 또는 서비스 런처 구성.
- Python dependency lock 파일 정리.
