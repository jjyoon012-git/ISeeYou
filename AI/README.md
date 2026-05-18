# ISeeYou AI

## 역할
AI 폴더는 모델 번들, 통합 추론 API 서버, 런타임 번들을 보관합니다. 현재 실제 API 서버는 `AI/Multimodal/inference/multimodal_inference_server.py` 하나에 통합되어 있으며, Image/Text/Video 모델 파일은 각 모달리티 폴더에서 참조합니다.

## 폴더 역할
- `Image`: 이미지 AI 생성물 탐지 모델 번들.
- `Text`: 한국어 LOG-AID 및 영어 DeBERTa 텍스트 탐지 모델 번들.
- `Video`: 7개 EfficientNet-B0 비디오 프레임 앙상블 모델 번들.
- `Multimodal`: UI와 연결되는 통합 HTTP API 서버와 멀티모달 런타임 번들.

## 공통 추론 흐름
1. UI가 `/multimodal-api/*`로 요청합니다.
2. Vite proxy가 `127.0.0.1:8001` 서버로 전달합니다.
3. 서버가 입력 파일/텍스트/URL을 처리하고 해당 모델 번들을 로드합니다.
4. 결과는 `realPercent`, `fakePercent`, `verdictLabel`, `metrics`, `reasoning`, `xai` 구조로 UI에 반환됩니다.

## 모델 입력/출력 형식
- Text: 문자열 또는 TXT 파일 입력, real/fake 확률과 문장/표현 XAI 반환.
- Image: 이미지 파일/URL 입력, real/fake 확률과 의심 영역/주파수 단서 반환.
- Video: 영상 파일/URL 입력, real/fake 확률과 프레임별/모델별 점수 반환.
- Multimodal: 영상과 선택 텍스트 입력, 모달별 점수와 종합 판단 반환.

## XAI 결과 형식
UI는 `xai.timeline`, `xai.modalityBars`, `xai.regions`, `videoXai`, `textLlmSections`, `llmSections` 등을 사용합니다. 실제 모델이 제공하지 않는 영역 근거는 과장하지 않고 설명용 보조 신호로 표시합니다.

## UI 연동
백엔드는 `AI/Multimodal/inference/multimodal_inference_server.py`를 실행해 `127.0.0.1:8001`에서 대기합니다. UI는 `vite.config.mjs`의 proxy로 이 서버와 통신합니다.
