# AI/Text

## 목적
텍스트가 AI로 생성됐을 가능성을 한국어/영어 모델로 판별하고, 문장과 표현 단위의 설명 가능한 신호를 제공합니다.

## 폴더 구조
```text
Text/
├── models/text_model_bundle/LOG_AID_ko/
├── models/text_model_bundle/DeBERTa_En/
└── README.md
```

## 주요 파일
- `LOG_AID_ko/text_detector_ko.py`, `logistic_regression.pkl`, `standard_scaler.pkl`: 한국어 LOG-AID 기반 탐지.
- `DeBERTa_En/text_detector_en.py`, `adapter_model.safetensors`, tokenizer 파일: 영어 DeBERTa 어댑터 기반 탐지.
- 실제 API 연결은 `AI/Multimodal/inference/multimodal_inference_server.py`의 `/analyze-text`에서 처리합니다.

## 입력/출력
- 입력: 긴 텍스트 또는 TXT 파일. 너무 짧은 단어/문장은 판별하지 않습니다.
- 출력: real/fake 확률, 언어/모델 상태, 문장 span, 표현 신호, 사용자 해석 문구.

## 흐름
1. 텍스트 길이와 문장 수 확인.
2. 언어 감지 후 한국어/영어 모델 선택.
3. 모델 확률 계산.
4. 반복, 긴 표현, 근거 표현 등 설명용 XAI 신호 생성.

## 현재 구현됨
- 한국어 LOG-AID 연결.
- 영어 DeBERTa 어댑터 연결.
- 짧은 텍스트 입력 차단.
- Text 전용 XAI 화면.

## 향후 개선 예정
- Text 전용 독립 CLI.
- 더 정교한 문장별 attribution.
- 외부 출처 대조 기능의 실제 검색/검증 파이프라인 확장.
