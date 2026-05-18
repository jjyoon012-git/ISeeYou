# AI/Image

## 목적
이미지의 시각 단서와 주파수/얼굴 단서를 바탕으로 AI 생성 이미지 가능성을 판별합니다.

## 폴더 구조
```text
Image/
├── models/image_model_bundle/
└── README.md
```

## 주요 파일
- `models/image_model_bundle/이미지모델_이원석/best_dualstream_final.pt`: RGB+FFT 이중 스트림 이미지 모델 체크포인트.
- `models/image_model_bundle/0501_데모버전/weights/best.pt`: 얼굴/보조 이미지 모델 후보 체크포인트. 현재 서버가 `best.pt`를 탐색할 수 있어 최종 구조에 포함했습니다.
- 관련 추론/전처리/XAI 코드는 현재 `AI/Multimodal/inference/multimodal_inference_server.py`에 통합되어 있습니다.

## 입력/출력
- 입력: PNG, JPG, WEBP 등 이미지 파일 또는 직접 이미지 URL.
- 출력: `Likely authentic` 또는 `Likely synthetic`, real/fake 확률, 의심 영역, 주파수 비교, 모델 근거 카드.

## 흐름
1. 이미지 로드 및 RGB 변환.
2. 전체 장면 또는 얼굴 초점 설정 확인.
3. RGB/FFT 및 얼굴 후보 모델 추론.
4. 최종 점수와 XAI 카드 생성.

## 현재 구현됨
- 이미지 업로드/URL 분석.
- 빠른/정밀 분석 모드 UI.
- RGB/FFT/얼굴 단서 결과 표시.

## 향후 개선 예정
- 이미지 전용 inference 파일 분리.
- 모델별 설정 파일 분리.
- 실제 Grad-CAM 계산 여부와 설명 라벨 정교화.
