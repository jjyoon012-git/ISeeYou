# AI/Video

## 목적
비디오에서 6개 대표 프레임을 균등 샘플링하고, 7개 EfficientNet-B0 프레임 모델을 앙상블해 AI 생성 영상 가능성을 판별합니다.

## 폴더 구조
```text
Video/
├── models/video/MODEL_HANDOFF.md
├── models/video/builder.py
├── models/video/masking.py
├── models/video/checkpoints_*_frame/best.pt
└── README.md
```

## 주요 파일
- `MODEL_HANDOFF.md`: 모델 통합 문서.
- `builder.py`: FrameClassifier 정의 참고.
- `masking.py`: text mask 구현 참고.
- 7개 `best.pt`: robustaug, EMA, holdout, seed, img320 앙상블 체크포인트.
- 실제 API 연결은 `AI/Multimodal/inference/multimodal_inference_server.py`의 `/analyze-video`, `/analyze-video-url`에서 처리합니다.

## 입력/출력
- 입력: MP4, MOV, WEBM 등 영상 파일 또는 URL.
- 출력: `p_real`, `p_generated`, 최종 verdict, 프레임별 점수, 모델별 점수, `videoXai` 매트릭스.

## 흐름
1. 전체 영상에서 6개 프레임 균등 샘플링.
2. 상단 8%, 하단 18% text mask 적용.
3. 모델별 224/320 resize와 ImageNet normalize.
4. 원본 + horizontal flip TTA softmax 평균.
5. 프레임별 7모델 결과 median 집계.
6. confidence_mean으로 최종 영상 확률 산출.

## 현재 구현됨
- 실제 7개 모델 앙상블 연결.
- Video 전용 XAI: 프레임별 확률, 모델 의견 매트릭스, text mask 비교.

## 향후 개선 예정
- 진짜 Grad-CAM 계산 추가 가능. 현재는 실제 gradient 기반 영역 XAI를 만들지 않으며, 가짜 heatmap을 표시하지 않습니다.
- 비디오 전용 독립 inference 스크립트 분리.
