# 모델 아티팩트 관리 메모

이 저장소는 GitHub 업로드를 위해 코드, 설정, 문서 중심으로 관리합니다. 실제 모델 가중치와 대용량 데이터셋은 Git에 포함하지 않습니다.

## 현재 로컬 연결 기준

- Multimodal runtime bundle: `AI/Multimodal/models/_service_runtime_bundle_v4b.pt`
- Multimodal source experiment: `D:\ISeeYou\experiments\final5000_gpu_anchor_fusion_v4b\full_model`
- Image scene model: `AI/Image/models/image_model_bundle/.../best_dualstream_final.pt`
- Image face model: `AI/Image/models/image_model_bundle/.../weights/best.pt`
- Text Korean LOG-AID: `AI/Text/models/text_model_bundle/LOG_AID_ko`
- Text English DeBERTa adapter: `AI/Text/models/text_model_bundle/DeBERTa_En`
- Video ensemble: `AI/Video/models/video` 내부 7개 EfficientNet-B0 계열 체크포인트

## 멀티모달 파라미터 선택 근거

2026-05-18 기준 `final8000` 학습은 OpenCLIP, FLAVA, BLIP-NLI, Frequency, Scene graph까지의 중간 결과만 확인되었고 AV-sync 및 최종 fusion 산출물은 확인되지 않았습니다. 따라서 서비스 연결은 완성된 후보 중 test accuracy와 ROC-AUC가 더 높은 `final5000_gpu_anchor_fusion_v4b` 기준 런타임 번들로 갱신했습니다.

비교 기준:

| 후보 | Test Accuracy | Test F1 | Test ROC-AUC | 판단 |
|---|---:|---:|---:|---|
| final5000_adaptive_multimodal_fusion_v2 | 0.8795 | 0.8738 | 0.9384 | 이전 source |
| final5000_gpu_anchor_fusion_v6 | 0.8766 | 0.8724 | 0.9441 | 보류 |
| final5000_gpu_anchor_fusion_v4b | 0.8917 | 0.8871 | 0.9506 | 현재 연결 |

## GitHub 업로드 제외

`.gitignore`에서 `*.pt`, `*.pth`, `*.pkl`, `*.safetensors`, `*.onnx`, `*.zip`, `.env*`, 로그, `node_modules`, `dist`를 제외합니다. 모델 파일이 필요한 실행 환경에서는 별도 로컬 경로 또는 외부 스토리지에서 복원해야 합니다.
