# 이 파일은 이미지 전처리 (얼굴 크롭, FFT 방식 B 변환, 텐서 변환)를 담당합니다 (v9)
# v7/v8 (방식 A: log-magnitude × 주파수² 가중치) 대비
# v9 (방식 B: FFT → 시그모이드 고주파 마스크 → 역FFT → 공간 도메인 이미지 복원)

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from typing import cast

import insightface
from insightface.app import FaceAnalysis

from config import (
    IMG_SIZE, FACE_MARGIN, IMAGENET_MEAN, IMAGENET_STD,
    FFT_CUTOFF, FFT_STEEPNESS,
)


# ── InsightFace 얼굴 감지기 초기화 ────────────────────────
# 전역으로 한 번만 로드 (매 호출마다 다시 로드하면 느림)
_face_app = FaceAnalysis(
    name="buffalo_sc",
    allowed_modules=["detection"],                              # 감지(detection)만 사용
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] # GPU 우선, 없으면 CPU
)
_face_app.prepare(
    ctx_id=0 if torch.cuda.is_available() else -1,
    det_size=(640, 640),
)


# 얼굴 크롭 상태를 나타내는 상수
CROP_SUCCESS        = "성공"          # 1차 탐지 후 정상 크롭
CROP_RETRY_SUCCESS  = "재탐지 성공"   # 패딩 추가 후 2차 탐지 성공
CROP_FAILED         = "탐지 실패"     # 얼굴 미검출 → 원본 사용
CROP_COORD_ERROR    = "좌표 오류"     # 크롭 좌표 계산 오류 → 원본 사용


# PIL 이미지에서 얼굴 영역을 크롭하는 함수
# 1차 시도 실패 시 패딩을 추가해서 재탐지 (소형/경계 얼굴 대응)
# 반환: (크롭된 이미지, 크롭 상태 문자열)
def crop_face(img_pil: Image.Image, margin: float = FACE_MARGIN) -> tuple[Image.Image, str]:
    img_np  = np.array(img_pil.convert("RGB"))
    img_bgr = img_np[:, :, ::-1]  # PIL(RGB) → OpenCV(BGR) 변환
    h, w    = img_bgr.shape[:2]

    faces = _face_app.get(img_bgr)  # 1차 얼굴 탐지

    # 1차 탐지 실패 시 패딩 추가 후 재탐지
    if not faces:
        padding_ratio = 0.3
        pad_h = int(h * padding_ratio)
        pad_w = int(w * padding_ratio)
        padded = cv2.copyMakeBorder(
            img_bgr, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        faces = _face_app.get(padded)

        if faces:
            # 패딩된 좌표를 원본 좌표로 복원
            best        = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = best.bbox.astype(int)
            x1 -= pad_w; x2 -= pad_w; y1 -= pad_h; y2 -= pad_h
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - margin * bw))
                y1 = max(0, int(y1 - margin * bh))
                x2 = min(w, int(x2 + margin * bw))
                y2 = min(h, int(y2 + margin * bh))
                return img_pil.crop((x1, y1, x2, y2)), CROP_RETRY_SUCCESS

        # 최종 실패 시 원본 반환
        return img_pil, CROP_FAILED

    # 1차 탐지 성공: 가장 큰 얼굴 선택
    best        = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = best.bbox.astype(int)
    bw, bh = x2 - x1, y2 - y1

    # 여백(margin) 추가
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))

    if x2 <= x1 or y2 <= y1:
        return img_pil, CROP_COORD_ERROR

    return img_pil.crop((x1, y1, x2, y2)), CROP_SUCCESS


# ── [v9 핵심] FFT 방식 B ─────────────────────────────────
# v7/v8 방식 A: FFT → log(mag) × weight (역FFT 없음, 주파수 도메인 맵)
# v9   방식 B: FFT → 시그모이드 고주파 마스크 → 역FFT → 공간 도메인 이미지 복원
#
# 학습 시(노트북 셀 b0e5e1b5)의 NumPy 기반 구현을 그대로 이식 — 가중치는 이 정확한
# 부동소수점 경로/마스크 정의에 적합되어 있으므로 torch.fft 등으로 바꾸지 말 것.
def apply_fft_highpass_B(
    img_tensor: torch.Tensor,
    cutoff: float = FFT_CUTOFF,
    steepness: float = FFT_STEEPNESS,
) -> torch.Tensor:
    img_np = img_tensor.cpu().numpy()
    result = np.zeros_like(img_np)
    H, W   = img_np.shape[1], img_np.shape[2]

    # 시그모이드 고주파 마스크 (학습 시와 동일하게 fftfreq 기반으로 생성)
    u = np.fft.fftfreq(H)
    v = np.fft.fftfreq(W)
    U, V = np.meshgrid(u, v, indexing='ij')
    D    = np.sqrt(U**2 + V**2)
    highpass_mask = 1 / (1 + np.exp(-steepness * (D - cutoff)))

    for c in range(img_np.shape[0]):
        fft          = np.fft.fft2(img_np[c])
        fft_filtered = fft * highpass_mask
        img_back     = np.fft.ifft2(fft_filtered).real    # 역FFT → 공간 도메인 복원
        result[c]    = np.clip(img_back, -1.0, 1.0)

    return torch.tensor(result, dtype=torch.float32)


# 이미지 경로를 받아서 모델 입력용 텐서 쌍 (x_rgb, x_fft)과 크롭 상태를 반환하는 함수
# fusion_fftB: 같은 base 텐서에서 RGB와 FFT 입력을 모두 파생 (노트북 셀 a4112a3e와 동일)
def preprocess_image(image_source) -> tuple[torch.Tensor, torch.Tensor, str]:
    # 추론 시에는 학습 augmentation 없이 CenterCrop만 적용
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    try:
        if isinstance(image_source, Image.Image):
            img_pil = image_source.convert("RGB")
        else:
            img_pil = Image.open(image_source).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"[에러] 이미지 파일을 찾을 수 없습니다: {image_source}")
    except Exception as e:
        raise RuntimeError(f"[에러] 이미지 로드 실패: {e}")

    img_cropped, crop_status = crop_face(img_pil)   # 얼굴 크롭 + 상태
    base = cast(torch.Tensor, eval_transform(img_cropped))  # (3, H, W) 정규화된 RGB 텐서
    x_rgb  = base                                   # RGB 브랜치 입력
    x_fftB = apply_fft_highpass_B(base)             # FFT 브랜치 입력 (방식 B 복원 이미지)

    # 배치 차원 추가 (모델은 배치 단위로 입력받음)
    return x_rgb.unsqueeze(0), x_fftB.unsqueeze(0), crop_status  # (1, 3, H, W) x 2 + 상태
