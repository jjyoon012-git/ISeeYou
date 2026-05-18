# 이 파일은 이미지 전처리 (얼굴 크롭, FFT 변환, 텐서 변환)를 담당합니다

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

import insightface
from insightface.app import FaceAnalysis

from config import IMG_SIZE, FACE_MARGIN, IMAGENET_MEAN, IMAGENET_STD


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


# RGB 텐서로부터 소프트 고주파 강조 FFT 텐서를 생성하는 함수
# 주파수 거리의 제곱으로 가중치를 줘서 고주파 성분을 부드럽게 강조
def apply_fft_highpass(img_tensor: torch.Tensor) -> torch.Tensor:
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    cy, cx = H // 2, W // 2

    # 각 픽셀의 주파수 중심 거리 계산 (멀수록 고주파)
    y = torch.arange(H) - cy
    x = torch.arange(W) - cx
    freq_dist = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    weight    = (freq_dist / freq_dist.max()) ** 2  # 제곱으로 고주파 강조

    channels = []
    for c in range(3):  # R, G, B 채널 각각 처리
        f   = torch.fft.fftshift(torch.fft.fft2(img_tensor[c].float()))
        mag = torch.log(torch.abs(f) + 1e-8) * weight  # 로그 스케일 + 고주파 가중치
        mag = (mag - mag.mean()) / (mag.std() + 1e-6)  # 정규화
        mag = torch.clamp(mag, -3.0, 3.0)              # 극단값 클리핑
        mag = (mag + 3.0) / 6.0                        # [0, 1] 범위로 스케일링
        channels.append(mag.unsqueeze(0))

    return torch.cat(channels, dim=0)  # (3, H, W)


# 이미지 경로를 받아서 모델 입력용 텐서 쌍 (x_rgb, x_fft)과 크롭 상태를 반환하는 함수
def preprocess_image(image_path: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    # 추론 시에는 학습 augmentation 없이 CenterCrop만 적용
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"[에러] 이미지 파일을 찾을 수 없습니다: {image_path}")
    except Exception as e:
        raise RuntimeError(f"[에러] 이미지 로드 실패: {e}")

    img_cropped, crop_status = crop_face(img_pil)  # 얼굴 크롭 + 상태
    x_rgb = eval_transform(img_cropped)             # (3, H, W) RGB 텐서
    x_fft = apply_fft_highpass(x_rgb)              # (3, H, W) FFT 텐서

    # 배치 차원 추가 (모델은 배치 단위로 입력받음)
    return x_rgb.unsqueeze(0), x_fft.unsqueeze(0), crop_status  # (1, 3, H, W) x 2 + 상태
