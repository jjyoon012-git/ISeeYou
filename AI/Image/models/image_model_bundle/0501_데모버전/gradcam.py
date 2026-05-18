# 이 파일은 Grad-CAM으로 모델이 어디를 보고 판단했는지 시각화합니다
# Late Fusion 모델이므로 RGB 브랜치와 FFT 브랜치 각각 따로 CAM을 추출합니다
# 사용법: python gradcam.py --image ../demo/fakes/aa1188a2-38bd-4f61-9004-bc6610950f77.png --device cpu

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import (
    MODEL_PATH, IMAGE_PATH, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
)
from model import load_model
from preprocess import preprocess_image

CLASS_NAMES = ["FAKE", "REAL"]   # logits[0]=Fake, logits[1]=Real


# ── matplotlib 한글 폰트 설정 (윈도우/맥/리눅스 자동 선택) ────────
for font_name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font_name
        break
plt.rcParams["axes.unicode_minus"] = False


# 특정 브랜치(rgb 또는 fft)의 마지막 conv 레이어에 hook을 걸어
# Grad-CAM 히트맵을 계산하는 함수
# target_class: 0=Fake / 1=Real (보통 모델이 예측한 클래스를 넣음)
def compute_gradcam(model, x_rgb: torch.Tensor, x_fft: torch.Tensor,
                    target_branch: str, target_class: int) -> np.ndarray:
    if target_branch == "rgb":
        target_layer = model.rgb_branch.conv_head
    elif target_branch == "fft":
        target_layer = model.fft_branch.conv_head
    else:
        raise ValueError(f"target_branch는 'rgb' 또는 'fft' (입력: {target_branch})")

    activations, gradients = [], []

    def fwd_hook(_m, _i, output):
        activations.append(output)

    def bwd_hook(_m, _gi, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # gradient 흘리기 위해 입력 텐서를 leaf로 복제 (원본은 보존)
    x_rgb_in = x_rgb.detach().clone().requires_grad_(True)
    x_fft_in = x_fft.detach().clone().requires_grad_(True)

    model.zero_grad()
    logits = model(x_rgb_in, x_fft_in)
    score  = logits[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    act  = activations[0]   # (1, C, h, w) — efficientnet_b4: (1, 1792, 7, 7)
    grad = gradients[0]     # 동일 shape

    # 채널별 GAP gradient → CAM weight
    weights = grad.mean(dim=(2, 3), keepdim=True)            # (1, C, 1, 1)
    cam = (weights * act).sum(dim=1, keepdim=True)            # (1, 1, h, w)
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)                            # [0, 1]
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy()                   # (224, 224)


# ImageNet 정규화된 RGB 텐서를 시각화용 [0,1] numpy 이미지로 복원
def denormalize_rgb(x_rgb: torch.Tensor) -> np.ndarray:
    img  = x_rgb[0].cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    return np.clip(img * std + mean, 0, 1)


# 원본 이미지 위에 CAM 히트맵을 컬러맵으로 덮어 씌우는 함수
def overlay_cam(img_01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * img_01 + alpha * heatmap, 0, 1)


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[에러] --device cuda 지정했지만 CUDA를 사용할 수 없습니다")
        return "cuda"
    # auto: GPU 사용 가능하고 메모리 1GB 이상 남아있을 때만 cuda, 아니면 cpu
    if torch.cuda.is_available():
        free_mem, _ = torch.cuda.mem_get_info()
        if free_mem >= 1.0 * 1024**3:
            return "cuda"
        print(f"[경고] GPU 여유 메모리 {free_mem/1024**2:.0f}MB 부족 → CPU로 전환")
    return "cpu"


def run(image_path: str, model_path: str, save_dir: str, device_arg: str = "auto") -> None:
    device = resolve_device(device_arg)
    print(f"[정보] 디바이스: {device}")
    print(f"[정보] 가중치 로드 중: {model_path}")
    model = load_model(model_path, device)

    print(f"[정보] 추론 중: {image_path}")
    x_rgb, x_fft, crop_status = preprocess_image(image_path)
    x_rgb, x_fft = x_rgb.to(device), x_fft.to(device)

    # 1) 예측 (gradient 불필요)
    with torch.no_grad():
        logits = model(x_rgb, x_fft)
        probs  = F.softmax(logits, dim=1)[0]
    pred_class = int(probs.argmax().item())
    pred_label = CLASS_NAMES[pred_class]
    fake_prob, real_prob = probs[0].item(), probs[1].item()

    # 2) 두 브랜치 모두 예측 클래스에 대해 Grad-CAM 추출
    cam_rgb = compute_gradcam(model, x_rgb, x_fft, "rgb", pred_class)
    cam_fft = compute_gradcam(model, x_rgb, x_fft, "fft", pred_class)

    # 3) 시각화용 입력 이미지 준비
    rgb_img = denormalize_rgb(x_rgb)
    fft_img = np.clip(x_fft[0].cpu().numpy().transpose(1, 2, 0), 0, 1)

    rgb_overlay = overlay_cam(rgb_img, cam_rgb)
    fft_overlay = overlay_cam(fft_img, cam_fft)

    # 4) 4-panel figure (원본 / RGB CAM / FFT 입력 / FFT CAM)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        (rgb_img,     "RGB 입력 (얼굴 크롭)"),
        (rgb_overlay, f"RGB 브랜치 Grad-CAM → {pred_label}"),
        (fft_img,     "FFT 입력 (고주파 강조)"),
        (fft_overlay, f"FFT 브랜치 Grad-CAM → {pred_label}"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{Path(image_path).name}  |  판정: {pred_label}  |  "
        f"Real={real_prob:.1%}  Fake={fake_prob:.1%}  |  얼굴크롭: {crop_status}",
        fontsize=12,
    )
    plt.tight_layout()

    # 5) 저장
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"gradcam_{Path(image_path).stem}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[완료] 저장: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimCap v7 Late Fusion Grad-CAM (RGB + FFT 두 브랜치 시각화)"
    )
    parser.add_argument("--image",    type=str, default=IMAGE_PATH, help="시각화할 이미지 경로")
    parser.add_argument("--model",    type=str, default=MODEL_PATH, help="가중치 파일 경로")
    parser.add_argument("--save-dir", type=str, default="gradcam_out", help="결과 저장 폴더")
    parser.add_argument("--device",   type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="추론 디바이스 (auto: GPU 여유 메모리 보고 자동 선택)")
    args = parser.parse_args()

    run(args.image, args.model, args.save_dir, args.device)
