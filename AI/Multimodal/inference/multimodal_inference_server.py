from __future__ import annotations

import base64
import cgi
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import imageio_ffmpeg
import numpy as np
import open_clip
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps

try:
    import yt_dlp  # type: ignore
except ImportError:  # pragma: no cover
    yt_dlp = None


HOST = "127.0.0.1"
PORT = 8001
FRAME_SIZE = 224
SAMPLE_FRAMES = 12
AUDIO_SR = 16000
MAX_SECONDS = 8.0
SAMPLING_WINDOWS = 3
OPENAI_MODEL = os.environ.get("OPENAI_API_MODEL", "gpt-5.5")
RUNTIME_SOURCE_DIR = Path(r"D:\ISeeYou\experiments\final5000_gpu_anchor_fusion_v4b\full_model")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_ROOT = PROJECT_ROOT / "AI"
RUNTIME_CACHE_PATH = AI_ROOT / "Multimodal" / "models" / "_service_runtime_bundle_v4b.pt"
TEXT_MODEL_BUNDLE_DIR = AI_ROOT / "Text" / "models" / "text_model_bundle"
TEXT_MODEL_LOCK = threading.Lock()
TEXT_MODEL_STATE: dict[str, Any] = {}
VIDEO_MODEL_BUNDLE_DIR = AI_ROOT / "Video" / "models" / "video"
VIDEO_MODEL_LOCK = threading.Lock()
VIDEO_MODEL_STATE: dict[str, Any] = {}
VIDEO_SAMPLE_FRAMES = 6
VIDEO_MODEL_SPECS = [
    {
        "label": "robustaug",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame",
        "image_size": 224,
        "role": "no EMA diversity",
    },
    {
        "label": "robustaug+EMA",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_frame",
        "image_size": 224,
        "role": "seed 42 EMA baseline",
    },
    {
        "label": "ff2f_holdout",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout_frame",
        "image_size": 224,
        "role": "Face2Face holdout generalization",
    },
    {
        "label": "seed1337",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame",
        "image_size": 224,
        "role": "random seed diversity",
    },
    {
        "label": "seed7",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed7_frame",
        "image_size": 224,
        "role": "random seed diversity",
    },
    {
        "label": "df_holdout",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_df_holdout_frame",
        "image_size": 224,
        "role": "Deepfakes holdout generalization",
    },
    {
        "label": "img320",
        "folder": "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_img320_frame",
        "image_size": 320,
        "role": "higher resolution diversity",
    },
]
METHOD_KEYS = ["openclip", "flava", "blip_nli", "avsync", "frequency", "scenegraph"]
MODE_TO_METHOD = {
    "mm-openclip": "openclip",
    "mm-flava": "flava",
    "mm-blip-nli": "blip_nli",
    "mm-avsync": "avsync",
    "mm-frequency": "frequency",
    "mm-scenegraph": "scenegraph",
}

SELECTED_MODE_ALIASES = {
    "video-openclip": "mm-openclip",
    "video-flava": "mm-flava",
    "video-blip-nli": "mm-blip-nli",
    "video-avsync": "mm-avsync",
    "video-frequency": "mm-frequency",
    "video-scenegraph": "mm-scenegraph",
}

HAAR_ROOT = Path(cv2.data.haarcascades)
FACE_CASCADES = [
    cv2.CascadeClassifier(str(HAAR_ROOT / "haarcascade_frontalface_default.xml")),
    cv2.CascadeClassifier(str(HAAR_ROOT / "haarcascade_frontalface_alt2.xml")),
    cv2.CascadeClassifier(str(HAAR_ROOT / "haarcascade_profileface.xml")),
]
SMILE_CASCADE = cv2.CascadeClassifier(str(HAAR_ROOT / "haarcascade_smile.xml"))
MODEL_LOCK = threading.Lock()
MODEL_STATE: dict[str, Any] = {}
IMAGE_MODEL_LOCK = threading.Lock()
IMAGE_MODEL_STATE: dict[str, Any] = {}
IMAGE_FACE_MODEL_LOCK = threading.Lock()
IMAGE_FACE_MODEL_STATE: dict[str, Any] = {}
IMAGE_MODEL_BUNDLE_DIR = AI_ROOT / "Image" / "models" / "image_model_bundle"


def _first_existing_checkpoint(filename: str) -> Path:
    for candidate in IMAGE_MODEL_BUNDLE_DIR.rglob(filename):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image checkpoint not found: {filename}")


IMAGE_MODEL_CKPT = _first_existing_checkpoint("best_dualstream_final.pt")
IMAGE_FACE_MODEL_CKPT = _first_existing_checkpoint("best.pt")


class FFTBranch(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class DualStreamDetector(nn.Module):
    def __init__(self, fft_dim: int = 256):
        super().__init__()
        self.rgb_backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.fft_branch = FFTBranch(out_dim=fft_dim)
        fusion_dim = self.rgb_backbone.num_features + fft_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rgb: torch.Tensor, fft: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(rgb)
        fft_feat = self.fft_branch(fft)
        fused = torch.cat([rgb_feat, fft_feat], dim=1)
        return self.classifier(fused).squeeze(1)


def make_efficientnet_b4(in_channels: int = 3, num_classes: int = 2) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=num_classes,
    )
    if in_channels != 3:
        old_conv = model.conv_stem
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for channel in range(3, in_channels):
                new_conv.weight[:, channel : channel + 1, :, :] = avg_weight
        model.conv_stem = new_conv
    return model


class LateFusionFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = make_efficientnet_b4(in_channels=3, num_classes=0)
        self.fft_branch = make_efficientnet_b4(in_channels=3, num_classes=0)
        feat_dim = self.rgb_branch.num_features
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x_rgb: torch.Tensor, x_fft: torch.Tensor) -> torch.Tensor:
        feat_rgb = self.rgb_branch(x_rgb)
        feat_fft = self.fft_branch(x_fft)
        feat_cat = torch.cat([feat_rgb, feat_fft], dim=1)
        return self.fc(feat_cat)


class VideoFrameClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.4,
        hidden_dim: int = 0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_local_env()


@dataclass
class FrameSignal:
    timestamp: float
    frame_rgb: np.ndarray
    face_box: tuple[int, int, int, int] | None
    mouth_box: tuple[int, int, int, int] | None
    motion_score: float
    face_crop: np.ndarray


@dataclass
class SegmentInsight:
    label: str
    start_sec: float
    end_sec: float
    score: float
    motion_mean: float
    audio_mean: float
    face_ratio: float
    mouth_ratio: float
    sharpness_mean: float


def analysis_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_selected_mode(selected_mode: str) -> str:
    cleaned = (selected_mode or "mm-flava").strip().lower()
    return SELECTED_MODE_ALIASES.get(cleaned, cleaned)


def validate_remote_video_url(raw_url: str) -> str:
    normalized = (raw_url or "").strip()
    if not normalized:
        raise ValueError("video url is required")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("http/https 영상 주소만 분석할 수 있습니다.")
    return normalized


def download_remote_video(remote_url: str, download_dir: Path) -> Path:
    output_template = str(download_dir / "remote_video.%(ext)s")
    preferred_format = "best[ext=mp4][height<=720]/best[height<=720]/best"
    if yt_dlp is not None:
        options = {
            "format": preferred_format,
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(remote_url, download=True)
            final_path = downloader.prepare_filename(info)
            merged_path = Path(final_path)
            if merged_path.suffix.lower() != ".mp4":
                candidate = merged_path.with_suffix(".mp4")
                if candidate.exists():
                    merged_path = candidate
            if merged_path.exists():
                return merged_path
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        preferred_format,
        "-o",
        output_template,
        remote_url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "yt-dlp download failed"
        raise RuntimeError(f"원격 영상 다운로드에 실패했습니다: {stderr}")
    files = sorted(download_dir.glob("remote_video.*"))
    if not files:
        raise RuntimeError("원격 영상 다운로드 결과 파일을 찾지 못했습니다.")
    mp4_files = [path for path in files if path.suffix.lower() == ".mp4"]
    return mp4_files[0] if mp4_files else files[0]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def contract_toward_neutral(score: float, strength: float) -> float:
    return clamp(0.5 + (score - 0.5) * strength, 0.0, 1.0)


def format_mmss(seconds: float) -> str:
    total = max(int(seconds), 0)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def rgb_array_to_data_url(image_rgb: np.ndarray, fmt: str = "PNG") -> str:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def compute_fft_map(gray_image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray_image, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(resized.astype(np.float32))
    shifted = np.fft.fftshift(spectrum)
    magnitude = np.log1p(np.abs(shifted))
    normalized = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized.astype(np.float32)


def fft_map_to_tensor(fft_map: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(fft_map, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def fft_map_to_data_url(fft_map: np.ndarray) -> str:
    image_u8 = np.clip(np.asarray(fft_map) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return rgb_array_to_data_url(colored)


def fft_map_to_rgb_image(fft_map: np.ndarray) -> Image.Image:
    image_u8 = np.clip(np.asarray(fft_map) * 255.0, 0, 255).astype(np.uint8)
    stacked = np.repeat(image_u8[:, :, None], 3, axis=2)
    return Image.fromarray(stacked)


def crop_image_box(frame_rgb: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    x, y, w, h = box
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, frame_rgb.shape[1])
    y1 = min(y + h, frame_rgb.shape[0])
    if x1 <= x0 or y1 <= y0:
        return cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    cropped = frame_rgb[y0:y1, x0:x1]
    return cv2.resize(cropped, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)


def build_image_frequency_profile(fft_map: np.ndarray) -> list[float]:
    radial = np.mean(fft_map, axis=0)
    if radial.size == 0:
        return [0.0] * 7
    chunks = np.array_split(radial, 7)
    values = [float(np.mean(chunk) * 100.0) if len(chunk) else 0.0 for chunk in chunks]
    return [round(clamp(value, 0.0, 100.0), 1) for value in values]


def compute_image_references(fft_map: np.ndarray) -> tuple[list[float], list[float]]:
    sample = np.array(build_image_frequency_profile(fft_map), dtype=np.float32)
    real_ref = np.clip(sample * np.array([0.96, 0.98, 1.0, 0.94, 0.88, 0.82, 0.78], dtype=np.float32) + np.array([6, 5, 4, 2, 0, -2, -4], dtype=np.float32), 0.0, 100.0)
    fake_ref = np.clip(sample * np.array([0.72, 0.78, 0.86, 1.0, 1.08, 1.14, 1.18], dtype=np.float32) + np.array([-8, -5, -2, 1, 4, 7, 9], dtype=np.float32), 0.0, 100.0)
    return [round(float(value), 1) for value in real_ref], [round(float(value), 1) for value in fake_ref]


def download_remote_image(remote_url: str, download_dir: Path) -> Path:
    parsed = urlparse(remote_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("http/https 이미지 주소만 분석할 수 있습니다.")
    request = urlrequest.Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlrequest.urlopen(request, timeout=25) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "image" not in content_type and not remote_url.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                raise ValueError("직접 접근 가능한 이미지 URL만 분석할 수 있습니다.")
            payload = response.read()
    except urlerror.URLError as error:
        raise RuntimeError(f"이미지 다운로드에 실패했습니다: {error}") from error

    output_path = download_dir / "remote_image.png"
    output_path.write_bytes(payload)
    return output_path


def ensure_model() -> tuple[Any, Any, Any, str]:
    with MODEL_LOCK:
        if MODEL_STATE:
            return (
                MODEL_STATE["model"],
                MODEL_STATE["preprocess"],
                MODEL_STATE["tokenizer"],
                MODEL_STATE["device"],
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=device,
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model.eval()
        MODEL_STATE.update(
            {
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
                "device": device,
            }
        )
        return model, preprocess, tokenizer, device


def ensure_image_model() -> tuple[DualStreamDetector, T.Compose, str]:
    with IMAGE_MODEL_LOCK:
        if IMAGE_MODEL_STATE:
            return (
                IMAGE_MODEL_STATE["model"],
                IMAGE_MODEL_STATE["transform"],
                IMAGE_MODEL_STATE["device"],
            )

        if not IMAGE_MODEL_CKPT.exists():
            raise FileNotFoundError(f"image checkpoint not found: {IMAGE_MODEL_CKPT}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DualStreamDetector().to(device)
        checkpoint = torch.load(IMAGE_MODEL_CKPT, map_location=device)
        state_dict = extract_image_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        transform = T.Compose(
            [
                T.Resize((FRAME_SIZE, FRAME_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        IMAGE_MODEL_STATE.update({"model": model, "transform": transform, "device": device})
        return model, transform, device


def extract_image_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("unsupported checkpoint format for image model")


def ensure_face_image_model() -> tuple[LateFusionFaceModel, T.Compose, str]:
    with IMAGE_FACE_MODEL_LOCK:
        if IMAGE_FACE_MODEL_STATE:
            return (
                IMAGE_FACE_MODEL_STATE["model"],
                IMAGE_FACE_MODEL_STATE["transform"],
                IMAGE_FACE_MODEL_STATE["device"],
            )

        if not IMAGE_FACE_MODEL_CKPT.exists():
            raise FileNotFoundError(f"face image checkpoint not found: {IMAGE_FACE_MODEL_CKPT}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LateFusionFaceModel().to(device)
        checkpoint = torch.load(IMAGE_FACE_MODEL_CKPT, map_location=device)
        state_dict = extract_image_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        transform = T.Compose(
            [
                T.Resize((FRAME_SIZE, FRAME_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        IMAGE_FACE_MODEL_STATE.update({"model": model, "transform": transform, "device": device})
        return model, transform, device


def gradcam_for_image_model(
    model: DualStreamDetector,
    rgb_tensor: torch.Tensor,
    fft_tensor: torch.Tensor,
    *,
    fake_focus: bool,
) -> np.ndarray:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations["value"] = output

    def backward_hook(_module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        if grad_output:
            gradients["value"] = grad_output[0]

    target_layer = getattr(model.rgb_backbone, "conv_head", None)
    if target_layer is None:
        return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(rgb_tensor, fft_tensor)
        score = 1.0 - torch.sigmoid(logits) if fake_focus else torch.sigmoid(logits)
        score.sum().backward()
        feature_map = activations.get("value")
        grad_map = gradients.get("value")
        if feature_map is None or grad_map is None:
            return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        weights = grad_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False)
        cam_np = cam.detach().cpu().squeeze().numpy().astype(np.float32)
        peak = float(cam_np.max())
        if peak > 1e-6:
            cam_np /= peak
        return cam_np
    finally:
        handle_forward.remove()
        handle_backward.remove()


def gradcam_for_face_image_model(
    model: LateFusionFaceModel,
    rgb_tensor: torch.Tensor,
    fft_tensor: torch.Tensor,
    *,
    fake_focus: bool,
) -> np.ndarray:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations["value"] = output

    def backward_hook(_module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        if grad_output:
            gradients["value"] = grad_output[0]

    target_layer = getattr(model.rgb_branch, "conv_head", None)
    if target_layer is None:
        return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(rgb_tensor, fft_tensor)
        class_index = 0 if fake_focus else 1
        score = logits[:, class_index]
        score.sum().backward()
        feature_map = activations.get("value")
        grad_map = gradients.get("value")
        if feature_map is None or grad_map is None:
            return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        weights = grad_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False)
        cam_np = cam.detach().cpu().squeeze().numpy().astype(np.float32)
        peak = float(cam_np.max())
        if peak > 1e-6:
            cam_np /= peak
        return cam_np
    finally:
        handle_forward.remove()
        handle_backward.remove()


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-6:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def scale_with_stats(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = float(stats["min"])
    maximum = float(stats["max"])
    if maximum - minimum < 1e-6:
        return np.zeros_like(values)
    scaled = (values - minimum) / (maximum - minimum)
    return np.clip(scaled, 0.0, 1.0)


def softened_reliability(
    precheck: np.ndarray,
    *,
    floor: float,
    severe_mask: np.ndarray | None = None,
    severe_scale: float = 0.55,
    min_clip: float = 0.10,
) -> np.ndarray:
    reliability = floor + (1.0 - floor) * np.clip(precheck, 0.0, 1.0)
    if severe_mask is not None:
        reliability = np.where(severe_mask, reliability * severe_scale, reliability)
    return np.clip(reliability, min_clip, 1.0)


def compute_runtime_stats(train_frame: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for column in ("visual_sharpness", "audio_energy_mean", "audio_energy_std", "motion_mean", "mean_face_area"):
        values = train_frame[column].fillna(0).to_numpy(dtype=np.float32)
        stats[column] = {"min": float(values.min()), "max": float(values.max())}
    stats["gate_thresholds"] = {
        "qsharp": max(float(train_frame["visual_sharpness"].quantile(0.20)), 1e-6),
        "qaudio": max(float(train_frame["audio_energy_mean"].quantile(0.20)), 1e-6),
        "qmotion": float(train_frame["motion_mean"].quantile(0.20)),
    }
    return stats


def add_service_quality_features(frame: pd.DataFrame, runtime_stats: dict[str, Any]) -> pd.DataFrame:
    enriched = frame.copy()
    token_counts = enriched["text_prompt"].fillna("").astype(str).str.split().str.len().clip(lower=0).to_numpy(dtype=np.float32)
    text_exists = (token_counts > 0).astype(np.float32)
    text_density = np.clip(token_counts / 8.0, 0.0, 1.0)
    duration = enriched["duration_sec"].fillna(0).to_numpy(dtype=np.float32)
    duration_reliability = np.clip(duration / 12.0, 0.25, 1.0)
    face_count_signal = np.where(enriched["face_count"].fillna(0).to_numpy(dtype=np.float32) > 0, 1.0, 0.35)
    audio_exists = enriched["audio_path"].fillna("").astype(str).str.len().gt(0).astype(float).to_numpy(dtype=np.float32)

    sharpness_norm = scale_with_stats(enriched["visual_sharpness"].fillna(0).to_numpy(dtype=np.float32), runtime_stats["visual_sharpness"])
    audio_energy_norm = scale_with_stats(enriched["audio_energy_mean"].fillna(0).to_numpy(dtype=np.float32), runtime_stats["audio_energy_mean"])
    audio_dynamic_norm = scale_with_stats(enriched["audio_energy_std"].fillna(0).to_numpy(dtype=np.float32), runtime_stats["audio_energy_std"])
    motion_norm = scale_with_stats(enriched["motion_mean"].fillna(0).to_numpy(dtype=np.float32), runtime_stats["motion_mean"])
    face_area_norm = scale_with_stats(enriched["mean_face_area"].fillna(0).to_numpy(dtype=np.float32), runtime_stats["mean_face_area"])
    face_detect_ratio = enriched["face_detect_ratio"].fillna(0).to_numpy(dtype=np.float32)
    mouth_track_ratio = enriched["mouth_track_ratio"].fillna(0).to_numpy(dtype=np.float32)

    text_quality = 0.35 * text_exists + 0.65 * text_density
    audio_quality = audio_exists * (0.65 * audio_energy_norm + 0.35 * audio_dynamic_norm)
    visual_quality = 0.35 * sharpness_norm + 0.35 * face_detect_ratio + 0.20 * face_area_norm + 0.10 * duration_reliability
    temporal_quality = 0.45 * motion_norm + 0.30 * duration_reliability + 0.25 * face_detect_ratio

    enriched["text_tokens"] = token_counts
    enriched["face_count_signal"] = face_count_signal
    enriched["precheck_openclip"] = 0.45 * visual_quality + 0.35 * text_quality + 0.20 * face_count_signal
    enriched["precheck_flava"] = 0.28 * visual_quality + 0.22 * text_quality + 0.20 * audio_quality + 0.15 * temporal_quality + 0.15 * face_detect_ratio
    enriched["precheck_blip_nli"] = 0.25 * visual_quality + 0.50 * text_quality + 0.15 * face_detect_ratio + 0.10 * temporal_quality
    enriched["precheck_avsync"] = 0.40 * face_detect_ratio + 0.35 * mouth_track_ratio + 0.15 * audio_quality + 0.10 * duration_reliability
    enriched["precheck_frequency"] = 0.28 * sharpness_norm + 0.22 * audio_quality + 0.15 * motion_norm + 0.15 * duration_reliability + 0.20 * (1.0 - np.clip(enriched["segment_mean"].fillna(0.5).to_numpy(dtype=np.float32), 0.0, 1.0))
    enriched["precheck_scenegraph"] = 0.45 * face_detect_ratio + 0.25 * face_area_norm + 0.20 * motion_norm + 0.10 * sharpness_norm

    openclip_severe = (text_exists < 0.5) & (visual_quality < 0.30)
    flava_severe = (visual_quality < 0.25) & (text_quality < 0.20) & (audio_quality < 0.20)
    blip_severe = (text_exists < 0.5) | (text_density < 0.12)
    avsync_severe = (face_detect_ratio < 0.25) | (mouth_track_ratio < 0.15) | (audio_quality < 0.15)
    frequency_severe = (sharpness_norm < 0.15) & (audio_quality < 0.15)
    scenegraph_severe = (face_detect_ratio < 0.20) & (motion_norm < 0.10)

    enriched["reliability_openclip"] = softened_reliability(enriched["precheck_openclip"].to_numpy(dtype=np.float32), floor=0.82, severe_mask=openclip_severe, severe_scale=0.68, min_clip=0.18)
    enriched["reliability_flava"] = softened_reliability(enriched["precheck_flava"].to_numpy(dtype=np.float32), floor=0.80, severe_mask=flava_severe, severe_scale=0.62, min_clip=0.18)
    enriched["reliability_blip_nli"] = softened_reliability(enriched["precheck_blip_nli"].to_numpy(dtype=np.float32), floor=0.80, severe_mask=blip_severe, severe_scale=0.48, min_clip=0.12)
    enriched["reliability_avsync"] = softened_reliability(enriched["precheck_avsync"].to_numpy(dtype=np.float32), floor=0.78, severe_mask=avsync_severe, severe_scale=0.38, min_clip=0.08)
    enriched["reliability_frequency"] = softened_reliability(enriched["precheck_frequency"].to_numpy(dtype=np.float32), floor=0.84, severe_mask=frequency_severe, severe_scale=0.68, min_clip=0.20)
    enriched["reliability_scenegraph"] = softened_reliability(enriched["precheck_scenegraph"].to_numpy(dtype=np.float32), floor=0.82, severe_mask=scenegraph_severe, severe_scale=0.55, min_clip=0.15)
    return enriched


def apply_gate_features(frame: pd.DataFrame, gate_thresholds: dict[str, float]) -> pd.DataFrame:
    enriched = frame.copy()
    prob_cols = [f"prob_fake_{name}" for name in METHOD_KEYS]
    enriched["simple_avg"] = enriched[prob_cols].mean(axis=1)
    enriched["consensus_std"] = enriched[prob_cols].std(axis=1)
    enriched["segment_inv"] = 1.0 - (
        0.75 * enriched["segment_topk_mean"].to_numpy(dtype=np.float32)
        + 0.25 * enriched["segment_peak"].to_numpy(dtype=np.float32)
    )

    qsharp = gate_thresholds["qsharp"]
    qaudio = gate_thresholds["qaudio"]
    qmotion = gate_thresholds["qmotion"]
    text_tokens = enriched["text_tokens"].to_numpy(dtype=np.float32)
    face_detect_ratio = enriched["face_detect_ratio"].to_numpy(dtype=np.float32)
    mouth_track_ratio = enriched["mouth_track_ratio"].to_numpy(dtype=np.float32)
    visual_sharpness = enriched["visual_sharpness"].to_numpy(dtype=np.float32)
    audio_energy_mean = enriched["audio_energy_mean"].to_numpy(dtype=np.float32)
    motion_mean = enriched["motion_mean"].to_numpy(dtype=np.float32)

    gate_rules = {
        "openclip": np.where((text_tokens < 3) & (visual_sharpness < qsharp), 0.15, 1.0),
        "flava": np.where((face_detect_ratio < 0.20) & (audio_energy_mean < qaudio), 0.25, 1.0),
        "blip_nli": np.where(text_tokens < 3, 0.10, 1.0),
        "avsync": np.where((face_detect_ratio < 0.30) | (mouth_track_ratio < 0.20) | (audio_energy_mean < qaudio), 0.05, 1.0),
        "frequency": np.where((visual_sharpness < qsharp) & (audio_energy_mean < qaudio), 0.20, 1.0),
        "scenegraph": np.where((face_detect_ratio < 0.20) & (motion_mean < qmotion), 0.20, 1.0),
    }
    for method_name in METHOD_KEYS:
        enriched[f"gate_{method_name}"] = gate_rules[method_name].astype(np.float32)
        reliability_term = 0.70 + 0.30 * enriched[f"reliability_{method_name}"].to_numpy(dtype=np.float32)
        enriched[f"gated_rel_{method_name}"] = (
            enriched[f"prob_fake_{method_name}"].to_numpy(dtype=np.float32)
            * enriched[f"gate_{method_name}"].to_numpy(dtype=np.float32)
            * reliability_term
        )
        enriched[f"gated_hard_{method_name}"] = (
            enriched[f"prob_fake_{method_name}"].to_numpy(dtype=np.float32)
            * enriched[f"gate_{method_name}"].to_numpy(dtype=np.float32)
        )
    return enriched


def compute_base_method_weights(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> dict[str, float]:
    raw_weights: dict[str, float] = {}
    train_labels = train_frame["label_id"].to_numpy(dtype=np.float32)
    val_labels = val_frame["label_id"].to_numpy(dtype=np.float32)
    for method_name in METHOD_KEYS:
        train_probs = train_frame[f"prob_fake_{method_name}"].to_numpy(dtype=np.float32)
        val_probs = val_frame[f"prob_fake_{method_name}"].to_numpy(dtype=np.float32)
        train_score = max(float(np.corrcoef(train_probs, train_labels)[0, 1]) if np.std(train_probs) > 1e-6 else 0.0, 0.01)
        val_score = max(float(np.corrcoef(val_probs, val_labels)[0, 1]) if np.std(val_probs) > 1e-6 else 0.0, 0.01)
        raw_weights[method_name] = 0.55 * train_score + 0.45 * val_score
    total = sum(raw_weights.values()) or 1.0
    return {name: value / total for name, value in raw_weights.items()}


class ServiceHeadNet(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 24),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(24, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(x))


class AnchorFusionNet(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.12),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.08),
        )
        self.branch = torch.nn.Linear(64, 1)
        self.router = torch.nn.Linear(64, 1)
        self.segment_gate = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, anchor: torch.Tensor, segment_score: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        branch_prob = torch.sigmoid(self.branch(hidden))
        route = torch.sigmoid(self.router(hidden))
        seg_gate = 0.20 * torch.sigmoid(self.segment_gate(hidden))
        mixed = (1.0 - route) * anchor + route * branch_prob
        output = (1.0 - seg_gate) * mixed + seg_gate * segment_score
        return output.clamp(1e-4, 1.0 - 1e-4)


def service_head_feature_map() -> dict[str, list[str]]:
    common = [
        "prob_fake_openclip",
        "segment_mean",
        "segment_topk_mean",
        "segment_peak",
        "duration_sec",
    ]
    return {
        "flava": common
        + [
            "face_detect_ratio",
            "mean_face_area",
            "mouth_track_ratio",
            "visual_sharpness",
            "audio_energy_mean",
            "audio_energy_std",
            "motion_mean",
            "text_tokens",
            "precheck_flava",
            "reliability_flava",
            "precheck_openclip",
            "reliability_openclip",
        ],
        "blip_nli": common
        + [
            "face_detect_ratio",
            "visual_sharpness",
            "text_tokens",
            "precheck_blip_nli",
            "reliability_blip_nli",
            "precheck_openclip",
            "reliability_openclip",
        ],
        "avsync": common
        + [
            "face_detect_ratio",
            "mean_face_area",
            "mouth_track_ratio",
            "audio_energy_mean",
            "audio_energy_std",
            "motion_mean",
            "precheck_avsync",
            "reliability_avsync",
        ],
        "frequency": common
        + [
            "visual_sharpness",
            "audio_energy_mean",
            "audio_energy_std",
            "motion_mean",
            "precheck_frequency",
            "reliability_frequency",
        ],
        "scenegraph": common
        + [
            "face_detect_ratio",
            "mean_face_area",
            "motion_mean",
            "visual_sharpness",
            "precheck_scenegraph",
            "reliability_scenegraph",
        ],
    }


def fusion_feature_names() -> list[str]:
    return [
        *(f"prob_fake_{name}" for name in METHOD_KEYS),
        "simple_avg",
        "consensus_std",
        "weighted_model_score",
        "segment_mean",
        "face_detect_ratio",
        "mean_face_area",
        "mouth_track_ratio",
        "visual_sharpness",
        "audio_energy_mean",
        "audio_energy_std",
        "motion_mean",
        "segment_topk_mean",
        "segment_peak",
        "segment_inv",
        *(f"gate_{name}" for name in METHOD_KEYS),
        *(f"gated_rel_{name}" for name in METHOD_KEYS),
    ]


def _tensorize(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = frame[feature_names].to_numpy(dtype=np.float32)
    labels = frame["label_id"].to_numpy(dtype=np.float32)
    return features, labels


def train_service_head(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_names: list[str],
    epochs: int = 120,
) -> tuple[ServiceHeadNet, dict[str, np.ndarray]]:
    x_train = train_frame[feature_names].to_numpy(dtype=np.float32)
    x_val = val_frame[feature_names].to_numpy(dtype=np.float32)
    y_train = train_frame["label_id"].to_numpy(dtype=np.float32)
    y_val = val_frame["label_id"].to_numpy(dtype=np.float32)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)

    device = torch.device("cpu")
    model = ServiceHeadNet(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()
    x_train_t = torch.tensor(x_train, device=device)
    y_train_t = torch.tensor(y_train[:, None], device=device)
    x_val_t = torch.tensor(x_val, device=device)
    y_val_t = torch.tensor(y_val[:, None], device=device)
    best_state = None
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        preds = model(x_train_t)
        loss = criterion(preds, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 2 != 0:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std}


def train_service_fusion(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_names: list[str],
    epochs: int = 110,
) -> tuple[AnchorFusionNet, dict[str, np.ndarray], float]:
    x_train, y_train = _tensorize(train_frame, feature_names)
    x_val, y_val = _tensorize(val_frame, feature_names)
    anchor_train = train_frame["simple_avg"].to_numpy(dtype=np.float32)
    anchor_val = val_frame["simple_avg"].to_numpy(dtype=np.float32)
    segment_train = train_frame["segment_inv"].to_numpy(dtype=np.float32)
    segment_val = val_frame["segment_inv"].to_numpy(dtype=np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)

    device = torch.device("cpu")
    model = AnchorFusionNet(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    x_train_t = torch.tensor(x_train, device=device)
    y_train_t = torch.tensor(y_train[:, None], device=device)
    x_val_t = torch.tensor(x_val, device=device)
    anchor_train_t = torch.tensor(anchor_train[:, None], device=device)
    anchor_val_t = torch.tensor(anchor_val[:, None], device=device)
    segment_train_t = torch.tensor(segment_train[:, None], device=device)
    segment_val_t = torch.tensor(segment_val[:, None], device=device)

    best_state = None
    best_score = -1.0
    best_threshold = 0.5
    for epoch in range(epochs):
        model.train()
        preds = model(x_train_t, anchor_train_t, segment_train_t)
        loss = criterion(preds, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 2 != 0:
            continue
        model.eval()
        with torch.no_grad():
            val_scores = model(x_val_t, anchor_val_t, segment_val_t).squeeze(1).detach().cpu().numpy()
        local_best_score = -1.0
        local_best_threshold = 0.5
        for threshold in np.arange(0.40, 0.61, 0.01):
            preds_binary = (val_scores >= threshold).astype(np.int64)
            accuracy = float((preds_binary == y_val.astype(np.int64)).mean())
            precision = float(((preds_binary == 1) & (y_val == 1)).sum() / max((preds_binary == 1).sum(), 1))
            recall = float(((preds_binary == 1) & (y_val == 1)).sum() / max((y_val == 1).sum(), 1))
            f1 = (2 * precision * recall / max(precision + recall, 1e-6)) if (precision + recall) > 0 else 0.0
            objective = 0.45 * f1 + 0.25 * accuracy + 0.15 * precision + 0.15 * recall
            if objective > local_best_score:
                local_best_score = objective
                local_best_threshold = float(threshold)
        if local_best_score > best_score:
            best_score = local_best_score
            best_threshold = local_best_threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std}, best_threshold


def train_runtime_bundle() -> dict[str, Any]:
    if not RUNTIME_SOURCE_DIR.exists():
        raise FileNotFoundError(f"runtime source dir not found: {RUNTIME_SOURCE_DIR}")
    frames = {
        split: pd.read_csv(RUNTIME_SOURCE_DIR / f"{split}_predictions.csv", encoding="utf-8-sig")
        for split in ("train", "val", "test")
    }
    runtime_stats = compute_runtime_stats(frames["train"])
    for split in ("train", "val", "test"):
        frames[split] = add_service_quality_features(frames[split], runtime_stats)
        frames[split] = apply_gate_features(frames[split], runtime_stats["gate_thresholds"])

    head_feature_map = service_head_feature_map()
    head_targets = ["flava", "blip_nli", "avsync", "frequency", "scenegraph"]
    head_models: dict[str, ServiceHeadNet] = {}
    head_norms: dict[str, dict[str, np.ndarray]] = {}
    for name in head_targets:
        model, norm = train_service_head(frames["train"], frames["val"], head_feature_map[name])
        head_models[name] = model
        head_norms[name] = norm

    runtime_frames = {}
    for split in ("train", "val", "test"):
        runtime_frames[split] = materialize_runtime_heads(
            frames[split],
            head_targets=head_targets,
            head_models=head_models,
            head_norms=head_norms,
            head_feature_map=head_feature_map,
        )

    base_weights = compute_base_method_weights(runtime_frames["train"], runtime_frames["val"])
    fusion_names = fusion_feature_names()
    for split in ("train", "val", "test"):
        weighted_scores = []
        for _, row in runtime_frames[split].iterrows():
            weighted_score, _ = compute_runtime_weighted_model_score(row, base_weights)
            weighted_scores.append(weighted_score)
        runtime_frames[split]["weighted_model_score"] = np.asarray(weighted_scores, dtype=np.float32)

    fusion_model, fusion_norm, fusion_threshold = train_service_fusion(
        runtime_frames["train"],
        runtime_frames["val"],
        fusion_names,
    )
    return {
        "runtime_stats": runtime_stats,
        "base_weights": base_weights,
        "head_feature_map": head_feature_map,
        "fusion_feature_names": fusion_names,
        "head_targets": head_targets,
        "head_models": head_models,
        "head_norms": head_norms,
        "fusion_model": fusion_model,
        "fusion_norm": fusion_norm,
        "fusion_threshold": fusion_threshold,
    }


def serialize_runtime_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_stats": bundle["runtime_stats"],
        "base_weights": bundle["base_weights"],
        "head_feature_map": bundle["head_feature_map"],
        "fusion_feature_names": bundle["fusion_feature_names"],
        "head_targets": bundle["head_targets"],
        "head_norms": bundle["head_norms"],
        "fusion_norm": bundle["fusion_norm"],
        "fusion_threshold": bundle["fusion_threshold"],
        "head_states": {name: model.state_dict() for name, model in bundle["head_models"].items()},
        "fusion_state": bundle["fusion_model"].state_dict(),
    }


def deserialize_runtime_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(analysis_device())
    head_models: dict[str, ServiceHeadNet] = {}
    for name, state in payload["head_states"].items():
        model = ServiceHeadNet(len(payload["head_feature_map"][name])).to(device)
        model.load_state_dict(state)
        model.eval()
        head_models[name] = model
    fusion_model = AnchorFusionNet(len(payload["fusion_feature_names"])).to(device)
    fusion_model.load_state_dict(payload["fusion_state"])
    fusion_model.eval()
    return {
        "runtime_stats": payload["runtime_stats"],
        "base_weights": payload["base_weights"],
        "head_feature_map": payload["head_feature_map"],
        "fusion_feature_names": payload["fusion_feature_names"],
        "head_targets": payload["head_targets"],
        "head_norms": payload["head_norms"],
        "fusion_norm": payload["fusion_norm"],
        "fusion_threshold": payload["fusion_threshold"],
        "head_models": head_models,
        "fusion_model": fusion_model,
    }


def ensure_runtime_bundle() -> dict[str, Any]:
    with MODEL_LOCK:
        bundle = MODEL_STATE.get("runtime_bundle")
        if bundle is not None:
            return bundle
        if RUNTIME_CACHE_PATH.exists():
            payload = torch.load(RUNTIME_CACHE_PATH, map_location=analysis_device(), weights_only=False)
            bundle = deserialize_runtime_bundle(payload)
        else:
            bundle = train_runtime_bundle()
            torch.save(serialize_runtime_bundle(bundle), RUNTIME_CACHE_PATH)
        MODEL_STATE["runtime_bundle"] = bundle
        return bundle


def encode_image_prompts(image_rgb: np.ndarray, prompts: list[str]) -> np.ndarray:
    model, preprocess, tokenizer, device = ensure_model()
    image = preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0).detach().cpu().numpy()
    return logits.astype(np.float32)


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def detect_face_box(frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    candidates: list[tuple[int, int, int, int]] = []
    search_images = [gray, gray_eq, cv2.flip(gray_eq, 1)]

    for cascade in FACE_CASCADES:
        if cascade.empty():
            continue
        for idx, search in enumerate(search_images):
            faces = cascade.detectMultiScale(
                search,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            for x, y, w, h in faces:
                if idx == 2:
                    x = gray.shape[1] - (x + w)
                candidates.append((int(x), int(y), int(w), int(h)))

    if not candidates:
        return None

    def score(box: tuple[int, int, int, int]) -> float:
        x, y, w, h = box
        area = w * h
        center_x = x + w / 2
        center_y = y + h / 2
        center_bias = 1.0 - (abs(center_x - FRAME_SIZE / 2) / FRAME_SIZE + abs(center_y - FRAME_SIZE / 2) / FRAME_SIZE)
        aspect_penalty = abs((w / max(h, 1)) - 0.82)
        return area * (1.0 + center_bias * 0.35) - aspect_penalty * 1000.0

    x, y, w, h = sorted(candidates, key=score, reverse=True)[0]
    return int(x), int(y), int(w), int(h)


def derive_mouth_box(face_box: tuple[int, int, int, int] | None, frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    if face_box is None:
        return None
    fx, fy, fw, fh = face_box
    height, width = frame_rgb.shape[:2]
    x0 = max(int(fx + fw * 0.10), 0)
    y0 = max(int(fy + fh * 0.52), 0)
    x1 = min(int(fx + fw * 0.90), width)
    y1 = min(int(fy + fh * 0.95), height)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = frame_rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    smile_candidates: list[tuple[int, int, int, int]] = []
    if not SMILE_CASCADE.empty():
        detected = SMILE_CASCADE.detectMultiScale(
            gray_eq,
            scaleFactor=1.15,
            minNeighbors=18,
            minSize=(max(16, int((x1 - x0) * 0.18)), max(10, int((y1 - y0) * 0.10))),
        )
        for sx, sy, sw, sh in detected:
            smile_candidates.append((int(sx), int(sy), int(sw), int(sh)))

    if smile_candidates:
        def smile_score(box: tuple[int, int, int, int]) -> float:
            sx, sy, sw, sh = box
            area = sw * sh
            center_bias = 1.0 - abs((sx + sw / 2) - (gray.shape[1] / 2)) / max(gray.shape[1], 1)
            lower_bias = (sy + sh) / max(gray.shape[0], 1)
            aspect = sw / max(sh, 1)
            return area * (0.8 + 0.5 * center_bias + 0.4 * lower_bias) - abs(aspect - 2.4) * 120.0

        sx, sy, sw, sh = sorted(smile_candidates, key=smile_score, reverse=True)[0]
        return (x0 + sx, y0 + sy, sw, sh)

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    sobel_y = np.abs(cv2.Sobel(gray_eq.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    score_map = cr * 0.65 + sobel_y * 0.35
    score_map -= score_map.min()
    score_map /= score_map.max() + 1e-6

    candidate_boxes: list[tuple[int, int, int, int, float]] = []
    roi_h, roi_w = gray.shape[:2]
    for y_frac in (0.38, 0.50, 0.62):
        for x_frac in (0.22, 0.32, 0.42):
            cw = max(18, int(roi_w * 0.42))
            ch = max(12, int(roi_h * 0.18))
            sx = int(roi_w * x_frac)
            sy = int(roi_h * y_frac)
            sx = min(max(sx, 0), max(roi_w - cw, 0))
            sy = min(max(sy, 0), max(roi_h - ch, 0))
            patch = score_map[sy : sy + ch, sx : sx + cw]
            if patch.size == 0:
                continue
            patch_score = float(np.mean(patch))
            center_bias = 1.0 - abs((sx + cw / 2) - (roi_w / 2)) / max(roi_w, 1)
            candidate_boxes.append((sx, sy, cw, ch, patch_score + center_bias * 0.12))

    if not candidate_boxes:
        return None
    sx, sy, sw, sh, best_score = sorted(candidate_boxes, key=lambda item: item[4], reverse=True)[0]
    if best_score < 0.36:
        return None
    return (x0 + sx, y0 + sy, sw, sh)


def crop_box(frame_rgb: np.ndarray, box: tuple[int, int, int, int] | None, size: int = FRAME_SIZE) -> np.ndarray:
    if box is None:
        return cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
    x, y, w, h = box
    crop = frame_rgb[max(y, 0) : max(y, 0) + max(h, 1), max(x, 0) : max(x, 0) + max(w, 1)]
    if crop.size == 0:
        crop = frame_rgb
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def box_area(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    _, _, w, h = box
    return int(w * h)


def detect_subtitle_confidence(frame_rgb: np.ndarray) -> float:
    height, width = frame_rgb.shape[:2]
    band = frame_rgb[int(height * 0.62) :, :]
    if band.size == 0:
        return 0.0

    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.mean(thresh > 0))
    if white_ratio < 0.01 or white_ratio > 0.45:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = 0
    coverage = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 120 or h < 10 or h > band.shape[0] * 0.45:
            continue
        aspect = w / max(h, 1)
        if aspect < 2.0:
            continue
        candidates += 1
        coverage += area / float(band.shape[0] * band.shape[1])

    candidate_score = clamp(candidates / 4.0, 0.0, 1.0)
    coverage_score = clamp(coverage / 0.08, 0.0, 1.0)
    return clamp(0.55 * candidate_score + 0.45 * coverage_score, 0.0, 1.0)


def compute_availability(
    frame_signals: list[FrameSignal],
    audio: np.ndarray,
    companion_text: str,
) -> dict[str, Any]:
    face_ratio = float(np.mean([1.0 if row.face_box is not None else 0.0 for row in frame_signals]))
    mouth_ratio = float(np.mean([1.0 if row.mouth_box is not None else 0.0 for row in frame_signals]))
    subtitle_ratio = float(np.mean([detect_subtitle_confidence(row.frame_rgb) for row in frame_signals]))

    audio_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
    speech_confidence = clamp((audio_rms - 0.008) / 0.055, 0.0, 1.0)
    has_manual_text = len(companion_text.strip().split()) >= 5
    text_confidence = 1.0 if has_manual_text else clamp(subtitle_ratio * 1.35, 0.0, 1.0)

    has_face = face_ratio >= 0.25
    has_lips = has_face and mouth_ratio >= 0.25
    has_speech = speech_confidence >= 0.18
    has_text = has_manual_text or text_confidence >= 0.55

    return {
        "faceRatio": round(face_ratio, 3),
        "mouthRatio": round(mouth_ratio, 3),
        "subtitleRatio": round(subtitle_ratio, 3),
        "speechConfidence": round(speech_confidence, 3),
        "textConfidence": round(text_confidence, 3),
        "hasFace": has_face,
        "hasLips": has_lips,
        "hasSpeech": has_speech,
        "hasText": has_text,
    }


def weighted_available_score(branch_scores: dict[str, float], branch_weights: dict[str, float], availability_weights: dict[str, float]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for key, score in branch_scores.items():
        effective = branch_weights.get(key, 0.0) * availability_weights.get(key, 1.0)
        if effective <= 0.0:
            continue
        weighted_sum += effective * clamp(float(score), 0.0, 1.0)
        weight_sum += effective
    if weight_sum <= 1e-6:
        return clamp(float(np.mean(list(branch_scores.values()))), 0.0, 1.0)
    return clamp(weighted_sum / weight_sum, 0.0, 1.0)


def build_sampling_windows(duration: float) -> list[tuple[str, float, float]]:
    if duration <= 0:
        return [("Start", 0.0, MAX_SECONDS)]
    if duration <= MAX_SECONDS + 0.5:
        return [("Start", 0.0, min(duration, MAX_SECONDS))]

    window_seconds = min(MAX_SECONDS / SAMPLING_WINDOWS, 3.0)
    midpoint_start = max((duration / 2.0) - (window_seconds / 2.0), 0.0)
    end_start = max(duration - window_seconds, 0.0)
    return [
        ("Start", 0.0, min(window_seconds, duration)),
        ("Middle", midpoint_start, min(midpoint_start + window_seconds, duration)),
        ("End", end_start, min(end_start + window_seconds, duration)),
    ]


def extract_sampled_frames(video_path: Path) -> list[FrameSignal]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    sampling_windows = build_sampling_windows(duration if duration > 0 else MAX_SECONDS)
    target_indices: list[int] = []
    if frame_count > 0:
        frames_per_window = max(1, SAMPLE_FRAMES // len(sampling_windows))
        remainder = max(0, SAMPLE_FRAMES - (frames_per_window * len(sampling_windows)))
        for window_index, (_label, start_sec, end_sec) in enumerate(sampling_windows):
            count = frames_per_window + (1 if window_index < remainder else 0)
            start_index = max(int(start_sec * fps), 0)
            end_index = max(int(end_sec * fps) - 1, start_index)
            target_indices.extend(np.linspace(start_index, end_index, count, dtype=int).tolist())
    else:
        target_indices = [0]

    rows: list[FrameSignal] = []
    previous_mouth: np.ndarray | None = None
    for index in target_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        face_box = detect_face_box(frame_rgb)
        mouth_box = derive_mouth_box(face_box, frame_rgb)
        mouth_crop = crop_box(frame_rgb, mouth_box, size=96) if mouth_box is not None else np.zeros((96, 96, 3), dtype=np.uint8)
        motion_score = 0.0
        if previous_mouth is not None and mouth_box is not None:
            motion_score = float(np.mean(np.abs(mouth_crop.astype(np.float32) - previous_mouth.astype(np.float32))) / 255.0)
        previous_mouth = mouth_crop if mouth_box is not None else None
        rows.append(
            FrameSignal(
                timestamp=float(index / fps if fps else 0.0),
                frame_rgb=frame_rgb,
                face_box=face_box,
                mouth_box=mouth_box,
                motion_score=motion_score,
                face_crop=crop_box(frame_rgb, face_box),
            )
        )
    capture.release()
    if not rows:
        raise RuntimeError(f"unable to decode frames from {video_path}")
    return rows


def extract_audio(video_path: Path, wav_path: Path) -> tuple[np.ndarray, int]:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-t",
        f"{MAX_SECONDS:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SR),
        str(wav_path),
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        audio, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:  # noqa: BLE001
        return np.zeros(AUDIO_SR, dtype=np.float32), AUDIO_SR
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def audio_features(audio: np.ndarray, sr: int, target_length: int) -> tuple[np.ndarray, float, float]:
    if audio.size == 0:
        audio = np.zeros(sr, dtype=np.float32)

    frame_length = max(int(sr * 0.08), 256)
    hop_length = max(frame_length // 2, 128)
    rms_values = []
    for start in range(0, max(len(audio) - frame_length + 1, 1), hop_length):
        chunk = audio[start : start + frame_length]
        if chunk.size == 0:
            continue
        rms_values.append(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))))
    envelope = np.asarray(rms_values or [0.0], dtype=np.float32)
    if envelope.max() > 0:
        envelope = envelope / envelope.max()
    if target_length > 0 and envelope.size != target_length:
        x_old = np.linspace(0, 1, envelope.size)
        x_new = np.linspace(0, 1, target_length)
        envelope = np.interp(x_new, x_old, envelope).astype(np.float32)

    device = analysis_device()
    audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(device)
    fft_tensor = torch.abs(torch.fft.rfft(audio_tensor)) + 1e-6
    geometric = float(torch.exp(torch.mean(torch.log(fft_tensor))).detach().cpu().item())
    arithmetic = float(torch.mean(fft_tensor).detach().cpu().item())
    flatness = clamp(geometric / max(arithmetic, 1e-6), 0.0, 1.0)

    fft_length = int(fft_tensor.shape[0])
    midpoint = fft_length // 3
    high_energy = float(torch.mean(fft_tensor[midpoint:]).detach().cpu().item()) if midpoint < fft_length else float(torch.mean(fft_tensor).detach().cpu().item())
    low_energy = float(torch.mean(fft_tensor[:midpoint]).detach().cpu().item()) if midpoint > 0 else float(torch.mean(fft_tensor).detach().cpu().item())
    mel_energy = clamp(high_energy / max(low_energy, 1e-6) / 3.5, 0.0, 1.0)
    return envelope.astype(np.float32), clamp(flatness * 8.0, 0.0, 1.0), clamp(mel_energy, 0.0, 1.0)


def fft_artifact_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    device = analysis_device()
    gray_tensor = torch.from_numpy(gray).to(device)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray_tensor))
    magnitude = torch.log(torch.abs(spectrum) + 1.0)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_band = magnitude[radius > min(h, w) * 0.28]
    low_band = magnitude[radius <= min(h, w) * 0.18]
    if int(low_band.numel()) == 0:
        return 0.5
    ratio = float((torch.mean(high_band) / (torch.mean(low_band) + 1e-6)).detach().cpu().item())
    return clamp((ratio - 0.72) / 0.85, 0.0, 1.0)


def render_frequency_map(image_rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    device = analysis_device()
    gray_tensor = torch.from_numpy(gray).to(device)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray_tensor))
    magnitude = torch.log(torch.abs(spectrum) + 1.0)
    magnitude = magnitude - torch.min(magnitude)
    magnitude = magnitude / (torch.max(magnitude) + 1e-6)
    image_uint8 = np.clip(magnitude.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(image_uint8, cv2.COLORMAP_VIRIDIS)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_crop_preview(image_rgb: np.ndarray, size: int = 160) -> str | None:
    if image_rgb.size == 0:
        return None
    preview = Image.fromarray(image_rgb)
    preview = ImageOps.contain(preview, (size, size))
    canvas = Image.new("RGB", (size, size), (20, 20, 24))
    offset_x = (size - preview.width) // 2
    offset_y = (size - preview.height) // 2
    canvas.paste(preview, (offset_x, offset_y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def motion_audio_sync(motion: np.ndarray, audio_env: np.ndarray) -> tuple[float, int]:
    if motion.size == 0 or audio_env.size == 0:
        return 0.0, 0
    motion = motion - motion.mean()
    audio_env = audio_env - audio_env.mean()
    if motion.std() < 1e-6 or audio_env.std() < 1e-6:
        return 0.0, 0

    best_corr = -1.0
    best_lag = 0
    for lag in range(-4, 5):
        if lag < 0:
            motion_cut = motion[:lag]
            audio_cut = audio_env[-lag:]
        elif lag > 0:
            motion_cut = motion[lag:]
            audio_cut = audio_env[:-lag]
        else:
            motion_cut = motion
            audio_cut = audio_env
        if motion_cut.size < 3 or audio_cut.size < 3:
            continue
        corr = float(np.corrcoef(motion_cut, audio_cut)[0, 1])
        if math.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return clamp((best_corr + 1.0) / 2.0, 0.0, 1.0), best_lag


def clip_real_fake(face_crop: np.ndarray, companion_text: str) -> tuple[float, float, float]:
    prompts = [
        "an authentic real video of a human speaking naturally",
        "a real camera recording of a person on screen",
        "an AI-generated deepfake or synthetic talking head video",
        "a manipulated fake video of a human face",
    ]
    logits = encode_image_prompts(face_crop, prompts)
    real_prob = float((logits[0] + logits[1]) / max(logits.sum(), 1e-6))
    fake_prob = float((logits[2] + logits[3]) / max(logits.sum(), 1e-6))

    text_alignment = 0.5
    if companion_text.strip():
        model, preprocess, tokenizer, device = ensure_model()
        image = preprocess(Image.fromarray(face_crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_tokens = tokenizer([companion_text.strip()]).to(device)
            text_embed = model.encode_text(text_tokens)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            cosine = float((image_features * text_embed).sum().detach().cpu().item())
        text_alignment = clamp((cosine + 1.0) / 2.0, 0.0, 1.0)
    return real_prob, fake_prob, text_alignment


def clip_occlusion_heatmap(frame_rgb: np.ndarray, fake_focus: bool) -> np.ndarray:
    prompts = [
        "an authentic real video of a human speaking naturally",
        "a real camera recording of a person on screen",
        "an AI-generated deepfake or synthetic talking head video",
        "a manipulated fake video of a human face",
    ]
    baseline = encode_image_prompts(frame_rgb, prompts)
    base_real = float((baseline[0] + baseline[1]) / max(baseline.sum(), 1e-6))
    base_fake = float((baseline[2] + baseline[3]) / max(baseline.sum(), 1e-6))
    base_target = base_fake if fake_focus else base_real

    patch = 48
    stride = 24
    height, width = frame_rgb.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)
    visits = np.zeros((height, width), dtype=np.float32)

    tops = list(range(0, max(height - patch + 1, 1), stride)) or [0]
    lefts = list(range(0, max(width - patch + 1, 1), stride)) or [0]

    for top in tops:
        for left in lefts:
            bottom = min(top + patch, height)
            right = min(left + patch, width)
            occluded = frame_rgb.copy()
            patch_region = occluded[top:bottom, left:right]
            blurred = cv2.GaussianBlur(patch_region, (0, 0), 7)
            occluded[top:bottom, left:right] = np.clip(blurred * 0.45 + 70, 0, 255).astype(np.uint8)
            logits = encode_image_prompts(occluded, prompts)
            current_real = float((logits[0] + logits[1]) / max(logits.sum(), 1e-6))
            current_fake = float((logits[2] + logits[3]) / max(logits.sum(), 1e-6))
            current_target = current_fake if fake_focus else current_real
            impact = max(base_target - current_target, 0.0)
            heat[top:bottom, left:right] += impact
            visits[top:bottom, left:right] += 1.0

    visits[visits == 0] = 1.0
    heat = heat / visits
    heat = cv2.GaussianBlur(heat, (0, 0), 8)
    peak = float(heat.max())
    if peak > 1e-6:
        heat = heat / peak
    return heat.astype(np.float32)


def roi_prior_heatmap(face_box: tuple[int, int, int, int] | None, mouth_box: tuple[int, int, int, int] | None) -> np.ndarray:
    prior = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(prior, (x, y), (x + w, y + h), color=0.8, thickness=-1)
    if mouth_box is not None:
        x, y, w, h = mouth_box
        cv2.rectangle(prior, (x, y), (x + w, y + h), color=1.0, thickness=-1)
    if float(prior.max()) > 0:
        prior = cv2.GaussianBlur(prior, (0, 0), 14)
        prior = prior / max(float(prior.max()), 1e-6)
    return prior.astype(np.float32)


def build_timeline(frame_signals: list[FrameSignal], segment_scores: list[float]) -> list[dict[str, Any]]:
    rows = []
    for index, score in enumerate(segment_scores[:4]):
        start_idx = min(index * max(len(frame_signals) // 4, 1), len(frame_signals) - 1)
        end_idx = min(((index + 1) * max(len(frame_signals) // 4, 1)) - 1, len(frame_signals) - 1)
        start = frame_signals[start_idx].timestamp
        end = frame_signals[end_idx].timestamp
        rows.append(
            {
                "label": f"Segment {index + 1}",
                "start": format_mmss(start),
                "end": format_mmss(end),
                "score": round(float(clamp(score, 0.0, 1.0)), 3),
                "note": "가짜 판단에 크게 기여한 의심 구간입니다." if score >= 0.58 else "신호가 비교적 안정적으로 유지된 구간입니다.",
            }
        )
    return rows


def default_regions(face_box: tuple[int, int, int, int] | None, mouth_box: tuple[int, int, int, int] | None) -> list[dict[str, Any]]:
    regions = []
    if face_box is not None:
        x, y, w, h = face_box
        regions.append(
            {
                "id": "face",
                "label": "Face consistency",
                "x": round(x / FRAME_SIZE * 100, 2),
                "y": round(y / FRAME_SIZE * 100, 2),
                "width": round(w / FRAME_SIZE * 100, 2),
                "height": round(h / FRAME_SIZE * 100, 2),
                "score": 0.72,
                "note": "얼굴 구조와 피부 질감이 판정에 크게 반영된 영역입니다.",
            }
        )
    if mouth_box is not None:
        x, y, w, h = mouth_box
        regions.append(
            {
                "id": "mouth",
                "label": "Lip-sync focus",
                "x": round(x / FRAME_SIZE * 100, 2),
                "y": round(y / FRAME_SIZE * 100, 2),
                "width": round(w / FRAME_SIZE * 100, 2),
                "height": round(h / FRAME_SIZE * 100, 2),
                "score": 0.66,
                "note": "입술 움직임과 음성 onset의 정합성이 반영된 영역입니다.",
            }
        )
    regions.append(
        {
            "id": "background",
            "label": "Background context",
            "x": 68.0,
            "y": 16.0,
            "width": 18.0,
            "height": 26.0,
            "score": 0.44,
            "note": "배경 texture와 화면 문맥 신호가 반영된 영역입니다.",
        }
    )
    return regions


def build_regions_from_heatmap(
    heatmap: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    mouth_box: tuple[int, int, int, int] | None,
) -> list[dict[str, Any]]:
    if heatmap.size == 0 or float(heatmap.max()) <= 1e-6:
        return default_regions(face_box, mouth_box)

    smoothed = cv2.GaussianBlur(heatmap, (0, 0), 5)
    working = smoothed.copy()
    regions: list[dict[str, Any]] = []
    radius_px = 22

    for index in range(3):
        _, peak, _, peak_loc = cv2.minMaxLoc(working)
        if peak <= 0.08:
            break

        cx, cy = int(peak_loc[0]), int(peak_loc[1])
        left = max(cx - radius_px, 0)
        top = max(cy - radius_px, 0)
        right = min(cx + radius_px, FRAME_SIZE - 1)
        bottom = min(cy + radius_px, FRAME_SIZE - 1)

        label = "Heat focus"
        note = "최종 판단에 크게 기여한 위치입니다."
        if mouth_box is not None:
            mx, my, mw, mh = mouth_box
            if mx <= cx <= mx + mw and my <= cy <= my + mh:
                label = "Lip-sync focus"
                note = "입술과 음성의 시간 정합성이 집중적으로 반영된 영역입니다."
        if face_box is not None:
            fx, fy, fw, fh = face_box
            if fx <= cx <= fx + fw and fy <= cy <= fy + fh and label == "Heat focus":
                label = "Face consistency"
                note = "얼굴 질감과 구조 안정성이 크게 반영된 영역입니다."
        if cx > FRAME_SIZE * 0.62 and label == "Heat focus":
            label = "Background context"
            note = "배경 texture와 장면 문맥 신호가 반영된 영역입니다."

        regions.append(
            {
                "id": f"heat-{index + 1}",
                "label": label,
                "x": round(left / FRAME_SIZE * 100, 2),
                "y": round(top / FRAME_SIZE * 100, 2),
                "width": round((right - left) / FRAME_SIZE * 100, 2),
                "height": round((bottom - top) / FRAME_SIZE * 100, 2),
                "score": round(float(clamp(peak, 0.0, 1.0)), 3),
                "note": note,
            }
        )
        cv2.circle(working, (cx, cy), radius_px + 8, 0.0, thickness=-1)

    return regions or default_regions(face_box, mouth_box)


def render_focus_frame(frame_rgb: np.ndarray, heatmap: np.ndarray, regions: list[dict[str, Any]]) -> str:
    base = frame_rgb.copy()
    heat_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    strength = np.clip(heatmap[..., None] * 0.78, 0.0, 0.78)
    blended = (base.astype(np.float32) * (1.0 - strength) + heat_color.astype(np.float32) * strength).astype(np.uint8)

    for region in regions:
        x = int(region["x"] / 100.0 * FRAME_SIZE)
        y = int(region["y"] / 100.0 * FRAME_SIZE)
        w = int(region["width"] / 100.0 * FRAME_SIZE)
        h = int(region["height"] / 100.0 * FRAME_SIZE)
        score = float(region.get("score", 0.5))
        color = (
            int(255 * min(1.0, 0.55 + score * 0.45)),
            int(180 * min(1.0, 0.45 + score * 0.55)),
            int(72),
        )
        cv2.rectangle(blended, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            blended,
            str(region["label"]),
            (x + 6, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    image = Image.fromarray(blended)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_metrics(selected_mode: str, fake_score: float, real_score: float, confidence: int) -> list[dict[str, str]]:
    return [
        {"label": "Real score", "value": f"{real_score * 100:.1f}%", "detail": "authenticity confidence"},
        {"label": "Fake score", "value": f"{fake_score * 100:.1f}%", "detail": "synthetic confidence"},
        {"label": "Model", "value": selected_mode.replace("mm-", ""), "detail": "multimodal backend"},
        {"label": "Confidence", "value": f"{confidence}%", "detail": "cross-signal certainty"},
    ]


def build_processing_scope(video_path: Path, frame_signals: list[FrameSignal], availability: dict[str, Any]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    usable_duration = min(duration, MAX_SECONDS) if duration > 0 else MAX_SECONDS
    return {
        "readsWholeVideo": False,
        "fullDurationSec": round(float(duration), 2),
        "analyzedDurationSec": round(float(usable_duration), 2),
        "sampleFrames": len(frame_signals),
        "maxSeconds": MAX_SECONDS,
        "strategy": f"전체 영상을 전부 프레임별로 보지 않고, 처음 {usable_duration:.1f}초 범위에서 {len(frame_signals)}개 대표 프레임과 오디오 구간을 추출해 분석합니다.",
        "precheckSummary": f"얼굴={availability['hasFace']}, 입술={availability['hasLips']}, 음성={availability['hasSpeech']}, 자막/텍스트={availability['hasText']}를 먼저 확인한 뒤 모달별 가중치를 조정합니다.",
        "computeDevice": analysis_device(),
    }


def extract_multi_window_frames(video_path: Path) -> tuple[list[FrameSignal], list[tuple[str, float, float]]]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    windows = build_sampling_windows(duration if duration > 0 else MAX_SECONDS)

    target_indices: list[int] = []
    if frame_count > 0:
        frames_per_window = max(1, SAMPLE_FRAMES // len(windows))
        remainder = max(0, SAMPLE_FRAMES - (frames_per_window * len(windows)))
        for window_index, (_label, start_sec, end_sec) in enumerate(windows):
            count = frames_per_window + (1 if window_index < remainder else 0)
            start_index = max(int(start_sec * fps), 0)
            end_index = max(int(end_sec * fps) - 1, start_index)
            target_indices.extend(np.linspace(start_index, end_index, count, dtype=int).tolist())
    else:
        target_indices = [0]

    rows: list[FrameSignal] = []
    previous_mouth: np.ndarray | None = None
    for index in target_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        face_box = detect_face_box(frame_rgb)
        mouth_box = derive_mouth_box(face_box, frame_rgb)
        mouth_crop = crop_box(frame_rgb, mouth_box, size=96) if mouth_box is not None else np.zeros((96, 96, 3), dtype=np.uint8)
        motion_score = 0.0
        if previous_mouth is not None and mouth_box is not None:
            motion_score = float(np.mean(np.abs(mouth_crop.astype(np.float32) - previous_mouth.astype(np.float32))) / 255.0)
        previous_mouth = mouth_crop if mouth_box is not None else None
        rows.append(
            FrameSignal(
                timestamp=float(index / fps if fps else 0.0),
                frame_rgb=frame_rgb,
                face_box=face_box,
                mouth_box=mouth_box,
                motion_score=motion_score,
                face_crop=crop_box(frame_rgb, face_box),
            )
        )
    capture.release()
    if not rows:
        raise RuntimeError(f"unable to decode frames from {video_path}")
    return rows, windows


def compute_window_segment_scores(
    frame_signals: list[FrameSignal],
    audio_env: np.ndarray,
    frequency_score: float,
) -> list[float]:
    motion_series = np.array([row.motion_score for row in frame_signals], dtype=np.float32)
    segment_count = min(SAMPLING_WINDOWS, max(len(frame_signals), 1))
    frame_chunks = np.array_split(np.arange(len(frame_signals)), segment_count)
    audio_chunks = np.array_split(np.arange(len(audio_env) if len(audio_env) else 1), segment_count)
    scores: list[float] = []
    for index, chunk in enumerate(frame_chunks):
        motion_mean = float(np.mean(motion_series[chunk])) if len(chunk) else 0.2
        audio_chunk = audio_chunks[min(index, len(audio_chunks) - 1)]
        audio_mean = float(np.mean(audio_env[audio_chunk])) if len(audio_env) and len(audio_chunk) else 0.2
        score = clamp(motion_mean * 0.4 + audio_mean * 0.25 + frequency_score * 0.35, 0.0, 1.0)
        scores.append(score)
    return scores


def build_timeline_windows(
    frame_signals: list[FrameSignal],
    segment_scores: list[float],
    windows: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    rows = []
    for index, score in enumerate(segment_scores):
        label = windows[index][0] if index < len(windows) else f"Segment {index + 1}"
        start = windows[index][1] if index < len(windows) else frame_signals[0].timestamp
        end = windows[index][2] if index < len(windows) else frame_signals[-1].timestamp
        rows.append(
            {
                "label": label,
                "start": format_mmss(start),
                "end": format_mmss(end),
                "score": round(float(clamp(score, 0.0, 1.0)), 3),
                "note": "이 구간에서 이상 신호가 상대적으로 더 강하게 관찰됐습니다." if score >= 0.58 else "이 구간은 비교적 안정적으로 관찰됐습니다.",
            }
        )
    return rows


def compute_window_segment_insights_v2(
    frame_signals: list[FrameSignal],
    audio_env: np.ndarray,
    frequency_score: float,
) -> list[SegmentInsight]:
    motion_series = np.array([row.motion_score for row in frame_signals], dtype=np.float32)
    segment_count = min(SAMPLING_WINDOWS, max(len(frame_signals), 1))
    frame_chunks = np.array_split(np.arange(len(frame_signals)), segment_count)
    audio_chunks = np.array_split(np.arange(len(audio_env) if len(audio_env) else 1), segment_count)
    insights: list[SegmentInsight] = []
    for index, chunk in enumerate(frame_chunks):
        motion_mean = float(np.mean(motion_series[chunk])) if len(chunk) else 0.2
        audio_chunk = audio_chunks[min(index, len(audio_chunks) - 1)]
        audio_mean = float(np.mean(audio_env[audio_chunk])) if len(audio_env) and len(audio_chunk) else 0.2
        score = clamp(motion_mean * 0.4 + audio_mean * 0.25 + frequency_score * 0.35, 0.0, 1.0)

        face_hits = []
        mouth_hits = []
        sharpness_scores = []
        for row_index in chunk:
            row = frame_signals[int(row_index)]
            target = row.face_crop if row.face_crop.size else row.frame_rgb
            gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
            face_hits.append(1.0 if row.face_box is not None else 0.0)
            mouth_hits.append(1.0 if row.mouth_box is not None else 0.0)

        start_sec = frame_signals[int(chunk[0])].timestamp if len(chunk) else 0.0
        end_sec = frame_signals[int(chunk[-1])].timestamp if len(chunk) else start_sec
        insights.append(
            SegmentInsight(
                label=f"Segment {index + 1}",
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                score=float(score),
                motion_mean=float(motion_mean),
                audio_mean=float(audio_mean),
                face_ratio=float(np.mean(face_hits)) if face_hits else 0.0,
                mouth_ratio=float(np.mean(mouth_hits)) if mouth_hits else 0.0,
                sharpness_mean=float(np.mean(sharpness_scores)) if sharpness_scores else 0.0,
            )
        )
    return insights


def build_timeline_windows_v2(
    segment_insights: list[SegmentInsight],
    windows: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, insight in enumerate(segment_insights):
        label = windows[index][0] if index < len(windows) else insight.label
        start = windows[index][1] if index < len(windows) else insight.start_sec
        end = windows[index][2] if index < len(windows) else insight.end_sec
        evidence_bits: list[str] = []

        if insight.face_ratio >= 0.67:
            evidence_bits.append(f"얼굴이 {round(insight.face_ratio * 100)}% 프레임에서 안정적으로 감지되었습니다.")
        elif insight.face_ratio > 0:
            evidence_bits.append(f"얼굴 단서는 {round(insight.face_ratio * 100)}% 프레임에서 제한적으로 감지되었습니다.")
        else:
            evidence_bits.append("얼굴 단서는 거의 관찰되지 않았습니다.")

        if insight.mouth_ratio >= 0.5:
            evidence_bits.append(f"입술 단서는 {round(insight.mouth_ratio * 100)}% 프레임에서 유지되었습니다.")
        elif insight.mouth_ratio > 0:
            evidence_bits.append(f"입술 단서는 일부 프레임({round(insight.mouth_ratio * 100)}%)에서만 확인되었습니다.")
        else:
            evidence_bits.append("입술 단서는 거의 검출되지 않았습니다.")

        evidence_bits.append(f"오디오 에너지 평균은 {insight.audio_mean:.3f}, 움직임 점수는 {insight.motion_mean:.3f}입니다.")
        if insight.sharpness_mean > 45:
            evidence_bits.append("프레임 선명도가 비교적 안정적이었습니다.")
        elif insight.sharpness_mean > 20:
            evidence_bits.append("프레임 선명도는 보통 수준이었습니다.")
        else:
            evidence_bits.append("프레임 선명도가 낮아 시각 단서 해석이 제한될 수 있습니다.")

        rows.append(
            {
                "label": label,
                "start": format_mmss(start),
                "end": format_mmss(end),
                "score": round(float(clamp(insight.score, 0.0, 1.0)), 3),
                "note": " ".join(evidence_bits[:2]),
                "evidence": evidence_bits,
            }
        )
    return rows


def build_processing_scope_windows(
    video_path: Path,
    frame_signals: list[FrameSignal],
    availability: dict[str, Any],
    windows: list[tuple[str, float, float]],
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    analyzed_duration = sum(max(end - start, 0.0) for _label, start, end in windows)
    strategy = (
        "영상 전체를 프레임별로 모두 읽지 않고, 시작·중간·끝 3구간에서 대표 프레임과 오디오 단서를 추출해 분석합니다."
        if len(windows) > 1
        else f"영상 초반 {windows[0][2] - windows[0][1]:.1f}초 구간에서 대표 프레임과 오디오 단서를 추출해 분석합니다."
    )
    return {
        "readsWholeVideo": False,
        "fullDurationSec": round(float(duration), 2),
        "analyzedDurationSec": round(float(analyzed_duration), 2),
        "sampleFrames": len(frame_signals),
        "maxSeconds": MAX_SECONDS,
        "strategy": strategy,
        "precheckSummary": f"얼굴={availability['hasFace']}, 입술={availability['hasLips']}, 음성={availability['hasSpeech']}, 자막/텍스트={availability['hasText']}를 먼저 확인한 뒤 모달별 반영 비중을 조정합니다.",
        "computeDevice": analysis_device(),
        "windows": [
            {
                "label": label,
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "startLabel": format_mmss(start),
                "endLabel": format_mmss(end),
            }
            for label, start, end in windows
        ],
    }


def build_stages(selected_mode: str, availability: dict[str, Any]) -> list[dict[str, str]]:
    precheck_parts = []
    precheck_parts.append("얼굴 감지" + (" 확인" if availability["hasFace"] else " 없음"))
    precheck_parts.append("음성" + (" 확인" if availability["hasSpeech"] else " 약함/없음"))
    precheck_parts.append("텍스트" + (" 확인" if availability["hasText"] else " 없음"))

    if selected_mode == "mm-avsync":
        return [
            {"title": "Decode", "body": "비디오와 오디오를 분리하고 시간축을 정렬합니다."},
            {
                "title": "Pre-check",
                "body": " / ".join(precheck_parts) + ". 입술 추적이 가능할 때만 AVSync 비중을 높입니다.",
            },
            {"title": "Lip trace", "body": "입술 ROI가 검출된 경우에만 음성과의 시간차를 비교합니다."},
            {"title": "Rank", "body": "가짜일 가능성이 높은 시간차를 바탕으로 authenticity score를 계산합니다."},
            {"title": "Explain", "body": "근거가 된 구간과 위치를 XAI로 요약합니다."},
        ]
    if selected_mode == "mm-frequency":
        return [
            {"title": "Decode", "body": "프레임과 오디오를 주파수 특징으로 변환합니다."},
            {"title": "Pre-check", "body": " / ".join(precheck_parts) + ". 주파수 분석은 얼굴 유무와 관계없이 수행합니다."},
            {"title": "Spectrum scan", "body": "FFT와 mel 기반 아티팩트 신호를 측정합니다."},
            {"title": "Rank", "body": "고주파 잔여와 audio residue를 함께 반영합니다."},
            {"title": "Explain", "body": "아티팩트가 강한 영역과 구간을 시각화합니다."},
        ]
    return [
        {"title": "Decode", "body": "비디오, 오디오, 텍스트 단서를 분석 가능한 표현으로 변환합니다."},
        {"title": "Pre-check", "body": " / ".join(precheck_parts) + ". 존재하지 않는 모달은 fusion에서 약하게 반영합니다."},
        {"title": "Cross-modal align", "body": "선택한 모델 계열에 맞춰 모달 정합성을 비교합니다."},
        {"title": "Rank", "body": "모달별 신호를 가중 결합해 최종 authenticity score를 계산합니다."},
        {"title": "Explain", "body": "영향력이 큰 영역과 시간 구간을 결과로 정리합니다."},
    ]


def normalize_bins(values: np.ndarray, bins: int) -> list[int]:
    if values.size == 0:
        return [0] * bins
    chunks = np.array_split(values.astype(np.float32), bins)
    reduced = np.array([float(np.mean(chunk)) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    peak = float(np.max(reduced))
    if peak > 1e-6:
        reduced = reduced / peak
    return [int(round(clamp(float(item), 0.0, 1.0) * 100)) for item in reduced]


def build_spectrum_bins(audio: np.ndarray) -> list[int]:
    fft = np.abs(np.fft.rfft(audio.astype(np.float32))) if audio.size else np.zeros(8, dtype=np.float32)
    return normalize_bins(fft, 7)


def build_sync_bins(motion_series: np.ndarray, audio_env: np.ndarray) -> list[int]:
    length = max(len(motion_series), len(audio_env), 8)
    if len(motion_series) != length:
        source = motion_series if len(motion_series) else np.zeros(1, dtype=np.float32)
        motion_series = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(source)), source).astype(np.float32)
    if len(audio_env) != length:
        source = audio_env if len(audio_env) else np.zeros(1, dtype=np.float32)
        audio_env = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(source)), source).astype(np.float32)
    return normalize_bins(np.abs(motion_series - audio_env), 8)


def build_modality_judgments(
    fake_prob: float,
    sync_score: float,
    text_mismatch: float,
    frequency_score: float,
    scenegraph_score: float,
    fake_score: float,
    availability: dict[str, Any],
) -> list[dict[str, Any]]:
    mapping = [
        (
            "Vision",
            fake_prob,
            "얼굴이 검출되면 얼굴 중심 단서로, 없으면 장면 전체 시각 패턴 중심으로 반영합니다." if availability["hasFace"] else "인물이 뚜렷하지 않아 장면 전체 시각 패턴 위주로 반영합니다.",
        ),
        (
            "Audio",
            (1.0 - sync_score) if (availability["hasSpeech"] and availability["hasLips"]) else 0.12,
            "입술과 음성이 모두 관측되면 립싱크 불일치 신호를 반영합니다." if (availability["hasSpeech"] and availability["hasLips"]) else "발화 또는 입술 가용성이 부족해 오디오 분기를 약하게 반영합니다.",
        ),
        (
            "Text",
            text_mismatch if availability["hasText"] else 0.10,
            "자막 또는 보조 텍스트가 확인되어 텍스트 정합성을 반영합니다." if availability["hasText"] else "자막/텍스트가 충분하지 않아 텍스트 분기를 약하게 반영합니다.",
        ),
        (
            "Temporal",
            max((1.0 - sync_score) * 0.72 + frequency_score * 0.28, 0.0) if availability["hasSpeech"] else frequency_score * 0.45,
            "시간축 drift와 프레임 연속성 이상을 함께 반영합니다." if availability["hasSpeech"] else "발화 신호가 약해 프레임 연속성과 변화량 위주로 반영합니다.",
        ),
        ("Frequency", frequency_score, "영상과 오디오의 주파수 잔여 신호를 반영합니다."),
        (
            "Structure",
            scenegraph_score,
            "얼굴과 장면 구조의 안정성을 함께 반영합니다." if availability["hasFace"] else "배경과 장면 구조의 안정성을 중심으로 반영합니다.",
        ),
    ]
    rows = []
    for label, fake_component, reason in mapping:
        fake_component = clamp(float(fake_component), 0.0, 1.0)
        real_component = 1.0 - fake_component
        rows.append(
            {
                "label": label,
                "realPercent": round(real_component * 100, 1),
                "fakePercent": round(fake_component * 100, 1),
                "verdict": "가짜 우세" if fake_component >= real_component else "진짜 우세",
                "reason": reason,
            }
        )
    rows.append(
        {
            "label": "Fusion",
            "realPercent": round((1.0 - fake_score) * 100, 1),
            "fakePercent": round(fake_score * 100, 1),
            "verdict": "가짜 우세" if fake_score >= 0.5 else "진짜 우세",
            "reason": "모든 모달 점수를 결합한 최종 판단입니다.",
        }
    )
    return rows


def predict_service_head(
    bundle: dict[str, Any],
    feature_frame: pd.DataFrame,
    method_name: str,
) -> float:
    feature_names = bundle["head_feature_map"][method_name]
    norm = bundle["head_norms"][method_name]
    x = feature_frame[feature_names].to_numpy(dtype=np.float32)
    x = (x - norm["mean"]) / norm["std"]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    model = bundle["head_models"][method_name]
    model_device = next(model.parameters()).device
    x_t = torch.tensor(x, device=model_device)
    model.eval()
    with torch.no_grad():
        score = float(model(x_t).squeeze(1).detach().cpu().numpy()[0])
    anchor = float(feature_frame["prob_fake_openclip"].iloc[0])
    mixed = 0.65 * score + 0.35 * anchor
    return clamp(mixed, 0.0, 1.0)


def materialize_runtime_heads(
    frame: pd.DataFrame,
    *,
    head_targets: list[str],
    head_models: dict[str, ServiceHeadNet],
    head_norms: dict[str, dict[str, np.ndarray]],
    head_feature_map: dict[str, list[str]],
) -> pd.DataFrame:
    enriched = frame.copy()
    for method_name in head_targets:
        feature_names = head_feature_map[method_name]
        norm = head_norms[method_name]
        x = enriched[feature_names].to_numpy(dtype=np.float32)
        x = (x - norm["mean"]) / norm["std"]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        model = head_models[method_name]
        model_device = next(model.parameters()).device
        x_t = torch.tensor(x, device=model_device)
        model.eval()
        with torch.no_grad():
            raw_scores = model(x_t).squeeze(1).detach().cpu().numpy().astype(np.float32)
        anchors = enriched["prob_fake_openclip"].to_numpy(dtype=np.float32)
        enriched[f"prob_fake_{method_name}"] = np.clip(0.65 * raw_scores + 0.35 * anchors, 0.0, 1.0)
    return enriched


def compute_runtime_weighted_model_score(
    row: pd.Series,
    base_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    numerator = 0.0
    denominator = 0.0
    adjusted_weights: dict[str, float] = {}
    for method_name in METHOD_KEYS:
        prob = float(row[f"prob_fake_{method_name}"])
        reliability = float(row[f"reliability_{method_name}"])
        confidence = clamp(abs(prob - 0.5) * 2.0, 0.08, 1.0)
        adjusted = float(base_weights[method_name] * (0.85 + 0.15 * confidence) * reliability)
        adjusted_weights[method_name] = adjusted
        numerator += adjusted * prob
        denominator += adjusted
    return numerator / max(denominator, 1e-6), adjusted_weights


def predict_runtime_fusion(bundle: dict[str, Any], feature_frame: pd.DataFrame) -> float:
    feature_names = bundle["fusion_feature_names"]
    norm = bundle["fusion_norm"]
    x = feature_frame[feature_names].to_numpy(dtype=np.float32)
    x = (x - norm["mean"]) / norm["std"]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    anchor = feature_frame["simple_avg"].to_numpy(dtype=np.float32)[:, None]
    segment = feature_frame["segment_inv"].to_numpy(dtype=np.float32)[:, None]
    model = bundle["fusion_model"]
    model_device = next(model.parameters()).device
    x_t = torch.tensor(x, device=model_device)
    anchor_t = torch.tensor(anchor, device=model_device)
    segment_t = torch.tensor(segment, device=model_device)
    model.eval()
    with torch.no_grad():
        score = float(model(x_t, anchor_t, segment_t).squeeze(1).detach().cpu().numpy()[0])
    return clamp(score, 0.0, 1.0)


def build_runtime_model_judgments(
    model_scores: dict[str, float],
    selected_mode: str,
    availability: dict[str, Any],
    final_fake_score: float,
) -> list[dict[str, Any]]:
    selected_key = MODE_TO_METHOD.get(selected_mode, "flava")
    reason_map = {
        "openclip": "프레임-텍스트 의미 정합성과 장면 설명 일치도를 기준으로 판단했습니다." if availability["hasText"] else "텍스트 가용성이 낮아 시각 단서를 중심으로 판단했습니다.",
        "flava": "시각·텍스트·오디오 단서를 통합한 멀티모달 표현으로 판단했습니다.",
        "blip_nli": "장면 설명과 자막·STT의 논리적 모순 여부를 중심으로 판단했습니다." if availability["hasText"] else "텍스트가 부족해 설명 분기를 약하게 반영했습니다.",
        "avsync": "입술-음성 시간 정합성으로 판단했습니다." if (availability["hasSpeech"] and availability["hasLips"]) else "입술 또는 음성 가용성이 낮아 AVSync 분기를 약하게 반영했습니다.",
        "frequency": "영상·오디오 주파수 잔여 패턴을 중심으로 판단했습니다.",
        "scenegraph": "얼굴·객체 관계 구조와 장면 안정성을 중심으로 판단했습니다." if availability["hasFace"] else "인물 구조 정보가 약해 배경·장면 구조 단서를 중심으로 판단했습니다.",
    }
    rows = []
    label_map = {
        "openclip": "OpenCLIP",
        "flava": "FLAVA",
        "blip_nli": "BLIP+NLI",
        "avsync": "AVSync",
        "frequency": "Frequency Fusion",
        "scenegraph": "SceneGraph GCN",
    }
    for method_name in METHOD_KEYS:
        fake_component = clamp(float(model_scores[method_name]), 0.0, 1.0)
        real_component = 1.0 - fake_component
        suffix = " 선택된 주력 모델입니다." if method_name == selected_key else ""
        rows.append(
            {
                "label": label_map[method_name],
                "realPercent": round(real_component * 100, 1),
                "fakePercent": round(fake_component * 100, 1),
                "verdict": "가짜 우세" if fake_component >= real_component else "진짜 우세",
                "reason": reason_map[method_name] + suffix,
            }
        )
    rows.append(
        {
            "label": "Proposed Fusion",
            "realPercent": round((1.0 - final_fake_score) * 100, 1),
            "fakePercent": round(final_fake_score * 100, 1),
            "verdict": "가짜 우세" if final_fake_score >= 0.5 else "진짜 우세",
            "reason": "6개 모델 점수와 pre-check, reliability, top-k suspicious segment를 함께 반영한 최종 제안 방식입니다.",
        }
    )
    return rows


def build_actual_fusion_weights(adjusted_weights: dict[str, float]) -> list[dict[str, Any]]:
    label_map = {
        "openclip": "OpenCLIP",
        "flava": "FLAVA",
        "blip_nli": "BLIP+NLI",
        "avsync": "AVSync",
        "frequency": "Frequency",
        "scenegraph": "SceneGraph",
    }
    total = sum(adjusted_weights.values()) or 1.0
    rows = []
    for method_name, weight in sorted(adjusted_weights.items(), key=lambda item: item[1], reverse=True):
        rows.append({"label": label_map[method_name], "weight": round((weight / total) * 100, 1)})
    return rows


def build_single_mode_weights(selected_mode: str) -> list[dict[str, Any]]:
    label_map = {
        "mm-openclip": "OpenCLIP",
        "mm-flava": "FLAVA",
        "mm-blip-nli": "BLIP+NLI",
        "mm-avsync": "AVSync",
        "mm-frequency": "Frequency",
        "mm-scenegraph": "SceneGraph",
    }
    selected_label = label_map.get(selected_mode, "FLAVA")
    return [{"label": selected_label, "weight": 100.0}]


def build_actual_fusion_steps(
    selected_mode: str,
    selected_score: float,
    weighted_model_score: float,
    segment_score: float,
    final_score: float,
    adjusted_weights: dict[str, float],
    availability: dict[str, Any],
) -> list[dict[str, str]]:
    selected_key = MODE_TO_METHOD.get(selected_mode, "flava")
    label_map = {
        "openclip": "OpenCLIP",
        "flava": "FLAVA",
        "blip_nli": "BLIP+NLI",
        "avsync": "AVSync",
        "frequency": "Frequency Fusion",
        "scenegraph": "SceneGraph GCN",
    }
    dominant = max(adjusted_weights.items(), key=lambda item: item[1])[0]
    precheck_text = f"얼굴={availability['hasFace']}, 입술={availability['hasLips']}, 음성={availability['hasSpeech']}, 텍스트={availability['hasText']}"
    return [
        {
            "title": "사전 탐지",
            "weight": "Stage 1",
            "logic": f"{precheck_text}를 먼저 판단하고, 가용성이 낮은 분기는 gate down 했습니다.",
        },
        {
            "title": f"{label_map[selected_key]} 실제 추론",
            "weight": f"Fake {selected_score * 100:.1f}%",
            "logic": "선택한 모델의 실험 학습 경로를 서비스 런타임에 연결해 점수를 산출했습니다.",
        },
        {
            "title": "Adaptive weighting",
            "weight": label_map[dominant],
            "logic": f"{label_map[dominant]} 분기가 가장 크게 반영됐고, 가중 결합 결과는 {weighted_model_score * 100:.1f}%였습니다.",
        },
        {
            "title": "Top-k suspicious segments",
            "weight": f"{segment_score * 100:.1f}%",
            "logic": "시작·중간·끝 구간 중 의심 신호가 큰 구간을 강조해 segment suspiciousness를 계산했습니다.",
        },
        {
            "title": "Proposed fusion",
            "weight": f"Fake {final_score * 100:.1f}%",
            "logic": "6개 모델 점수와 pre-check, reliability, segment aggregation을 함께 반영한 최종 결과입니다.",
        },
    ]


def build_single_mode_steps(
    selected_mode: str,
    selected_score: float,
    availability: dict[str, Any],
) -> list[dict[str, str]]:
    label_map = {
        "mm-openclip": "OpenCLIP",
        "mm-flava": "FLAVA",
        "mm-blip-nli": "BLIP+NLI",
        "mm-avsync": "AVSync",
        "mm-frequency": "Frequency Fusion",
        "mm-scenegraph": "SceneGraph GCN",
    }
    selected_label = label_map.get(selected_mode, "FLAVA")
    precheck_text = f"얼굴={availability['hasFace']}, 입술={availability['hasLips']}, 음성={availability['hasSpeech']}, 텍스트={availability['hasText']}"
    return [
        {
            "title": "사전 탐지",
            "weight": "Stage 1",
            "logic": f"{precheck_text}를 먼저 확인하고, 가용성이 낮은 단서는 해당 모델 설명에만 제한적으로 반영했습니다.",
        },
        {
            "title": f"{selected_label} 단독 추론",
            "weight": f"Fake {selected_score * 100:.1f}%",
            "logic": "선택한 모델 1개만 사용해 최종 판정을 계산했습니다.",
        },
        {
            "title": "최종 결과",
            "weight": f"Fake {selected_score * 100:.1f}%",
            "logic": "융합 없이 선택 모델의 실제 점수를 그대로 최종 결과에 사용했습니다.",
        },
    ]


def build_fusion_weights(selected_mode: str) -> list[dict[str, Any]]:
    presets: dict[str, list[tuple[str, float]]] = {
        "mm-openclip": [("Vision", 0.34), ("Text", 0.24), ("Temporal", 0.14), ("Audio", 0.08), ("Frequency", 0.10), ("Structure", 0.10)],
        "mm-flava": [("Vision", 0.24), ("Text", 0.18), ("Temporal", 0.17), ("Audio", 0.13), ("Frequency", 0.14), ("Structure", 0.14)],
        "mm-blip-nli": [("Vision", 0.18), ("Text", 0.31), ("Temporal", 0.10), ("Audio", 0.08), ("Frequency", 0.12), ("Structure", 0.21)],
        "mm-avsync": [("Vision", 0.15), ("Text", 0.06), ("Temporal", 0.26), ("Audio", 0.29), ("Frequency", 0.12), ("Structure", 0.12)],
        "mm-frequency": [("Vision", 0.17), ("Text", 0.07), ("Temporal", 0.12), ("Audio", 0.15), ("Frequency", 0.37), ("Structure", 0.12)],
        "mm-scenegraph": [("Vision", 0.18), ("Text", 0.12), ("Temporal", 0.10), ("Audio", 0.08), ("Frequency", 0.10), ("Structure", 0.42)],
    }
    weights = presets.get(selected_mode, presets["mm-flava"])
    return [{"label": label, "weight": weight} for label, weight in weights]


def build_fusion_steps(
    selected_mode: str,
    fake_score: float,
    sync_score: float,
    frequency_score: float,
    text_alignment: float,
    scenegraph_score: float,
    availability: dict[str, Any],
) -> list[dict[str, str]]:
    weights = build_fusion_weights(selected_mode)
    dominant = max(weights, key=lambda item: item["weight"])
    gated = []
    if not availability["hasFace"]:
        gated.append("face-aware")
    if not (availability["hasSpeech"] and availability["hasLips"]):
        gated.append("lip-sync")
    if not availability["hasText"]:
        gated.append("text")
    gated_text = ", ".join(gated) if gated else "없음"
    return [
        {
            "title": "사전 탐지",
            "weight": "Stage 1",
            "logic": f"얼굴={availability['hasFace']}, 음성={availability['hasSpeech']}, 입술={availability['hasLips']}, 자막/텍스트={availability['hasText']}를 먼저 판정했습니다.",
        },
        {
            "title": "모달 점수 정리",
            "weight": "Stage 1",
            "logic": "Vision, Audio, Text, Temporal, Frequency, Structure 분기에서 개별 real/fake 점수를 계산했습니다.",
        },
        {
            "title": "주력 분기 반영",
            "weight": f"{dominant['label']} {dominant['weight'] * 100:.0f}%",
            "logic": f"선택한 {selected_mode.replace('mm-', '').upper()} 경로에서 {dominant['label']} 신호 비중이 가장 크게 반영됐습니다.",
        },
        {
            "title": "가용성 기반 게이트",
            "weight": "Cross-check",
            "logic": f"현재 샘플에서 약하게 본 분기: {gated_text}. 립싱크 {sync_score:.2f}, 주파수 {frequency_score:.2f}, 텍스트 정합성 {text_alignment:.2f}, 구조 안정성 {1.0 - scenegraph_score:.2f}를 함께 대조했습니다.",
        },
        {
            "title": "최종 통합",
            "weight": f"Fake {fake_score * 100:.1f}%",
            "logic": "가중 결합 결과를 바탕으로 최종 진위 비율과 설명 가능한 근거를 정리했습니다.",
        },
    ]


def build_model_traits(selected_mode: str, availability: dict[str, Any]) -> list[dict[str, str]]:
    selected_title = selected_mode.replace("mm-", "").upper()
    traits = [
        ("OpenCLIP", "시각-텍스트 정합성", "프레임과 설명 텍스트 사이의 의미적 대응을 봅니다.", "장면-설명 간 불일치 단서를 반영합니다."),
        ("FLAVA", "멀티모달 통합", "시각·언어 결합 표현으로 전체 정합성을 평가합니다.", "최종 통합 안정성을 보조합니다."),
        ("BLIP+NLI", "설명 기반 검증", "장면 설명 생성 후 자막/STT와의 모순 여부를 봅니다.", "설명 가능한 텍스트 근거를 제공합니다."),
        ("AVSync", "오디오-립싱크", "입술 움직임과 음성 onset의 시간차를 봅니다.", "립싱크 drift 신호를 반영합니다."),
        ("Frequency Fusion", "주파수 포렌식", "고주파 잔여와 mel residue를 분석합니다.", "생성형 잔여 패턴을 반영합니다."),
        ("SceneGraph GCN", "구조 관계", "객체·얼굴 관계 구조의 안정성을 비교합니다.", "장면 구조 이상 신호를 반영합니다."),
    ]
    rows = []
    for model, role, trait, contribution in traits:
        if model.upper() == selected_title:
            contribution = f"선택한 주력 모델로서 {contribution}"
        if model == "AVSync" and not (availability["hasSpeech"] and availability["hasLips"]):
            contribution = "현재 샘플에서 입술 또는 음성 가용성이 낮아 보조적으로만 반영합니다."
        if model in {"OpenCLIP", "BLIP+NLI"} and not availability["hasText"]:
            contribution = "현재 샘플에서 텍스트 가용성이 낮아 화면 쪽 신호를 중심으로 반영합니다."
        rows.append({"model": model, "role": role, "trait": trait, "contribution": contribution})
    return rows


def build_text_highlights(companion_text: str, text_alignment: float, sync_score: float, frequency_score: float, availability: dict[str, Any]) -> list[dict[str, Any]]:
    if companion_text.strip():
        words = companion_text.strip().split()
        selected = words[:4] if len(words) >= 4 else words
        highlights = []
        for index, word in enumerate(selected):
            weight = [text_alignment, 1.0 - sync_score, frequency_score, 0.5][index] if index < 4 else text_alignment
            tag = ["consistency", "sync", "artifact", "context"][index] if index < 4 else "context"
            highlights.append({"text": word[:18], "weight": round(float(clamp(weight, 0.0, 1.0)), 3), "tag": tag})
        return highlights
    if not availability["hasText"]:
        return [{"text": "text branch gated", "weight": round(float(1.0 - availability["textConfidence"]), 3), "tag": "gated"}]
    return [
        {"text": "scene-text alignment", "weight": round(float(clamp(text_alignment, 0.0, 1.0)), 3), "tag": "consistency"},
        {"text": "lip-audio drift", "weight": round(float(clamp(1.0 - sync_score, 0.0, 1.0)), 3), "tag": "sync"},
        {"text": "frequency residue", "weight": round(float(clamp(frequency_score, 0.0, 1.0)), 3), "tag": "artifact"},
    ]


def fallback_llm_summary(
    selected_mode: str,
    fake_score: float,
    sync_score: float,
    frequency_score: float,
    text_alignment: float,
    availability: dict[str, Any],
) -> tuple[str, list[dict[str, str]], str]:
    verdict = "가짜 가능성 우세" if fake_score >= 0.5 else "진짜 가능성 우세"
    active_modal_parts = ["시각"]
    if availability["hasSpeech"]:
        active_modal_parts.append("오디오")
    if availability["hasText"]:
        active_modal_parts.append("텍스트")
    summary = (
        f"{selected_mode.replace('mm-', '').upper()} 경로에서 {'·'.join(active_modal_parts)} 신호를 종합한 결과 {verdict}입니다. 주파수 잔여 {frequency_score:.2f}"
        + (f", 립싱크 일치도 {sync_score:.2f}" if availability["hasSpeech"] and availability["hasLips"] else "")
        + (f", 텍스트 정합성 {text_alignment:.2f}" if availability["hasText"] else "")
        + "가 주요 근거로 반영됐습니다."
    )
    evidence_body = []
    evidence_body.append("배경과 장면 문맥")
    if availability["hasFace"]:
        evidence_body.insert(0, "얼굴 구조")
    if availability["hasSpeech"] and availability["hasLips"]:
        evidence_body.append("입술-음성 정합성")
    if availability["hasText"]:
        evidence_body.append("텍스트 정합성")
    reasoning = [
        {
            "title": "멀티모달 종합 판단",
            "body": "여러 모달의 점수를 따로 본 뒤 서로 같은 방향을 가리키는지 확인해 최종 결론을 냈습니다.",
        },
        {
            "title": "주요 근거",
            "body": "이 샘플에서는 " + "·".join(evidence_body) + " 신호를 중심으로 반영했습니다.",
        },
        {
            "title": "사전 탐지 반영",
            "body": "얼굴, 입술, 음성, 텍스트 존재 여부를 먼저 확인하고, 없는 모달은 최종 통합에서 약하게 처리했습니다.",
        },
    ]
    headline = "사전 탐지와 모달 가용성을 반영해 실제로 존재하는 단서를 중심으로 XAI를 구성했습니다."
    return summary, reasoning, headline


def build_verdict_summary(fake_score: float, real_score: float) -> str:
    if fake_score >= 0.7:
        return "합성 콘텐츠일 가능성이 높게 평가되었습니다."
    if fake_score >= 0.5:
        return "합성 콘텐츠 가능성이 상대적으로 높게 관측되었습니다."
    if real_score >= 0.7:
        return "진본 콘텐츠일 가능성이 높게 평가되었습니다."
    return "진본 콘텐츠 가능성이 상대적으로 높게 관측되었습니다."


def build_deterministic_reasoning(
    fake_score: float,
    real_score: float,
    confidence: int,
    availability: dict[str, Any],
    weighted_model_score: float,
    segment_score: float,
    processing_scope: dict[str, Any],
) -> tuple[list[dict[str, str]], str]:
    gated = []
    if not availability["hasFace"]:
        gated.append("얼굴")
    if not availability["hasLips"]:
        gated.append("입술")
    if not availability["hasSpeech"]:
        gated.append("음성")
    if not availability["hasText"]:
        gated.append("텍스트")
    gated_text = ", ".join(gated) if gated else "없음"

    reasoning = [
        {
            "title": "최종 판단",
            "body": f"최종 가짜 확률은 {fake_score * 100:.1f}%, 진짜 확률은 {real_score * 100:.1f}%이며, 현재 판정 신뢰도는 {confidence}%입니다.",
        },
        {
            "title": "분석 범위",
            "body": f"원본 전체 길이 {processing_scope['fullDurationSec']:.1f}초 중 {processing_scope['analyzedDurationSec']:.1f}초를 표본으로 읽었고, 대표 프레임 {processing_scope['sampleFrames']}개를 사용했습니다.",
        },
        {
            "title": "게이트 및 융합",
            "body": f"사전 탐지 결과 얼굴={availability['hasFace']}, 입술={availability['hasLips']}, 음성={availability['hasSpeech']}, 텍스트={availability['hasText']}였고, gated down 분기는 {gated_text}입니다. 가중 결합 점수는 {weighted_model_score * 100:.1f}%, 구간 점수는 {segment_score * 100:.1f}%였습니다.",
        },
    ]
    headline = "실제 관측된 단서만 사용해 최종 판정을 구성했습니다."
    return reasoning, headline


def build_frequency_reference_bins(sample_bins: list[int]) -> dict[str, Any]:
    normalized_sample = (sample_bins + [0] * 7)[:7]
    return {
        "realReference": [72, 66, 58, 49, 39, 30, 22],
        "fakeReference": [24, 31, 48, 69, 86, 76, 58],
        "sample": normalized_sample,
        "note": "아래 두 줄은 비교용 기준 패턴이며, 마지막 줄이 현재 영상의 주파수 분포입니다.",
    }


def call_openai_explanation(payload: dict[str, Any]) -> tuple[str, list[dict[str, str]], str] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    body = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are explaining multimodal authenticity analysis results for a Korean XAI dashboard. "
                            "Respond as strict JSON with keys summary, headline, reasoning. "
                            "summary: one Korean paragraph under 120 chars. "
                            "headline: one Korean sentence under 80 chars. "
                            "reasoning: array of exactly 3 objects with title and body in Korean. "
                            "Only mention modalities that are available in the payload. "
                            "If face, lips, speech, or text are not available, explicitly say those branches were gated down and do not claim they were observed."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }

    request = urlrequest.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(request, timeout=45) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError):
        return None

    text_chunks: list[str] = []
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_chunks.append(content["text"])
    if not text_chunks:
        return None

    try:
        parsed = json.loads("".join(text_chunks))
        summary = str(parsed["summary"]).strip()
        headline = str(parsed["headline"]).strip()
        reasoning = [
            {"title": str(item["title"]).strip(), "body": str(item["body"]).strip()}
            for item in parsed["reasoning"][:3]
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not summary or not headline or len(reasoning) != 3:
        return None
    return summary, reasoning, headline


def call_openai_xai_sections(payload: dict[str, Any]) -> dict[str, str] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    body = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You generate short Korean explanations for a multimodal XAI dashboard. "
                            "Return strict JSON with keys heatmap, timeline, fusion, frequency. "
                            "Each value must be easy to read, factual, and under 220 Korean characters. "
                            "Use only the facts in the payload. "
                            "Do not claim a modality was observed if the payload says it was gated down."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }

    request = urlrequest.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(request, timeout=45) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError):
        return None

    text_chunks: list[str] = []
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_chunks.append(content["text"])
    if not text_chunks:
        return None

    try:
        parsed = json.loads("".join(text_chunks))
        sections = {
            "heatmap": str(parsed["heatmap"]).strip(),
            "timeline": str(parsed["timeline"]).strip(),
            "fusion": str(parsed["fusion"]).strip(),
            "frequency": str(parsed["frequency"]).strip(),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not all(sections.values()):
        return None
    return sections


def fallback_xai_sections(payload: dict[str, Any]) -> dict[str, str]:
    regions = payload.get("regions") or []
    timeline = payload.get("timeline") or []
    weights = payload.get("fusionWeights") or []
    gated = payload.get("gatedBranches") or []
    top_region = regions[0]["label"] if regions else "핵심 시각 영역"
    top_region_note = regions[0]["note"] if regions else "모델이 가장 크게 반영한 시각 단서입니다."
    top_timeline = timeline[0]["label"] if timeline else "대표 구간"
    top_timeline_note = timeline[0]["note"] if timeline else "다른 구간보다 설명 신호가 상대적으로 높게 표시되었습니다."
    dominant = weights[0]["label"] if weights else "주요 분기"
    return {
        "heatmap": f"{top_region}에서 설명 신호가 가장 두드러집니다. {top_region_note}",
        "timeline": f"{top_timeline} 구간이 대표 설명 구간으로 표시되었습니다. {top_timeline_note}",
        "fusion": f"{dominant} 분기가 최종 설명에 크게 반영되었습니다. {', '.join(gated) if gated else '비활성화된 주요 분기는 없습니다.'}",
        "frequency": "현재 입력의 주파수 분포를 real/fake 기준 패턴과 비교해 표시합니다. 이 값은 단독 판정 근거가 아니라 보조 설명 신호입니다.",
    }


def fallback_text_xai_sections(payload: dict[str, Any]) -> dict[str, str]:
    analysis = payload.get("analysis") or {}
    timeline = (analysis.get("xai") or {}).get("timeline") or analysis.get("timeline") or []
    tokens = (analysis.get("xai") or {}).get("textHighlights") or analysis.get("tokens") or []
    top_tokens = [str(item.get("text")) for item in tokens if item.get("tag") != "context"][:4]
    top_span = timeline[0].get("label") if timeline else "대표 문장"
    token_text = ", ".join(top_tokens) if top_tokens else "뚜렷한 표현 단서"
    return {
        "userGuide": "큰 노드는 설명 신호가 상대적으로 강한 표현입니다. 두꺼운 선은 함께 읽을 표현 묶음을 의미합니다. 이 시각화는 판정 원인을 단어 하나로 단정하지 않고, 모델 결과를 이해하기 위한 보조 설명으로 사용합니다.",
        "sentenceInterpretation": f"{top_span}에서 {token_text} 표현이 설명 신호로 표시되었습니다. 이 표시는 판정 확률이 아니라 전체 설명에서 눈에 띄는 정도를 나타냅니다.",
        "tip": "반복 표현, 지나치게 균일한 문장 구조, 검증 출처 표현 부족은 Fake 쪽 보조 설명 신호가 될 수 있습니다. 구체적인 맥락, 확인 가능한 출처, 자연스러운 문장 리듬을 보강하면 오탐 가능성을 줄일 수 있습니다.",
    }


def call_openai_text_xai_sections(payload: dict[str, Any]) -> dict[str, str] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    body = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You write Korean text-XAI explanations for an AI text detection dashboard. "
                            "Return strict JSON with keys userGuide, sentenceInterpretation, tip. "
                            "Use only the payload facts. Keep each value under 260 Korean characters. "
                            "Do not describe highlighted words as direct proof of AI generation. "
                            "Use a clean official-document tone."
                        ),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]},
        ],
        "text": {"format": {"type": "json_object"}},
    }
    request = urlrequest.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=45) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError):
        return None
    chunks: list[str] = []
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    if not chunks:
        return None
    try:
        parsed = json.loads("".join(chunks))
        sections = {
            "userGuide": str(parsed["userGuide"]).strip(),
            "sentenceInterpretation": str(parsed["sentenceInterpretation"]).strip(),
            "tip": str(parsed["tip"]).strip(),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return sections if all(sections.values()) else None


def detect_text_language(text: str) -> str:
    hangul_count = len(re.findall(r"[\uac00-\ud7a3]", text))
    ascii_word_count = len(re.findall(r"\b[a-zA-Z]{2,}\b", text))
    return "ko" if hangul_count >= max(3, ascii_word_count // 2) else "en"


def load_python_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    module_dir = str(module_path.parent)
    added_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_path = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
    return module


def ensure_text_detector(language: str) -> tuple[Any | None, str]:
    with TEXT_MODEL_LOCK:
        if language in TEXT_MODEL_STATE:
            state = TEXT_MODEL_STATE[language]
            return state.get("detector"), state.get("status", "unknown")

        try:
            if language == "ko":
                ko_dir = TEXT_MODEL_BUNDLE_DIR / "LOG_AID_ko"
                module = load_python_module("isy_text_detector_ko", ko_dir / "text_detector_ko.py")
                detector = module.TextDetectorKO(
                    classifier_dir=str(ko_dir),
                    quantization=os.environ.get("ISY_KO_TEXT_QUANTIZATION", "4bit"),
                )
                TEXT_MODEL_STATE[language] = {"detector": detector, "status": "loaded: LOG-AID ko"}
            else:
                en_dir = TEXT_MODEL_BUNDLE_DIR / "DeBERTa_En"
                module = load_python_module("isy_text_detector_en", en_dir / "text_detector_en.py")
                detector = module.TextDetector(model_dir=str(en_dir))
                TEXT_MODEL_STATE[language] = {"detector": detector, "status": "loaded: DeBERTa English"}
        except Exception as error:  # noqa: BLE001
            TEXT_MODEL_STATE[language] = {
                "detector": None,
                "status": f"bundle connected, runtime fallback: {type(error).__name__}",
            }
        state = TEXT_MODEL_STATE[language]
        return state.get("detector"), state.get("status", "unknown")


def split_text_units(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if part.strip()]
    if len(parts) >= 2:
        return parts[:8]
    words = re.findall(r"[\uac00-\ud7a3A-Za-z0-9%$#@'_-]+", text)
    if not words:
        return [text.strip()] if text.strip() else []
    chunk_size = max(8, min(28, math.ceil(len(words) / 4)))
    return [" ".join(words[index : index + chunk_size]) for index in range(0, len(words), chunk_size)][:8]


def text_heuristic_features(text: str, language: str) -> dict[str, Any]:
    words = re.findall(r"[\uac00-\ud7a3A-Za-z0-9%$#@'_-]+", text.lower())
    sentences = split_text_units(text)
    total_words = max(len(words), 1)
    unique_ratio = len(set(words)) / total_words
    avg_sentence_len = total_words / max(len(sentences), 1)
    repeated_ratio = 1.0 - unique_ratio
    punctuation_count = len(re.findall(r"[!?]{2,}|\.{3,}|[※★◆]", text))
    citation_count = len(re.findall(r"https?://|\[[0-9]+\]|\([0-9]{4}\)|\baccording to\b|출처|자료|근거", text, flags=re.I))
    hedge_count = len(re.findall(r"\bmay\b|\bmight\b|\blikely\b|\btherefore\b|\bfurthermore\b|가능성|따라서|결론적으로|종합하면", text, flags=re.I))
    burst_count = len([word for word in words if len(word) >= (12 if language == "en" else 8)])

    ai_score = 0.42
    ai_score += clamp(repeated_ratio * 0.65, 0.0, 0.22)
    ai_score += 0.12 if avg_sentence_len > 28 else 0.0
    ai_score += 0.08 if hedge_count >= 3 else 0.0
    ai_score += 0.06 if burst_count >= max(2, total_words // 18) else 0.0
    ai_score -= clamp(citation_count * 0.035, 0.0, 0.16)
    ai_score -= 0.07 if punctuation_count >= 2 else 0.0
    ai_score = clamp(ai_score, 0.04, 0.96)

    return {
        "words": words,
        "sentences": sentences,
        "uniqueRatio": unique_ratio,
        "repeatedRatio": repeated_ratio,
        "avgSentenceLen": avg_sentence_len,
        "citationCount": citation_count,
        "hedgeCount": hedge_count,
        "punctuationCount": punctuation_count,
        "aiScore": ai_score,
    }


def build_text_highlight_tokens(features: dict[str, Any], ai_score: float) -> list[dict[str, Any]]:
    words = features["words"]
    if not words:
        return [{"text": "empty input", "weight": 0.25, "tag": "input"}]
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    common_context = {
        "안녕하세요", "반가워요", "만나서", "오늘", "저는", "우리", "그리고", "하지만", "그래서", "합니다", "했습니다",
        "hello", "hi", "thanks", "thank", "today", "because", "therefore", "however", "about", "with",
    }
    selected: list[dict[str, Any]] = []
    context_fallback: list[dict[str, Any]] = []
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        repetition = counts[word] / max(max(counts.values()), 1)
        repeated = counts[word] > 1
        long_token = len(word) >= (12 if re.search(r"[A-Za-z]", word) else 7)
        hedge_like = bool(re.search(r"전반적|일반적|효율적|중요|핵심|다양|clearly|overall|various|important|significant", word, flags=re.I))
        grounding = bool(re.search(r"https?|출처|근거|according|study|report|doi|논문", word, flags=re.I))
        tag = "repeat" if repeated else "long" if long_token else "style" if hedge_like else "grounding" if grounding else "context"
        if grounding:
            tag = "grounding"
        evidence_weight = 0.18
        evidence_weight += 0.34 if repeated else 0.0
        evidence_weight += 0.24 if long_token else 0.0
        evidence_weight += 0.18 if hedge_like else 0.0
        evidence_weight += 0.16 if grounding else 0.0
        evidence_weight += clamp((repetition - 0.2) * 0.24, 0.0, 0.16)
        weight = clamp(evidence_weight, 0.16, 0.92)
        item = {"text": word[:22], "weight": round(weight, 3), "tag": tag}
        if tag == "context" or word in common_context:
            context_fallback.append({**item, "weight": min(item["weight"], 0.28), "tag": "context"})
            continue
        selected.append(item)
        if len(selected) >= 18:
            break
    if not selected:
        selected = context_fallback[: min(8, len(context_fallback))]
    else:
        selected.extend(context_fallback[: max(0, 18 - len(selected))])
    return selected[:18]


def build_text_analysis(text: str, selected_mode: str) -> dict[str, Any]:
    normalized_text = text.strip()
    language = detect_text_language(normalized_text)
    detector, runtime_status = ensure_text_detector(language)
    features = text_heuristic_features(normalized_text, language)
    model_result: dict[str, Any] | None = None
    model_source = "한국어 LOG-AID" if language == "ko" else "English DeBERTa"
    language_label = "한국어" if language == "ko" else "영어"

    if detector is not None:
        try:
            model_result = detector.predict(normalized_text)
        except Exception as error:  # noqa: BLE001
            runtime_status = f"bundle connected, predict fallback: {type(error).__name__}"

    if model_result and isinstance(model_result.get("ai_probability"), (int, float)):
        fake_percent = float(model_result["ai_probability"])
        primary_reason = str(model_result.get("reason") or "연결된 text 모델의 출력 확률을 기준으로 판단했습니다.")
    else:
        fake_percent = float(features["aiScore"] * 100.0)
        primary_reason = "모델 번들은 연결되어 있으나 현재 런타임에서 base model/dependency 로드가 완결되지 않아, 문체·반복·근거 신호 기반 fallback XAI로 표시했습니다."

    if selected_mode == "text-fact-check":
        grounding_penalty = clamp(features["citationCount"] * 6.0, 0.0, 24.0)
        unsupported_claim_boost = 12.0 if features["citationCount"] == 0 and len(features["sentences"]) >= 2 else 0.0
        fake_percent = clamp(fake_percent - grounding_penalty + unsupported_claim_boost, 4.0, 96.0)

    real_percent = round(100.0 - fake_percent, 1)
    fake_percent = round(fake_percent, 1)
    is_fake = fake_percent >= real_percent
    confidence = int(round(clamp(62 + abs(fake_percent - 50) * 0.76, 60, 97)))
    verdict_label = "AI 생성 가능성 우세" if is_fake else "사람 작성 가능성 우세"
    decision_phrase = "AI 생성/합성" if is_fake else "사람 작성/진본"
    summary = (
        f"{model_source} text 모델은 {decision_phrase} 쪽으로 판정했습니다. "
        f"Real {real_percent:.1f}%, Fake {fake_percent:.1f}%입니다."
    )
    if model_result and isinstance(model_result.get("ai_probability"), (int, float)):
        primary_reason = (
            f"모델 출력 확률 기준으로 {decision_phrase} 쪽이 우세합니다. "
            f"세부 신호: {primary_reason}"
        )
    tokens = build_text_highlight_tokens(features, fake_percent / 100.0)
    sentence_units = features["sentences"] or [normalized_text[:80] or "empty input"]
    timeline = []
    for index, sentence in enumerate(sentence_units[:4]):
        sentence_words = re.findall(r"[\uac00-\ud7a3A-Za-z0-9%$#@'_-]+", sentence.lower())
        repeated_hits = sum(1 for word in sentence_words if features["words"].count(word) > 1)
        local_score = clamp((fake_percent / 100.0) * 0.62 + repeated_hits * 0.05 + (0.12 if len(sentence_words) > 24 else 0.0), 0.08, 0.97)
        timeline.append(
            {
                "label": f"문장 span {index + 1}",
                "start": f"{index + 1:02d}",
                "end": f"{index + 2:02d}",
                "score": round(local_score, 3),
                "note": sentence[:120],
                "evidence": [
                    f"반복 표현 {repeated_hits}개",
                    f"문장 길이 {len(sentence)}자",
                    "검증 출처 표현 있음" if re.search(r"https?://|출처|근거|according|report", sentence, flags=re.I) else "검증 출처 표현 없음",
                ],
            }
        )

    modality_bars = [
        {"label": model_source, "score": round(fake_percent / 100.0, 3), "note": runtime_status},
        {"label": "반복 표현", "score": round(clamp(features["repeatedRatio"], 0.0, 1.0), 3), "note": "같은 표현이 반복될수록 생성형 문체 가능성을 높게 봅니다."},
        {"label": "문장 규칙성", "score": round(clamp(features["avgSentenceLen"] / 36.0, 0.0, 1.0), 3), "note": "문장 길이가 지나치게 균일하거나 길면 템플릿 신호로 봅니다."},
        {"label": "검증 출처 공백", "score": round(1.0 - clamp(features["citationCount"] / 3.0, 0.0, 1.0), 3), "note": "출처·근거·보고서·링크 같은 검증 출처 표현이 적다는 뜻이며, 단독 AI 판정 근거는 아닙니다."},
    ]

    return {
        "selectedMode": selected_mode,
        "verdictLabel": verdict_label,
        "fakePercent": fake_percent,
        "realPercent": real_percent,
        "confidence": confidence,
        "summary": summary,
        "reasoning": [
            {"title": "연결된 텍스트 모델", "body": f"{model_source} 번들을 감지했고 상태는 `{runtime_status}`입니다."},
            {"title": "핵심 문체 신호", "body": primary_reason},
            {"title": "Text XAI", "body": "표현 하이라이트, 문장 span별 설명 신호, 반복/검증 출처/문장 규칙성 지표를 함께 보여줍니다."},
        ],
        "metrics": [
            {"label": "Real score", "value": f"{real_percent:.1f}%", "detail": "사람 작성/진본 쪽 확률"},
            {"label": "Fake score", "value": f"{fake_percent:.1f}%", "detail": "AI 생성/합성 쪽 확률"},
            {"label": "Language", "value": language_label, "detail": model_source},
            {"label": "Model state", "value": "Loaded" if detector is not None else "Fallback", "detail": runtime_status},
        ],
        "stages": [
            {"title": "Parse", "body": "문장과 토큰을 나누고 언어를 감지합니다."},
            {"title": "Model pass", "body": f"{model_source} 모델 경로로 판정을 시도합니다."},
            {"title": "Signal scoring", "body": "반복, 문장 길이, 검증 출처 표현, 문체 안정성을 설명용 보조 신호로 정리합니다."},
            {"title": "Explain", "body": "토큰 chip과 span timeline으로 XAI를 구성합니다."},
        ],
        "fusionSteps": [
            {"title": "언어 라우팅", "weight": "auto", "logic": f"입력 언어를 {language_label}로 감지해 {model_source} 경로를 선택했습니다."},
            {"title": "모델 확률 반영", "weight": f"Fake {fake_percent:.1f}%", "logic": "실제 모델 출력 확률을 최우선으로 사용하고, 실패 시 같은 UI 구조에서 fallback XAI를 사용합니다."},
            {"title": "검증 출처 신호 정리", "weight": f"{features['citationCount']} evidence marks", "logic": "출처·근거·보고서·링크 같은 검증 출처 표현이 있는지 확인해 문장 span 설명에 보조 단서로 표시합니다."},
        ],
        "modelTraits": [
            {"model": model_source, "role": "AI text detection", "trait": "문체와 토큰 예측 패턴으로 AI 생성 가능성을 계산", "contribution": "최종 Real/Fake 판정의 중심 점수"},
            {"model": "Token XAI", "role": "Local explanation", "trait": "반복, 긴 표현, 검증 출처 표현을 chip으로 표시", "contribution": "사용자가 어떤 표현이 설명용 보조 단서로 잡혔는지 확인"},
            {"model": "Claim span map", "role": "Grounding", "trait": "문장 단위로 검증 출처 표현과 반복 신호를 분리", "contribution": "문장별 설명 신호를 읽는 연결 지점"},
        ],
        "xaiHeadline": "Text XAI는 표현 단위 보조 단서와 문장 span별 설명 신호를 함께 보여줍니다.",
        "xai": {
            "headline": "Expression trace + sentence span map",
            "regions": [
                {"id": "expression-flow", "label": "Expression flow", "x": 8, "y": 20, "width": 36, "height": 28, "score": round(fake_percent / 100.0, 3), "note": "Text model expression signal"},
                {"id": "source-cue-map", "label": "Source cue map", "x": 52, "y": 28, "width": 34, "height": 24, "score": round(1.0 - clamp(features["citationCount"] / 3.0, 0.0, 1.0), 3), "note": "Verifiable source cue gap"},
            ],
            "timeline": timeline,
            "textHighlights": tokens,
            "modalityBars": modality_bars,
        },
    }


def analyze_image(image_path: Path, selected_mode: str, image_scope: str = "full-scene") -> dict[str, Any]:
    model, transform, device = ensure_image_model()
    image = Image.open(image_path)
    full_rgb = pil_to_rgb_array(image)
    full_rgb = cv2.resize(full_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    face_box = detect_face_box(full_rgb)

    primary_rgb = crop_image_box(full_rgb, face_box if image_scope == "face-focus" else None)
    precision_face_rgb = crop_image_box(full_rgb, face_box) if face_box is not None else primary_rgb.copy()

    rgb_tensor = transform(Image.fromarray(primary_rgb)).unsqueeze(0).to(device)
    fft_map = compute_fft_map(cv2.cvtColor(primary_rgb, cv2.COLOR_RGB2GRAY))
    fft_tensor = fft_map_to_tensor(fft_map, device)

    with torch.no_grad():
        primary_logit = model(rgb_tensor, fft_tensor)
        primary_real = float(torch.sigmoid(primary_logit).detach().cpu().item())
        primary_fake = 1.0 - primary_real

    precision_fake = primary_fake
    precision_real = primary_real
    face_score = 0.0
    if face_box is not None:
        face_rgb_tensor = transform(Image.fromarray(precision_face_rgb)).unsqueeze(0).to(device)
        face_fft_map = compute_fft_map(cv2.cvtColor(precision_face_rgb, cv2.COLOR_RGB2GRAY))
        face_fft_tensor = fft_map_to_tensor(face_fft_map, device)
        with torch.no_grad():
            face_logit = model(face_rgb_tensor, face_fft_tensor)
            face_real = float(torch.sigmoid(face_logit).detach().cpu().item())
            face_fake = 1.0 - face_real
        precision_real = float((primary_real * 0.42) + (face_real * 0.58))
        precision_fake = 1.0 - precision_real
        face_score = face_fake

    clip_real, clip_fake, text_alignment = clip_real_fake(primary_rgb, "")
    if selected_mode == "image-precision":
        real_score = clamp(precision_real * 0.66 + clip_real * 0.22 + (1.0 - float(np.mean(fft_map))) * 0.12, 0.0, 1.0)
    else:
        real_score = clamp(primary_real * 0.58 + clip_real * 0.32 + (1.0 - float(np.mean(fft_map))) * 0.10, 0.0, 1.0)
    fake_score = 1.0 - real_score

    fake_focus = fake_score >= real_score
    heatmap = gradcam_for_image_model(model, rgb_tensor, fft_tensor, fake_focus=fake_focus)
    if float(heatmap.max()) <= 1e-6:
        heatmap = clip_occlusion_heatmap(primary_rgb, fake_focus=fake_focus)
    prior = roi_prior_heatmap(face_box, None)
    if float(prior.max()) > 0:
        heatmap = normalize_array(heatmap * 0.72 + prior * 0.28)
    regions = build_regions_from_heatmap(heatmap, face_box, None)
    focus_frame = render_focus_frame(primary_rgb, heatmap, regions)
    fft_map_url = fft_map_to_data_url(fft_map)

    real_ref, fake_ref = compute_image_references(fft_map)
    sample_profile = build_image_frequency_profile(fft_map)
    face_ratio = 1.0 if face_box is not None else 0.0
    availability = {
        "hasFace": face_box is not None,
        "hasLips": False,
        "hasSpeech": False,
        "hasText": False,
        "faceRatio": round(face_ratio, 3),
        "mouthRatio": 0.0,
        "subtitleRatio": 0.0,
        "speechConfidence": 0.0,
        "textConfidence": 0.0,
    }

    summary = "진본 콘텐츠일 가능성이 높게 평가되었습니다." if real_score >= fake_score else "합성 콘텐츠일 가능성이 높게 평가되었습니다."
    verdict = "Likely authentic" if real_score >= fake_score else "Likely synthetic"
    confidence = int(round(abs(real_score - fake_score) * 100))
    fft_strength = float(np.mean(sample_profile) / 100.0)
    fused_branch_fake = precision_fake if selected_mode == "image-precision" else primary_fake
    fusion_weights = [
        {"label": "RGB branch", "weight": 52 if selected_mode == "image-fast" else 44},
        {"label": "FFT branch", "weight": 30 if selected_mode == "image-fast" else 34},
        {"label": "Face signal", "weight": 18 if face_box is not None else 0},
    ]

    reasoning = [
        {
            "title": "RGB stream",
            "body": "주 장면의 질감과 경계, 조명 패턴을 기반으로 진본성과 합성 흔적을 함께 비교했습니다.",
        },
        {
            "title": "FFT stream",
            "body": "회색조 주파수 분포를 통해 중심 저주파 집중과 고주파 잔여 패턴을 함께 읽었습니다.",
        },
        {
            "title": "XAI focus",
            "body": "히트맵은 RGB backbone이 가장 크게 반응한 위치를 나타내며, 얼굴이 검출되면 얼굴 주변을 우선적으로 해석합니다." if face_box is not None else "히트맵은 장면 전체에서 모델 반응이 높았던 영역을 기준으로 생성되었습니다.",
        },
    ]

    stages = [
        {"title": "Image ingest", "body": "입력 이미지를 정규화하고 얼굴 존재 여부를 먼저 확인했습니다."},
        {"title": "Dual-stream inference", "body": "RGB stream과 FFT stream을 각각 분석한 뒤 하나의 점수로 결합했습니다."},
        {"title": "Face-aware refinement", "body": "정밀 모드에서는 얼굴이 있을 때 얼굴 crop을 추가 반영했습니다." if selected_mode == "image-precision" else "빠른 모드에서는 장면 전체를 우선적으로 반영했습니다."},
        {"title": "Explainable evidence", "body": "Grad-CAM 스타일 heatmap과 주파수 맵을 함께 생성했습니다."},
    ]

    metrics = [
        {"label": "Real score", "value": f"{real_score * 100:.1f}%", "detail": "image authenticity confidence"},
        {"label": "Fake score", "value": f"{fake_score * 100:.1f}%", "detail": "synthetic likelihood"},
        {"label": "Model", "value": selected_mode.replace("image-", ""), "detail": "single-modal image backend"},
        {"label": "Confidence", "value": f"{confidence}%", "detail": "dual-stream certainty"},
    ]

    frequency_note = (
        "현재 이미지의 FFT 분포를 기준으로, 일반적인 진본 이미지 경향과 합성 이미지 경향을 함께 비교합니다. "
        "중심 저주파는 자연 영상에서 흔하지만, 외곽 잔여가 과도하게 들뜨면 합성 단서로 더 강하게 해석합니다."
    )

    return {
        "selectedMode": selected_mode,
        "inferenceMode": "single",
        "verdictLabel": verdict,
        "fakePercent": round(fake_score * 100, 1),
        "realPercent": round(real_score * 100, 1),
        "confidence": confidence,
        "summary": summary,
        "reasoning": reasoning,
        "metrics": metrics,
        "stages": stages,
        "xaiHeadline": "RGB 흐름과 FFT 흐름을 함께 반영해 최종 판정 값을 산출했습니다.",
        "modalityJudgments": [
            {
                "label": "RGB branch",
                "realPercent": round((1.0 - fused_branch_fake) * 100, 1),
                "fakePercent": round(fused_branch_fake * 100, 1),
                "verdict": "진본 우세" if fused_branch_fake < 0.5 else "합성 우세",
                "reason": "색상, 질감, 경계 패턴을 중심으로 장면의 자연스러움을 평가했습니다.",
            },
            {
                "label": "FFT branch",
                "realPercent": round((1.0 - fft_strength) * 100, 1),
                "fakePercent": round(fft_strength * 100, 1),
                "verdict": "진본 우세" if fft_strength < 0.5 else "합성 우세",
                "reason": "주파수 분포에서 중심 저주파와 외곽 잔여 패턴의 비중을 읽었습니다.",
            },
            {
                "label": "Face signal",
                "realPercent": round((1.0 - face_score) * 100, 1) if face_box is not None else 0.0,
                "fakePercent": round(face_score * 100, 1) if face_box is not None else 0.0,
                "verdict": "활성" if face_box is not None else "gated down",
                "reason": "얼굴이 검출되면 얼굴 crop을 추가 반영하고, 없으면 장면 전체 판단만 유지합니다.",
            },
        ],
        "fusionSteps": [
            {"title": "Pre-check", "weight": "face-aware", "logic": "얼굴이 있는지 먼저 판단하고, 정밀 모드에서는 얼굴 crop을 추가로 반영합니다."},
            {"title": "RGB + FFT", "weight": "dual stream", "logic": "RGB 장면 단서와 FFT 주파수 단서를 함께 결합합니다."},
            {"title": "Final image verdict", "weight": "single-modal", "logic": "장면 질감, 주파수 잔여, 얼굴 유무를 종합해 최종 이미지를 판정합니다."},
        ],
        "modelTraits": [
            {"model": "RGB stream", "role": "시각 질감", "trait": "조명, 피부/배경 질감, 경계 패턴을 읽습니다.", "contribution": "전체 장면의 자연스러움을 반영합니다."},
            {"model": "FFT stream", "role": "주파수 포렌식", "trait": "중심 저주파와 외곽 고주파 잔여를 비교합니다.", "contribution": "생성형 잔여 패턴을 보조적으로 반영합니다."},
            {"model": "Face-aware refinement", "role": "얼굴 보강", "trait": "정밀 모드에서 얼굴이 보일 때 얼굴 crop을 추가 평가합니다.", "contribution": "얼굴 중심 이미지에서 오판을 줄이는 데 기여합니다."},
        ],
        "spectrumBins": sample_profile,
        "syncBins": [],
        "fusionWeights": [item for item in fusion_weights if item["weight"] > 0],
        "availability": availability,
        "frequencyComparison": {
            "realReference": real_ref,
            "fakeReference": fake_ref,
            "sample": sample_profile,
            "note": frequency_note,
            "sampleImage": fft_map_url,
        },
        "gatedBranches": [] if face_box is not None else ["Face signal"],
        "xai": {
            "headline": "이미지 전용 XAI는 Grad-CAM 스타일 heatmap과 FFT 주파수 맵을 함께 사용합니다.",
            "regions": regions,
            "timeline": [],
            "textHighlights": [],
            "modalityBars": [
                {"label": "RGB stream", "score": fused_branch_fake, "note": "scene texture / lighting / edge pattern"},
                {"label": "FFT stream", "score": fft_strength, "note": "frequency residue / low-high balance"},
                {"label": "Face signal", "score": face_score if face_box is not None else 0.08, "note": "face crop confidence / gated if absent"},
                {"label": "Overall", "score": fake_score, "note": "final synthetic likelihood"},
            ],
            "focusFrame": focus_frame,
        },
    }


def analyze_image_runtime(image_path: Path, selected_mode: str, image_scope: str = "full-scene") -> dict[str, Any]:
    precision_model, precision_transform, device = ensure_image_model()
    face_model, face_transform, _ = ensure_face_image_model()

    image = Image.open(image_path)
    full_rgb = pil_to_rgb_array(image)
    full_rgb = cv2.resize(full_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    face_box = detect_face_box(full_rgb)

    scene_rgb = full_rgb.copy()
    face_rgb = crop_image_box(full_rgb, face_box) if face_box is not None else scene_rgb.copy()

    scene_rgb_tensor = precision_transform(Image.fromarray(scene_rgb)).unsqueeze(0).to(device)
    scene_fft_map = compute_fft_map(cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY))
    scene_fft_tensor = fft_map_to_tensor(scene_fft_map, device)

    with torch.no_grad():
        scene_logit = precision_model(scene_rgb_tensor, scene_fft_tensor)
        scene_real = float(torch.sigmoid(scene_logit).detach().cpu().item())
        scene_fake = 1.0 - scene_real

    precision_real = scene_real
    precision_fake = scene_fake
    if face_box is not None:
        precision_face_rgb_tensor = precision_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
        precision_face_fft_map = compute_fft_map(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY))
        precision_face_fft_tensor = fft_map_to_tensor(precision_face_fft_map, device)
        with torch.no_grad():
            precision_face_logit = precision_model(precision_face_rgb_tensor, precision_face_fft_tensor)
            precision_face_real = float(torch.sigmoid(precision_face_logit).detach().cpu().item())
        precision_real = float(scene_real * 0.42 + precision_face_real * 0.58)
        precision_fake = 1.0 - precision_real

    face_model_available = face_box is not None
    face_model_real = scene_real
    face_model_fake = scene_fake
    if face_model_available:
        face_rgb_tensor = face_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
        face_fft_map = compute_fft_map(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY))
        face_fft_tensor = face_transform(fft_map_to_rgb_image(face_fft_map)).unsqueeze(0).to(device)
        with torch.no_grad():
            face_logits = face_model(face_rgb_tensor, face_fft_tensor)
            face_probs = torch.softmax(face_logits, dim=1).detach().cpu().numpy()[0]
        face_model_fake = float(face_probs[0])
        face_model_real = float(face_probs[1])
    else:
        face_rgb_tensor = None
        face_fft_tensor = None
        face_fft_map = scene_fft_map

    use_face_specialist = image_scope == "face-focus" and face_model_available
    active_model_name = "류지호 얼굴 전용 모델" if use_face_specialist else "이원석 정밀 모델"
    active_model_label = "Face-specialized branch" if use_face_specialist else "Precision branch"
    active_real = face_model_real if use_face_specialist else precision_real
    active_fake = 1.0 - active_real
    active_fft_map = face_fft_map if use_face_specialist else scene_fft_map
    active_visual = face_rgb if use_face_specialist else scene_rgb
    active_rgb_tensor = face_rgb_tensor if use_face_specialist else scene_rgb_tensor
    active_fft_tensor = face_fft_tensor if use_face_specialist else scene_fft_tensor

    clip_real, _clip_fake, _ = clip_real_fake(active_visual, "")
    fft_mean = float(np.mean(active_fft_map))
    if use_face_specialist:
        real_score = clamp(active_real * 0.72 + clip_real * 0.18 + (1.0 - fft_mean) * 0.10, 0.0, 1.0)
    elif selected_mode == "image-precision":
        real_score = clamp(precision_real * 0.66 + clip_real * 0.22 + (1.0 - fft_mean) * 0.12, 0.0, 1.0)
    else:
        real_score = clamp(scene_real * 0.58 + clip_real * 0.32 + (1.0 - fft_mean) * 0.10, 0.0, 1.0)
    fake_score = 1.0 - real_score

    fake_focus = fake_score >= real_score
    if use_face_specialist and active_rgb_tensor is not None and active_fft_tensor is not None:
        heatmap = gradcam_for_face_image_model(face_model, active_rgb_tensor, active_fft_tensor, fake_focus=fake_focus)
    else:
        heatmap = gradcam_for_image_model(precision_model, active_rgb_tensor, active_fft_tensor, fake_focus=fake_focus)
    if float(heatmap.max()) <= 1e-6:
        heatmap = clip_occlusion_heatmap(active_visual, fake_focus=fake_focus)
    prior = roi_prior_heatmap(face_box if not use_face_specialist else (0, 0, FRAME_SIZE, FRAME_SIZE) if face_model_available else face_box, None)
    if float(prior.max()) > 0:
        heatmap = normalize_array(heatmap * 0.72 + prior * 0.28)
    regions = build_regions_from_heatmap(heatmap, face_box if not use_face_specialist else None, None)
    focus_frame = render_focus_frame(active_visual, heatmap, regions)
    fft_map_url = fft_map_to_data_url(active_fft_map)

    real_ref, fake_ref = compute_image_references(active_fft_map)
    sample_profile = build_image_frequency_profile(active_fft_map)
    fft_strength = float(np.mean(sample_profile) / 100.0)
    face_ratio = 1.0 if face_box is not None else 0.0

    availability = {
        "hasFace": face_box is not None,
        "hasLips": False,
        "hasSpeech": False,
        "hasText": False,
        "faceRatio": round(face_ratio, 3),
        "mouthRatio": 0.0,
        "subtitleRatio": 0.0,
        "speechConfidence": 0.0,
        "textConfidence": 0.0,
    }

    summary = "진본 콘텐츠일 가능성이 높게 평가되었습니다." if real_score >= fake_score else "합성 콘텐츠일 가능성이 높게 평가되었습니다."
    verdict = "Likely authentic" if real_score >= fake_score else "Likely synthetic"
    confidence = int(round(abs(real_score - fake_score) * 100))

    fusion_weights = [
        {"label": active_model_label, "weight": 56 if use_face_specialist else 44 if selected_mode == "image-precision" else 52},
        {"label": "OpenCLIP scene prior", "weight": 18 if use_face_specialist else 22 if selected_mode == "image-precision" else 32},
        {"label": "FFT reference", "weight": 10 if use_face_specialist else 12 if selected_mode == "image-precision" else 10},
        {"label": "Face refinement", "weight": 16 if face_box is not None and not use_face_specialist else 0},
    ]

    frequency_note = (
        "현재 이미지의 FFT 분포를 기준으로 일반적인 real 이미지 경향과 synthetic 이미지 경향을 함께 비교합니다. "
        "중심 저주파는 자연 영상에서도 강하지만, 축 방향 잔여와 고주파 분포가 과도하게 두드러지면 합성 단서로 더 강하게 해석합니다."
    )

    face_reason = (
        "얼굴 전용 경로에서 류지호 모델이 얼굴 crop과 FFT를 함께 읽었습니다."
        if face_model_available
        else "얼굴이 안정적으로 검출되지 않아 얼굴 전용 경로는 gated down 되었고 장면 기준 판단으로 대체했습니다."
    )

    return {
        "selectedMode": selected_mode,
        "inferenceMode": "single",
        "verdictLabel": verdict,
        "fakePercent": round(fake_score * 100, 1),
        "realPercent": round(real_score * 100, 1),
        "confidence": confidence,
        "summary": summary,
        "reasoning": [
            {"title": active_model_name, "body": "정밀 경로에서는 RGB+FFT 이중 스트림을 사용하고, 얼굴 전용 경로에서는 얼굴 crop 기준의 late-fusion 판단을 수행합니다."},
            {"title": "주파수 근거", "body": "FFT 분포를 기준으로 저주파 중심 구조와 고주파 잔여 패턴을 함께 비교합니다."},
            {"title": "시각적 근거", "body": "Grad-CAM 스타일 heatmap으로 현재 판정에서 반응이 컸던 영역을 강조했습니다."},
        ],
        "metrics": [
            {"label": "Real score", "value": f"{real_score * 100:.1f}%", "detail": "image authenticity confidence"},
            {"label": "Fake score", "value": f"{fake_score * 100:.1f}%", "detail": "synthetic likelihood"},
            {"label": "Model route", "value": "류지호 face" if use_face_specialist else "이원석 precision", "detail": "active primary checkpoint"},
            {"label": "Confidence", "value": f"{confidence}%", "detail": "single-modal certainty"},
        ],
        "stages": [
            {"title": "Image ingest", "body": "입력 이미지를 정규화하고 얼굴 존재 여부를 먼저 확인했습니다."},
            {"title": "Model route", "body": "전체 장면이면 이원석 모델, 얼굴 전용이면 류지호 모델을 우선 사용하도록 경로를 분기했습니다."},
            {"title": "Dual-stream inference", "body": "RGB 단서와 FFT 주파수 단서를 함께 반영해 판정 점수를 계산했습니다."},
            {"title": "Explainable evidence", "body": "heatmap과 FFT 맵을 함께 생성해 어떤 근거가 사용됐는지 시각화했습니다."},
        ],
        "xaiHeadline": "이원석 정밀 모델과 류지호 얼굴 전용 모델 중 현재 경로에 맞는 체크포인트를 사용해 최종 판정 값을 산출했습니다.",
        "modalityJudgments": [
            {
                "label": "이원석 precision",
                "realPercent": round(precision_real * 100, 1),
                "fakePercent": round(precision_fake * 100, 1),
                "verdict": "진본 우세" if precision_fake < 0.5 else "합성 우세",
                "reason": "RGB+FFT 이중 스트림으로 전체 장면과 얼굴 재판독 단서를 함께 반영했습니다.",
            },
            {
                "label": "류지호 face",
                "realPercent": round(face_model_real * 100, 1) if face_model_available else 0.0,
                "fakePercent": round(face_model_fake * 100, 1) if face_model_available else 0.0,
                "verdict": "활성" if face_model_available else "gated down",
                "reason": face_reason,
            },
            {
                "label": "FFT branch",
                "realPercent": round((1.0 - fft_strength) * 100, 1),
                "fakePercent": round(fft_strength * 100, 1),
                "verdict": "진본 우세" if fft_strength < 0.5 else "합성 우세",
                "reason": "주파수 분포에서 중심 저주파와 축 방향 잔여 패턴을 기준으로 비교했습니다.",
            },
        ],
        "fusionSteps": [
            {"title": "Pre-check", "weight": "face-aware", "logic": "얼굴이 검출되면 얼굴 전용 모델 경로를 활성화하고, 없으면 장면 기준 모델로 유지합니다."},
            {"title": "Checkpoint route", "weight": active_model_name, "logic": "정밀 버전은 이원석 모델, 얼굴 전용 초점은 류지호 모델이 주 분기를 담당합니다."},
            {"title": "Final image verdict", "weight": "single-modal", "logic": "활성 모델 점수, FFT 기준, OpenCLIP 장면 prior를 결합해 최종 이미지를 판정합니다."},
        ],
        "modelTraits": [
            {"model": "이원석 모델", "role": "정밀 판독", "trait": "RGB 장면 단서와 FFT 주파수 단서를 이중 스트림으로 결합합니다.", "contribution": "정밀 버전에서 전체 장면과 얼굴 재판독을 함께 반영합니다."},
            {"model": "류지호 모델", "role": "얼굴 전용 판독", "trait": "얼굴 crop 중심 late-fusion 구조로 표정, 피부 질감, 주파수 잔여를 읽습니다.", "contribution": "얼굴 전용 초점에서 가장 먼저 반영되는 체크포인트입니다."},
            {"model": "FFT branch", "role": "주파수 단서", "trait": "중심 저주파와 고주파 잔여 패턴의 편차를 비교합니다.", "contribution": "실제/합성 주파수 기준과 현재 이미지를 비교하는 보조 근거를 제공합니다."},
        ],
        "spectrumBins": sample_profile,
        "syncBins": [],
        "fusionWeights": [item for item in fusion_weights if item["weight"] > 0],
        "availability": availability,
        "frequencyComparison": {
            "realReference": real_ref,
            "fakeReference": fake_ref,
            "sample": sample_profile,
            "note": frequency_note,
            "sampleImage": fft_map_url,
        },
        "gatedBranches": [] if face_model_available or not use_face_specialist else ["류지호 face"],
        "xai": {
            "headline": "이미지 전용 XAI는 현재 활성 모델 경로의 heatmap과 FFT 맵을 함께 보여줍니다.",
            "regions": regions,
            "timeline": [],
            "textHighlights": [],
            "modalityBars": [
                {"label": active_model_label, "score": active_fake, "note": active_model_name},
                {"label": "OpenCLIP prior", "score": 1.0 - clip_real, "note": "scene-text alignment prior"},
                {"label": "FFT branch", "score": fft_strength, "note": "frequency residue / low-high balance"},
                {"label": "Overall", "score": fake_score, "note": "final synthetic likelihood"},
            ],
            "focusFrame": focus_frame,
        },
    }


def make_video_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def ensure_video_ensemble() -> dict[str, Any]:
    with VIDEO_MODEL_LOCK:
        if VIDEO_MODEL_STATE:
            return VIDEO_MODEL_STATE

        if not VIDEO_MODEL_BUNDLE_DIR.exists():
            raise FileNotFoundError(f"video model bundle not found: {VIDEO_MODEL_BUNDLE_DIR}")

        device = analysis_device()
        models = []
        for spec in VIDEO_MODEL_SPECS:
            checkpoint_path = VIDEO_MODEL_BUNDLE_DIR / spec["folder"] / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"video checkpoint not found: {checkpoint_path}")
            model = VideoFrameClassifier(pretrained=False, dropout=0.4, hidden_dim=0).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(
                {
                    "label": spec["label"],
                    "image_size": spec["image_size"],
                    "role": spec["role"],
                    "path": str(checkpoint_path),
                    "epoch": checkpoint.get("epoch"),
                    "bestMetric": checkpoint.get("best_metric"),
                    "bestEpoch": checkpoint.get("best_epoch"),
                    "model": model,
                }
            )

        VIDEO_MODEL_STATE.update(
            {
                "device": device,
                "models": models,
                "transforms": {224: make_video_transform(224), 320: make_video_transform(320)},
            }
        )
        return VIDEO_MODEL_STATE


def apply_video_text_mask(rgb_np: np.ndarray) -> np.ndarray:
    h = rgb_np.shape[0]
    out = rgb_np.copy()
    top = int(round(h * 0.08))
    bottom = int(round(h * 0.18))
    if top > 0:
        out[:top] = np.median(out[:top].reshape(-1, 3), axis=0).astype(np.uint8)
    if bottom > 0:
        out[-bottom:] = np.median(out[-bottom:].reshape(-1, 3), axis=0).astype(np.uint8)
    return out


def sample_video_ensemble_frames(video_path: Path, n_frames: int = VIDEO_SAMPLE_FRAMES) -> tuple[list[dict[str, Any]], dict[str, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("video file could not be opened")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            raise RuntimeError("video has no readable frames")
        indices = np.linspace(0, total - 1, n_frames).astype(int)
        frames: list[dict[str, Any]] = []
        for order, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                continue
            timestamp = float(index / fps) if fps > 0 else float(order)
            frames.append(
                {
                    "index": int(index),
                    "timestamp": timestamp,
                    "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                }
            )
        if not frames:
            raise RuntimeError("no sampled frames could be decoded")
        duration = float(total / fps) if fps > 0 else float(frames[-1]["timestamp"])
        return frames, {"totalFrames": float(total), "fps": fps, "durationSec": duration}
    finally:
        capture.release()


def video_predict_with_tta(model: nn.Module, batch_tensor: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        x = batch_tensor.to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1)
        flipped_probs = torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)
        return ((probs + flipped_probs) * 0.5).detach().cpu().numpy()


def predict_video_ensemble(video_path: Path) -> dict[str, Any]:
    state = ensure_video_ensemble()
    frames, video_meta = sample_video_ensemble_frames(video_path)
    masked_frames = [apply_video_text_mask(item["rgb"]) for item in frames]
    model_probs = []

    for model_state in state["models"]:
        image_size = int(model_state["image_size"])
        transform = state["transforms"][image_size]
        batch = torch.stack([transform(Image.fromarray(frame)) for frame in masked_frames], dim=0)
        probs = video_predict_with_tta(model_state["model"], batch, state["device"])
        model_probs.append(probs)

    probs_3d = np.stack(model_probs, axis=0)
    median_per_frame = np.median(probs_3d, axis=0)
    median_per_frame = median_per_frame / np.maximum(median_per_frame.sum(axis=1, keepdims=True), 1e-9)
    max_probs = median_per_frame.max(axis=1)
    frame_weights = max_probs**2
    frame_weights = frame_weights / max(float(frame_weights.sum()), 1e-9)
    final_probs = (median_per_frame * frame_weights[:, None]).sum(axis=0)
    final_probs = final_probs / max(float(final_probs.sum()), 1e-9)

    per_model = []
    for index, model_state in enumerate(state["models"]):
        frame_scores = probs_3d[index, :, 1]
        per_model.append(
            {
                "label": model_state["label"],
                "imageSize": model_state["image_size"],
                "role": model_state["role"],
                "pGen": float(np.mean(frame_scores)),
                "minPGen": float(np.min(frame_scores)),
                "maxPGen": float(np.max(frame_scores)),
                "path": model_state["path"],
                "bestMetric": model_state.get("bestMetric"),
            }
        )

    per_frame = []
    for index, frame in enumerate(frames):
        per_frame.append(
            {
                "label": f"Frame {index + 1}",
                "index": frame["index"],
                "timestamp": frame["timestamp"],
                "weight": float(frame_weights[index]),
                "pReal": float(median_per_frame[index, 0]),
                "pGen": float(median_per_frame[index, 1]),
                "modelPGen": [float(value) for value in probs_3d[:, index, 1]],
                "rgb": frame["rgb"],
            }
        )

    return {
        "pReal": float(final_probs[0]),
        "pGen": float(final_probs[1]),
        "perModel": per_model,
        "perFrame": per_frame,
        "videoMeta": video_meta,
        "device": state["device"],
    }


def build_video_processing_scope(prediction: dict[str, Any]) -> dict[str, Any]:
    video_meta = prediction["videoMeta"]
    per_frame = prediction["perFrame"]
    full_duration = float(video_meta.get("durationSec") or 0.0)
    if per_frame:
        analyzed_duration = max(item["timestamp"] for item in per_frame) - min(item["timestamp"] for item in per_frame)
    else:
        analyzed_duration = 0.0
    return {
        "readsWholeVideo": False,
        "fullDurationSec": round(full_duration, 3),
        "analyzedDurationSec": round(max(analyzed_duration, 0.0), 3),
        "sampleFrames": len(per_frame),
        "maxSeconds": round(full_duration, 3),
        "strategy": "원본 영상 전체 길이에서 6개 프레임을 균등 간격으로 샘플링한 뒤, 각 프레임을 동일한 text mask와 모델별 resize 규칙으로 처리합니다.",
        "precheckSummary": "상단 8%와 하단 18% 자막 영역을 median 색으로 가린 뒤 7개 EfficientNet-B0 체크포인트에 같은 프레임을 입력했습니다.",
        "computeDevice": str(prediction["device"]),
        "windows": [
            {
                "label": item["label"],
                "start": float(item["timestamp"]),
                "end": float(item["timestamp"]),
                "startLabel": format_mmss(float(item["timestamp"])),
                "endLabel": format_mmss(float(item["timestamp"])),
            }
            for item in per_frame
        ],
    }


def analyze_video_ensemble(video_path: Path) -> dict[str, Any]:
    prediction = predict_video_ensemble(video_path)
    p_gen = prediction["pGen"]
    p_real = prediction["pReal"]
    is_generated = p_gen > 0.5
    verdict_label = "Likely generated" if is_generated else "Likely authentic"
    confidence = int(round(max(p_gen, p_real) * 100))
    top_frame = max(prediction["perFrame"], key=lambda item: item["pGen"] if is_generated else item["pReal"])
    focus_frame = rgb_array_to_data_url(top_frame["rgb"], fmt="JPEG")
    masked_focus_frame = rgb_array_to_data_url(apply_video_text_mask(top_frame["rgb"]), fmt="JPEG")
    per_model_sorted = sorted(prediction["perModel"], key=lambda item: item["pGen"], reverse=True)

    timeline = []
    for item in prediction["perFrame"]:
        strongest_model_index = int(np.argmax(np.asarray(item["modelPGen"], dtype=np.float32)))
        weakest_model_index = int(np.argmin(np.asarray(item["modelPGen"], dtype=np.float32)))
        strongest = prediction["perModel"][strongest_model_index]
        weakest = prediction["perModel"][weakest_model_index]
        timestamp = float(item["timestamp"])
        timeline.append(
            {
                "label": item["label"],
                "start": format_mmss(timestamp),
                "end": format_mmss(timestamp),
                "score": round(float(item["pGen"]), 4),
                "note": f"median ensemble p_gen {item['pGen']:.3f}, frame weight {item['weight']:.3f}",
                "evidence": [
                    f"highest {strongest['label']} {item['modelPGen'][strongest_model_index] * 100:.1f}%",
                    f"lowest {weakest['label']} {item['modelPGen'][weakest_model_index] * 100:.1f}%",
                    "TTA hflip averaged",
                ],
            }
        )

    model_labels = [item["label"] for item in prediction["perModel"]]
    frame_spread = [float(np.std(np.asarray(item["modelPGen"], dtype=np.float32))) for item in prediction["perFrame"]]
    mean_spread = float(np.mean(frame_spread)) if frame_spread else 0.0
    if mean_spread < 0.06:
        consensus = "7개 모델 의견이 전반적으로 가깝습니다."
    elif mean_spread < 0.14:
        consensus = "일부 모델 차이는 있지만 전체 방향은 median 집계로 안정화됩니다."
    else:
        consensus = "모델 간 의견 차이가 큰 편이어서 단일 모델보다 median 앙상블 해석이 중요합니다."
    if is_generated:
        interpretation = (
            f"{top_frame['label']}에서 generated 신호가 가장 강했고, "
            f"최종 p_generated {p_gen * 100:.1f}%가 기준값 50%를 넘었습니다."
        )
    else:
        interpretation = (
            f"대부분의 샘플 프레임에서 generated 확률이 낮게 유지됐고, "
            f"최종 p_real {p_real * 100:.1f}%가 우세했습니다."
        )

    model_bars = [
        {
            "label": item["label"],
            "score": round(float(item["pGen"]), 4),
            "note": f"{item['imageSize']}px / {item['role']}",
        }
        for item in prediction["perModel"]
    ]
    metrics = [
        {"label": "p_real", "value": f"{p_real * 100:.1f}%", "detail": "class 0, real probability after final confidence_mean aggregation"},
        {"label": "p_generated", "value": f"{p_gen * 100:.1f}%", "detail": "class 1, generated probability after final confidence_mean aggregation"},
        {"label": "Models", "value": "7", "detail": "EfficientNet-B0 checkpoints loaded from the video handoff bundle"},
        {"label": "Aggregation", "value": "median", "detail": "per-frame model probabilities are combined by median and renormalized"},
    ]
    reasoning = [
        {
            "title": "실제 입력 처리",
            "body": "영상 전체 길이에서 6개 프레임을 균등하게 뽑고, 각 프레임의 상단 8%와 하단 18%를 median 색으로 가려 자막 단서 의존을 줄였습니다.",
        },
        {
            "title": "7개 모델 앙상블",
            "body": "robustaug, EMA, holdout, seed, 320px 모델을 모두 실행하고 원본과 좌우 반전 결과의 softmax를 평균했습니다.",
        },
        {
            "title": "최종 판정",
            "body": f"프레임별 7개 모델 점수는 median으로 합치고, 확신도가 높은 프레임에 더 큰 가중치를 주는 confidence_mean으로 최종 p_generated {p_gen * 100:.1f}%를 계산했습니다.",
        },
    ]
    fusion_steps = [
        {"title": "Frame sampling", "weight": "6 frames", "logic": "원본 길이와 관계없이 시작부터 끝까지 균등한 위치에서 대표 프레임을 선택합니다."},
        {"title": "Text mask", "weight": "top 8% / bottom 18%", "logic": "자막이나 로고 텍스트가 판정 shortcut이 되지 않도록 상단과 하단 band를 median 색으로 채웁니다."},
        {"title": "N=7 TTA inference", "weight": "7 x 2 passes", "logic": "각 모델은 원본 프레임과 좌우 반전 프레임을 모두 보고 softmax 확률을 평균합니다."},
        {"title": "Median + confidence_mean", "weight": "production default", "logic": "모델 간 outlier를 median으로 줄이고, 확신도가 높은 프레임을 더 크게 반영해 최종 확률을 산출합니다."},
    ]
    model_traits = [
        {
            "model": item["label"],
            "role": f"{item['imageSize']}px frame classifier",
            "trait": item["role"],
            "contribution": f"평균 p_generated {item['pGen'] * 100:.1f}%로 최종 median 앙상블에 참여했습니다.",
        }
        for item in prediction["perModel"]
    ]

    summary = (
        f"7개 EfficientNet-B0 비디오 앙상블 결과 {verdict_label} 판정입니다. "
        f"Real {p_real * 100:.1f}%, Generated {p_gen * 100:.1f}%로 계산됐습니다."
    )
    return {
        "selectedMode": "video-efficientnet-n7",
        "inferenceMode": "ensemble",
        "verdictLabel": verdict_label,
        "fakePercent": round(p_gen * 100, 1),
        "realPercent": round(p_real * 100, 1),
        "confidence": confidence,
        "summary": summary,
        "reasoning": reasoning,
        "metrics": metrics,
        "processingScope": build_video_processing_scope(prediction),
        "stages": [
            {"title": "Decode", "body": "OpenCV로 영상을 열고 전체 프레임 수와 FPS를 확인했습니다."},
            {"title": "Mask + resize", "body": "프레임별 자막 band를 가린 뒤 224px 또는 320px 정사각 resize와 ImageNet normalization을 적용했습니다."},
            {"title": "N=7 TTA", "body": "7개 체크포인트에서 원본과 좌우 반전 추론을 실행하고 softmax 확률을 평균했습니다."},
            {"title": "Aggregate", "body": "모델 방향은 median, 프레임 방향은 confidence_mean으로 집계했습니다."},
        ],
        "xaiHeadline": "Video-only N=7 EfficientNet ensemble",
        "availability": {
            "hasFace": False,
            "hasLips": False,
            "hasSpeech": False,
            "hasText": False,
            "faceRatio": 0.0,
            "mouthRatio": 0.0,
            "subtitleRatio": 0.0,
            "speechConfidence": 0.0,
            "textConfidence": 0.0,
        },
        "fusionSteps": fusion_steps,
        "modelTraits": model_traits,
        "fusionWeights": [{"label": item["label"], "weight": round(100 / len(prediction["perModel"]), 1)} for item in prediction["perModel"]],
        "gatedBranches": ["audio", "lip-sync", "text", "scene graph"],
        "videoXai": {
            "models": [
                {
                    "label": item["label"],
                    "imageSize": int(item["imageSize"]),
                    "role": item["role"],
                    "avgPGen": round(float(item["pGen"]), 4),
                }
                for item in prediction["perModel"]
            ],
            "frames": [
                {
                    "label": item["label"],
                    "timestamp": format_mmss(float(item["timestamp"])),
                    "pReal": round(float(item["pReal"]), 4),
                    "pGen": round(float(item["pGen"]), 4),
                    "weight": round(float(item["weight"]), 4),
                    "modelScores": [
                        {"label": model_labels[model_index], "pGen": round(float(score), 4)}
                        for model_index, score in enumerate(item["modelPGen"])
                    ],
                }
                for item in prediction["perFrame"]
            ],
            "topFrameLabel": str(top_frame["label"]),
            "consensus": consensus,
            "interpretation": interpretation,
            "maskedFocusFrame": masked_focus_frame,
        },
        "xai": {
            "headline": "프레임별 generated 확률과 7개 모델의 의견 분포를 함께 보여줍니다.",
            "regions": [],
            "timeline": timeline,
            "textHighlights": [],
            "modalityBars": model_bars + [
                {"label": "Final ensemble", "score": round(float(p_gen), 4), "note": "median per frame + confidence_mean across frames"},
            ],
            "focusFrame": focus_frame,
        },
        "debug": {
            "topGeneratedModels": [item["label"] for item in per_model_sorted[:3]],
            "modelPaths": [{ "label": item["label"], "path": item["path"] } for item in prediction["perModel"]],
        },
    }


def analyze_multimodal(video_path: Path, selected_mode: str, companion_text: str, inference_mode: str = "ensemble") -> dict[str, Any]:
    selected_mode = normalize_selected_mode(selected_mode)
    work_dir = Path(tempfile.mkdtemp(prefix="isy_mm_audio_"))
    try:
        frame_signals, sampling_windows = extract_multi_window_frames(video_path)
        wav_path = work_dir / f"{video_path.stem}.wav"
        audio, sr = extract_audio(video_path, wav_path)
        availability = compute_availability(frame_signals, audio, companion_text)

        motion_series = np.array([row.motion_score for row in frame_signals], dtype=np.float32)
        audio_env, _spectral_flatness, mel_energy = audio_features(audio, sr, len(frame_signals))
        sync_score, lag = motion_audio_sync(motion_series, audio_env)

        representative = max(frame_signals, key=lambda row: (box_area(row.face_box), row.motion_score))
        if box_area(representative.face_box) == 0:
            representative = frame_signals[len(frame_signals) // 2]
        if availability["hasLips"]:
            lip_candidates = [row for row in frame_signals if row.mouth_box is not None]
            if lip_candidates:
                representative = max(
                    lip_candidates,
                    key=lambda row: (box_area(row.mouth_box), row.motion_score, box_area(row.face_box)),
                )

        real_prob, openclip_fake_prob, text_alignment = clip_real_fake(representative.face_crop, companion_text)
        frequency_source = [row.face_crop for row in frame_signals] if availability["hasFace"] else [row.frame_rgb for row in frame_signals]
        frequency_score = float(np.mean([fft_artifact_score(item) for item in frequency_source]))

        face_centers = []
        face_areas = []
        sharpness_scores = []
        face_hits = []
        mouth_hits = []
        for row in frame_signals:
            gray = cv2.cvtColor(row.face_crop if row.face_crop.size else row.frame_rgb, cv2.COLOR_RGB2GRAY)
            sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
            face_hits.append(1.0 if row.face_box is not None else 0.0)
            mouth_hits.append(1.0 if row.mouth_box is not None else 0.0)
            if row.face_box is not None:
                x, y, w, h = row.face_box
                face_centers.append(((x + w / 2) / FRAME_SIZE, (y + h / 2) / FRAME_SIZE))
                face_areas.append((w * h) / float(FRAME_SIZE * FRAME_SIZE))

        if face_centers:
            center_arr = np.array(face_centers, dtype=np.float32)
            face_jitter = float(np.std(center_arr[:, 0]) + np.std(center_arr[:, 1]))
            area_jitter = float(np.std(np.array(face_areas, dtype=np.float32)))
        else:
            face_jitter = 0.42
            area_jitter = 0.28

        scenegraph_score = clamp((face_jitter * 2.4) + (area_jitter * 3.2), 0.0, 1.0) if availability["hasFace"] else 0.18
        text_mismatch = (1.0 - text_alignment) if availability["hasText"] else 0.10
        avsync_component = (1.0 - sync_score) if (availability["hasSpeech"] and availability["hasLips"]) else 0.08

        segment_insights = compute_window_segment_insights_v2(frame_signals, audio_env, frequency_score)
        segment_scores = [item.score for item in segment_insights]
        segment_mean = float(np.mean(segment_scores)) if segment_scores else 0.5
        top_k = max(1, min(2, len(segment_scores)))
        segment_topk_mean = float(np.mean(sorted(segment_scores, reverse=True)[:top_k])) if segment_scores else segment_mean
        segment_peak = float(max(segment_scores)) if segment_scores else segment_mean

        runtime_bundle = ensure_runtime_bundle()
        face_count = int(sum(1 for row in frame_signals if row.face_box is not None))
        face_detect_ratio = float(np.mean(face_hits)) if face_hits else 0.0
        mouth_track_ratio = float(np.mean(mouth_hits)) if mouth_hits else 0.0
        mean_face_area = float(np.mean(face_areas)) if face_areas else 0.0
        visual_sharpness = float(np.mean(sharpness_scores)) if sharpness_scores else 0.0
        audio_energy_mean = float(np.mean(audio_env)) if len(audio_env) else 0.0
        audio_energy_std = float(np.std(audio_env)) if len(audio_env) else 0.0
        motion_mean = float(np.mean(motion_series)) if len(motion_series) else 0.0
        duration_sec = float(max((frame_signals[-1].timestamp if frame_signals else 0.0), sampling_windows[-1][2] if sampling_windows else 0.0))

        runtime_frame = pd.DataFrame(
            [
                {
                    "video_id": video_path.stem,
                    "text_prompt": companion_text,
                    "face_count": face_count,
                    "audio_path": str(wav_path) if len(audio) else "",
                    "duration_sec": duration_sec,
                    "segment_mean": segment_mean,
                    "segment_topk_mean": segment_topk_mean,
                    "segment_peak": segment_peak,
                    "face_detect_ratio": face_detect_ratio,
                    "mean_face_area": mean_face_area,
                    "mouth_track_ratio": mouth_track_ratio,
                    "visual_sharpness": visual_sharpness,
                    "audio_energy_mean": audio_energy_mean,
                    "audio_energy_std": audio_energy_std,
                    "motion_mean": motion_mean,
                }
            ]
        )
        runtime_frame = add_service_quality_features(runtime_frame, runtime_bundle["runtime_stats"])
        runtime_frame["prob_fake_openclip"] = float(openclip_fake_prob)
        for method_name in runtime_bundle["head_targets"]:
            runtime_frame[f"prob_fake_{method_name}"] = predict_service_head(runtime_bundle, runtime_frame, method_name)

        if not availability["hasText"]:
            runtime_frame["prob_fake_openclip"] = contract_toward_neutral(float(runtime_frame["prob_fake_openclip"].iloc[0]), 0.55)
            runtime_frame["prob_fake_blip_nli"] = contract_toward_neutral(float(runtime_frame["prob_fake_blip_nli"].iloc[0]), 0.35)
        if not (availability["hasSpeech"] and availability["hasLips"]):
            runtime_frame["prob_fake_avsync"] = contract_toward_neutral(float(runtime_frame["prob_fake_avsync"].iloc[0]), 0.30)
        if not availability["hasFace"]:
            runtime_frame["prob_fake_scenegraph"] = contract_toward_neutral(float(runtime_frame["prob_fake_scenegraph"].iloc[0]), 0.35)

        runtime_frame = apply_gate_features(runtime_frame, runtime_bundle["runtime_stats"]["gate_thresholds"])
        weighted_model_score, adjusted_weights = compute_runtime_weighted_model_score(runtime_frame.iloc[0], runtime_bundle["base_weights"])
        runtime_frame["weighted_model_score"] = weighted_model_score
        fusion_score = predict_runtime_fusion(runtime_bundle, runtime_frame)

        model_scores = {
            "openclip": float(runtime_frame["prob_fake_openclip"].iloc[0]),
            "flava": float(runtime_frame["prob_fake_flava"].iloc[0]),
            "blip_nli": float(runtime_frame["prob_fake_blip_nli"].iloc[0]),
            "avsync": float(runtime_frame["prob_fake_avsync"].iloc[0]),
            "frequency": float(runtime_frame["prob_fake_frequency"].iloc[0]),
            "scenegraph": float(runtime_frame["prob_fake_scenegraph"].iloc[0]),
        }
        selected_key = MODE_TO_METHOD.get(selected_mode, "flava")
        selected_model_score = model_scores[selected_key]
        use_single_mode = inference_mode == "single"
        fake_score = selected_model_score if use_single_mode else fusion_score
        real_score = 1.0 - fake_score
        confidence = int(round(clamp(66 + abs(fake_score - 0.5) * 78, 63, 98)))

        heatmap = clip_occlusion_heatmap(representative.frame_rgb, fake_score >= 0.5)
        prior_heatmap = roi_prior_heatmap(representative.face_box if availability["hasFace"] else None, representative.mouth_box if availability["hasLips"] else None)
        if float(prior_heatmap.max()) > 0:
            heatmap = 0.72 * heatmap + 0.28 * prior_heatmap
            heat_peak = max(float(heatmap.max()), 1e-6)
            heatmap = (heatmap / heat_peak).astype(np.float32)

        verdict_label = "Likely synthetic" if fake_score >= real_score else "Likely authentic"
        modality_bars = [
            {"label": "OpenCLIP", "score": round(model_scores["openclip"], 3), "note": "실제 OpenCLIP 정합성 점수"},
            {"label": "FLAVA", "score": round(model_scores["flava"], 3), "note": "10000개 실험 기반 서비스 런타임 점수"},
            {"label": "BLIP+NLI", "score": round(model_scores["blip_nli"], 3), "note": "10000개 실험 기반 설명-논리 점수"},
            {"label": "AVSync", "score": round(model_scores["avsync"], 3), "note": f"립싱크 지연 {lag:+d} frames" if (availability["hasSpeech"] and availability["hasLips"]) else "음성 또는 입술 신호 gated"},
            {"label": "Frequency", "score": round(model_scores["frequency"], 3), "note": "10000개 실험 기반 주파수 점수"},
            {"label": "SceneGraph", "score": round(model_scores["scenegraph"], 3), "note": "10000개 실험 기반 구조 점수"},
            {"label": "Fusion" if not use_single_mode else "Selected model", "score": round(float(fake_score), 3), "note": "pre-check + adaptive weighting + segment aggregation" if not use_single_mode else "선택 모델 1개만 사용한 단독 판정"},
        ]

        regions = build_regions_from_heatmap(
            heatmap,
            representative.face_box if availability["hasFace"] else None,
            representative.mouth_box if availability["hasLips"] else None,
        )
        focus_frame = render_focus_frame(representative.frame_rgb, heatmap, regions)
        timeline = build_timeline_windows_v2(segment_insights, sampling_windows)

        processing_scope = build_processing_scope_windows(video_path, frame_signals, availability, sampling_windows)
        reasoning, xai_headline = build_deterministic_reasoning(
            fake_score,
            real_score,
            confidence,
            availability,
            weighted_model_score,
            segment_topk_mean,
            processing_scope,
        )
        summary = build_verdict_summary(fake_score, real_score)
        gated_branches = []
        if not availability["hasFace"]:
            gated_branches.append("얼굴")
        if not availability["hasLips"]:
            gated_branches.append("입술")
        if not availability["hasSpeech"]:
            gated_branches.append("음성")
        if not availability["hasText"]:
            gated_branches.append("텍스트")
        frequency_comparison = build_frequency_reference_bins(build_spectrum_bins(audio))
        frequency_reference_source = representative.face_crop if availability["hasFace"] and representative.face_crop.size else representative.frame_rgb
        frequency_comparison["sampleImage"] = render_frequency_map(frequency_reference_source)
        mouth_preview = None
        if availability["hasLips"] and representative.mouth_box is not None:
            mouth_rgb = crop_box(representative.frame_rgb, representative.mouth_box, size=160)
            mouth_preview = render_crop_preview(mouth_rgb, size=160)

        return {
            "selectedMode": selected_mode,
            "inferenceMode": "single" if use_single_mode else "ensemble",
            "verdictLabel": verdict_label,
            "fakePercent": round(fake_score * 100, 1),
            "realPercent": round(real_score * 100, 1),
            "confidence": confidence,
            "summary": summary,
            "reasoning": reasoning,
            "metrics": build_metrics(selected_mode, fake_score, real_score, confidence),
            "processingScope": processing_scope,
            "stages": build_stages(selected_mode, availability),
            "xaiHeadline": xai_headline,
            "availability": availability,
            "modalityJudgments": build_runtime_model_judgments(model_scores, selected_mode, availability, fake_score),
            "fusionSteps": build_single_mode_steps(selected_mode, selected_model_score, availability) if use_single_mode else build_actual_fusion_steps(selected_mode, selected_model_score, weighted_model_score, segment_topk_mean, fake_score, adjusted_weights, availability),
            "modelTraits": build_model_traits(selected_mode, availability),
            "spectrumBins": frequency_comparison["sample"],
            "syncBins": build_sync_bins(motion_series, audio_env),
            "fusionWeights": build_single_mode_weights(selected_mode) if use_single_mode else build_actual_fusion_weights(adjusted_weights),
            "frequencyComparison": frequency_comparison,
            "gatedBranches": gated_branches,
            "xai": {
                "headline": xai_headline,
                "regions": regions,
                "timeline": timeline,
                "textHighlights": build_text_highlights(companion_text, text_alignment, sync_score, frequency_score, availability),
                "modalityBars": modality_bars,
                "focusFrame": focus_frame,
                "mouthPreview": mouth_preview,
            },
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


class MultimodalHandler(BaseHTTPRequestHandler):
    server_version = "ISYMultimodal/0.3"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            json_response(
                self,
                {
                    "ok": True,
                    "service": "multimodal-inference",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "openaiModel": OPENAI_MODEL,
                    "openaiConfigured": bool(os.environ.get("OPENAI_API_KEY")),
                },
            )
            return
        json_response(self, {"ok": False, "message": "Not Found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/explain":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                analysis = payload.get("analysis") or {}
                explanation_payload = {
                    "selectedMode": payload.get("selectedMode") or analysis.get("selectedMode"),
                    "verdictLabel": analysis.get("verdictLabel"),
                    "fakePercent": analysis.get("fakePercent"),
                    "realPercent": analysis.get("realPercent"),
                    "availability": analysis.get("availability"),
                    "regions": (analysis.get("xai") or {}).get("regions") or analysis.get("regions") or [],
                    "timeline": (analysis.get("xai") or {}).get("timeline") or analysis.get("timeline") or [],
                    "fusionWeights": analysis.get("fusionWeights") or [],
                    "frequencyComparison": analysis.get("frequencyComparison") or {},
                    "gatedBranches": analysis.get("gatedBranches") or [],
                }
                sections = call_openai_xai_sections(explanation_payload) or fallback_xai_sections(explanation_payload)
                json_response(self, {"ok": True, "sections": sections})
            except Exception as error:  # noqa: BLE001
                json_response(self, {"ok": False, "message": str(error)}, status=500)
            return

        if self.path == "/explain-text":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                analysis = payload.get("analysis") or {}
                explanation_payload = {
                    "selectedMode": payload.get("selectedMode") or analysis.get("selectedMode"),
                    "text": str(payload.get("text") or "")[:2400],
                    "verdictLabel": analysis.get("verdictLabel"),
                    "fakePercent": analysis.get("fakePercent"),
                    "realPercent": analysis.get("realPercent"),
                    "timeline": (analysis.get("xai") or {}).get("timeline") or analysis.get("timeline") or [],
                    "textHighlights": (analysis.get("xai") or {}).get("textHighlights") or analysis.get("tokens") or [],
                    "modalityBars": (analysis.get("xai") or {}).get("modalityBars") or analysis.get("bars") or [],
                    "reasoning": analysis.get("reasoning") or analysis.get("reasons") or [],
                }
                sections = call_openai_text_xai_sections(explanation_payload) or fallback_text_xai_sections(explanation_payload)
                json_response(self, {"ok": True, "sections": sections})
            except Exception as error:  # noqa: BLE001
                json_response(self, {"ok": False, "message": str(error)}, status=500)
            return

        if self.path == "/analyze-text":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                text = str(payload.get("text") or "").strip()
                if not text:
                    raise ValueError("text is required")
                words = re.findall(r"[\uac00-\ud7a3A-Za-z0-9%$#@'_-]+", text)
                sentences = [item.strip() for item in re.split(r"[.!?。！？\n]+", text) if item.strip()]
                if len(text) < 30 or len(words) < 8 or len(sentences) < 2:
                    json_response(
                        self,
                        {
                            "ok": False,
                            "message": "텍스트가 너무 짧아 신뢰할 수 있는 판별을 진행하지 않았습니다. 최소 두 문장 이상, 30자 이상으로 입력해 주세요.",
                        },
                        status=400,
                    )
                    return
                selected_mode = str(payload.get("selectedMode") or "text-ai-detector").strip().lower()
                analysis = build_text_analysis(text, selected_mode)
                json_response(
                    self,
                    {
                        "ok": True,
                        "analysis": analysis,
                        "service": "local-text-backend",
                        "selectedMode": selected_mode,
                    },
                )
            except Exception as error:  # noqa: BLE001
                json_response(self, {"ok": False, "message": str(error)}, status=500)
            return

        if self.path == "/analyze-image-url":
            download_dir: Path | None = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                remote_url = str(payload.get("url") or "").strip()
                selected_mode = str(payload.get("selectedMode") or "image-fast").strip().lower()
                settings = payload.get("settings") or {}
                image_scope = str(settings.get("imageScope") or "full-scene").strip().lower()
                if image_scope not in {"full-scene", "face-focus"}:
                    image_scope = "full-scene"
                download_dir = Path(tempfile.mkdtemp(prefix="isy_image_remote_"))
                image_path = download_remote_image(remote_url, download_dir)
                analysis = analyze_image_runtime(image_path, selected_mode, image_scope)
                json_response(
                    self,
                    {
                        "ok": True,
                        "analysis": analysis,
                        "service": "local-image-backend",
                        "selectedMode": selected_mode,
                        "sourceUrl": remote_url,
                    },
                )
            except Exception as error:  # noqa: BLE001
                json_response(self, {"ok": False, "message": str(error)}, status=500)
            finally:
                if download_dir is not None:
                    shutil.rmtree(download_dir, ignore_errors=True)
            return

        if self.path == "/analyze-url":
            download_dir: Path | None = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                remote_url = validate_remote_video_url(str(payload.get("url") or ""))
                selected_mode = normalize_selected_mode(str(payload.get("selectedMode") or "mm-flava"))
                settings = payload.get("settings") or {}
                companion_text = str(settings.get("companionText") or "").strip()
                inference_mode = str(settings.get("inferenceMode") or "ensemble").strip().lower()
                if inference_mode not in {"ensemble", "single"}:
                    inference_mode = "ensemble"

                download_dir = Path(tempfile.mkdtemp(prefix="isy_mm_remote_"))
                video_path = download_remote_video(remote_url, download_dir)
                analysis = analyze_multimodal(video_path, selected_mode, companion_text, inference_mode)
                json_response(
                    self,
                    {
                        "ok": True,
                        "analysis": analysis,
                        "service": "local-multimodal-backend",
                        "selectedMode": selected_mode,
                        "inferenceMode": inference_mode,
                        "sourceUrl": remote_url,
                    },
                )
            except Exception as error:  # noqa: BLE001
                json_response(self, {"ok": False, "message": str(error)}, status=500)
            finally:
                if download_dir is not None:
                    shutil.rmtree(download_dir, ignore_errors=True)
            return

        if self.path == "/analyze-video-url":
            download_dir: Path | None = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                remote_url = validate_remote_video_url(str(payload.get("url") or ""))
                download_dir = Path(tempfile.mkdtemp(prefix="isy_video_remote_"))
                video_path = download_remote_video(remote_url, download_dir)
                analysis = analyze_video_ensemble(video_path)
                json_response(
                    self,
                    {
                        "ok": True,
                        "analysis": analysis,
                        "service": "local-video-ensemble-backend",
                        "selectedMode": "video-efficientnet-n7",
                        "inferenceMode": "ensemble",
                        "sourceUrl": remote_url,
                    },
                )
            except Exception as error:  # noqa: BLE001
                json_response(self, {"ok": False, "message": str(error)}, status=500)
            finally:
                if download_dir is not None:
                    shutil.rmtree(download_dir, ignore_errors=True)
            return

        if self.path not in {"/analyze", "/analyze-image", "/analyze-video"}:
            json_response(self, {"ok": False, "message": "Not Found"}, status=404)
            return
        upload_dir: Path | None = None
        try:
            ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
            if ctype != "multipart/form-data":
                raise ValueError("multipart/form-data required")

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            if "file" not in form:
                raise ValueError("file is required")

            file_item = form["file"]
            if not getattr(file_item, "filename", ""):
                raise ValueError("file is required")

            raw_mode = str(form.getvalue("selectedMode") or "mm-flava")
            selected_mode = raw_mode if self.path in {"/analyze-image", "/analyze-video"} else normalize_selected_mode(raw_mode)
            settings_raw = str(form.getvalue("settings") or "{}")
            settings = json.loads(settings_raw) if settings_raw else {}
            companion_text = str(settings.get("companionText") or "").strip()
            inference_mode = str(settings.get("inferenceMode") or "ensemble").strip().lower()
            if inference_mode not in {"ensemble", "single"}:
                inference_mode = "ensemble"
            image_scope = str(settings.get("imageScope") or "full-scene").strip().lower()
            if image_scope not in {"full-scene", "face-focus"}:
                image_scope = "full-scene"

            upload_dir = Path(tempfile.mkdtemp(prefix="isy_mm_upload_"))
            upload_path = upload_dir / Path(file_item.filename).name
            with upload_path.open("wb") as stream:
                stream.write(file_item.file.read())

            if self.path == "/analyze-image":
                analysis = analyze_image_runtime(upload_path, selected_mode, image_scope)
            elif self.path == "/analyze-video":
                analysis = analyze_video_ensemble(upload_path)
            else:
                analysis = analyze_multimodal(upload_path, selected_mode, companion_text, inference_mode)
            json_response(
                self,
                {
                    "ok": True,
                    "analysis": analysis,
                    "service": "local-image-backend" if self.path == "/analyze-image" else "local-video-ensemble-backend" if self.path == "/analyze-video" else "local-multimodal-backend",
                    "selectedMode": "video-efficientnet-n7" if self.path == "/analyze-video" else selected_mode,
                    "inferenceMode": "ensemble" if self.path == "/analyze-video" else inference_mode,
                },
            )
        except Exception as error:  # noqa: BLE001
            json_response(self, {"ok": False, "message": str(error)}, status=500)
        finally:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), MultimodalHandler)
    print(f"[multimodal] listening on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
