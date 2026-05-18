from __future__ import annotations

import importlib.util
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_FRAME_COUNT = 6
MAX_VIDEO_BYTES = 350 * 1024 * 1024


@dataclass(frozen=True)
class VideoCheckpointSpec:
    name: str
    image_size: int
    relative_path: str

    @property
    def path(self) -> Path:
        return VIDEO_DIR / self.relative_path


CHECKPOINT_SPECS = [
    VideoCheckpointSpec(
        "robustaug",
        224,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt",
    ),
    VideoCheckpointSpec(
        "robustaug_ema",
        224,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_frame/best.pt",
    ),
    VideoCheckpointSpec(
        "ff2f_holdout",
        224,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout_frame/best.pt",
    ),
    VideoCheckpointSpec(
        "seed1337",
        224,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame/best.pt",
    ),
    VideoCheckpointSpec(
        "seed7",
        224,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed7_frame/best.pt",
    ),
    VideoCheckpointSpec(
        "df_holdout",
        224,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_df_holdout_frame/best.pt",
    ),
    VideoCheckpointSpec(
        "img320",
        320,
        "checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_img320_frame/best.pt",
    ),
]


def _load_builder_module():
    builder_path = VIDEO_DIR / "builder.py"
    spec = importlib.util.spec_from_file_location("isy_video_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load video builder from {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_video_assets() -> list[Path]:
    missing = [spec.path for spec in CHECKPOINT_SPECS if not spec.path.exists()]
    if not (VIDEO_DIR / "builder.py").exists():
        missing.append(VIDEO_DIR / "builder.py")
    return missing


def _make_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def apply_text_mask(rgb_np: np.ndarray) -> np.ndarray:
    h = rgb_np.shape[0]
    out = rgb_np.copy()
    top = int(round(h * 0.08))
    bottom = int(round(h * 0.18))
    if top > 0:
        median_top = np.median(out[:top].reshape(-1, 3), axis=0).astype(np.uint8)
        out[:top] = median_top
    if bottom > 0:
        median_bottom = np.median(out[-bottom:].reshape(-1, 3), axis=0).astype(np.uint8)
        out[-bottom:] = median_bottom
    return out


def sample_frames(video_path: str | Path, n_frames: int = DEFAULT_FRAME_COUNT) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    try:
        if total > 0:
            indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, bgr = cap.read()
                if ret and bgr is not None:
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            while len(frames) < n_frames:
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise ValueError("No readable frames found in video")
    return frames


def _median_aggregate(probs: np.ndarray) -> np.ndarray:
    out = np.median(probs, axis=0)
    denom = out.sum(axis=1, keepdims=True)
    return out / np.clip(denom, 1e-9, None)


def _confidence_mean_across_frames(probs: np.ndarray) -> np.ndarray:
    max_p = probs.max(axis=1)
    weights = max_p**2
    weights = weights / max(float(weights.sum()), 1e-9)
    return (probs * weights[:, None]).sum(axis=0)


class VideoModelBundle:
    def __init__(self, device: str | torch.device, frame_count: int = DEFAULT_FRAME_COUNT):
        missing = check_video_assets()
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing video model assets: {missing_text}")

        self.device = torch.device(device)
        self.frame_count = int(frame_count)
        self.transforms = {224: _make_transform(224), 320: _make_transform(320)}
        self.models: list[tuple[VideoCheckpointSpec, nn.Module]] = []

        builder = _load_builder_module()
        for spec in CHECKPOINT_SPECS:
            model = builder.build_model(
                backbone="efficientnet_b0",
                num_classes=2,
                pretrained=False,
                dropout=0.4,
                freeze_backbone=False,
                hidden_dim=0,
            )
            checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.models.append((spec, model))

    def _predict_with_tta_hflip(self, model: nn.Module, batch: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            x = batch.to(self.device, non_blocking=True)
            p1 = torch.softmax(model(x), dim=1)
            p2 = torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)
            probs = (p1 + p2) * 0.5
        return probs.cpu().numpy()

    def predict_path(self, video_path: str | Path) -> dict:
        frames = sample_frames(video_path, self.frame_count)
        masked = [apply_text_mask(frame) for frame in frames]

        model_frame_probs = np.zeros((len(self.models), len(masked), 2), dtype=np.float32)
        for model_index, (spec, model) in enumerate(self.models):
            transform = self.transforms[spec.image_size]
            batch = torch.stack(
                [transform(Image.fromarray(frame)) for frame in masked],
                dim=0,
            )
            model_frame_probs[model_index] = self._predict_with_tta_hflip(model, batch)

        median_per_frame = _median_aggregate(model_frame_probs)
        final = _confidence_mean_across_frames(median_per_frame)
        p_real = float(final[0])
        p_gen = float(final[1])

        return {
            "fake_probability": round(p_gen, 4),
            "real_probability": round(p_real, 4),
            "label": "FAKE" if p_gen > 0.5 else "REAL",
            "consistency_score": round(max(p_real, p_gen) * 100),
            "model": "video-efficientnet-b0-n7-median-tta",
            "frames_sampled": len(masked),
            "video_aggregation": "median+confidence_mean",
        }


def load_video_model(device: str | torch.device) -> VideoModelBundle:
    return VideoModelBundle(device=device)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "video"


def download_direct_video(url: str, headers: dict[str, str] | None = None) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="isy-video-url-"))
    target = tmpdir / "input_video"
    try:
        response = requests.get(
            url,
            headers=headers or {},
            timeout=20,
            stream=True,
            allow_redirects=False,
        )
        try:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not (
                content_type.startswith("video/")
                or content_type in {"application/octet-stream", "binary/octet-stream"}
            ):
                raise ValueError(f"Response is not a video (content-type: {content_type})")

            total = 0
            with target.open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        raise ValueError(f"Video is too large: over {MAX_VIDEO_BYTES} bytes")
                    f.write(chunk)
        finally:
            response.close()
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    return target


def download_youtube_video(video_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,32}", video_id or ""):
        raise ValueError("Invalid YouTube video id")

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for YouTube video analysis") from exc

    tmpdir = Path(tempfile.mkdtemp(prefix="isy-youtube-"))
    outtmpl = str(tmpdir / f"{_safe_filename(video_id)}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"

    options = {
        "format": "bv*[height<=480][ext=mp4]/bv*[height<=480]/b[height<=480][ext=mp4]/b[height<=480]/worst",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
        "retries": 1,
        "continuedl": False,
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])

        candidates = [path for path in tmpdir.iterdir() if path.is_file()]
        if not candidates:
            raise RuntimeError("YouTube download did not produce a video file")
        return max(candidates, key=lambda path: path.stat().st_size)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def remove_temp_video(video_path: str | Path) -> None:
    path = Path(video_path)
    tmp_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved = path.resolve()
        if tmp_root not in resolved.parents:
            return
        shutil.rmtree(resolved.parent, ignore_errors=True)
    except Exception:
        return
