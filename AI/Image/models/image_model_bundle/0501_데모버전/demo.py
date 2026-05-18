# 이 파일은 Late Fusion 모델 추론 데모의 진입점입니다
# 사용법 (단일): python demo.py --image samples/real_001.jpg
# 사용법 (폴더): python demo.py --folder fakes/

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from config import MODEL_PATH, IMAGE_PATH
from model import load_model
from preprocess import preprocess_image

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 폴더 안의 이미지 파일 경로를 모두 수집하는 함수
def collect_images(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"[에러] 폴더를 찾을 수 없습니다: {folder_path}")

    image_paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise ValueError(f"[에러] 폴더 안에 이미지 파일이 없습니다: {folder_path}")

    return image_paths


# 단일 이미지에 대해 Real/Fake 추론을 수행하는 함수
# 모델과 device를 인자로 받아서 폴더 모드에서 재사용 가능하도록 분리
def predict_one(model, device: str, image_path: str) -> dict:
    x_rgb, x_fft, crop_status = preprocess_image(image_path)
    x_rgb = x_rgb.to(device)
    x_fft = x_fft.to(device)

    with torch.no_grad():
        logits = model(x_rgb, x_fft)          # (1, 2)
        probs  = F.softmax(logits, dim=1)[0]  # (2,) — [Fake 확률, Real 확률]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    return {
        "label"      : "REAL" if real_prob >= 0.5 else "FAKE",
        "real_prob"  : real_prob,
        "fake_prob"  : fake_prob,
        "crop_status": crop_status,
    }


# 결과를 보기 좋게 출력하는 함수
def print_result(result: dict, image_path: str) -> None:
    bar_len   = 30
    real_fill = int(result["real_prob"] * bar_len)
    fake_fill = bar_len - real_fill

    print(f"  {'─' * 43}")
    print(f"  이미지: {Path(image_path).name}")
    print(f"  얼굴크롭: {result['crop_status']}")
    print(f"  판정:   {result['label']}")
    print(f"  Real:   {'█' * real_fill}{'░' * fake_fill}  {result['real_prob']:.1%}")
    print(f"  Fake:   {'█' * fake_fill}{'░' * real_fill}  {result['fake_prob']:.1%}")


# 폴더 전체 결과 요약을 출력하는 함수
def print_summary(results: list[dict]) -> None:
    total    = len(results)
    n_real   = sum(1 for r in results if r["label"] == "REAL")
    n_fake   = total - n_real
    avg_real = sum(r["real_prob"] for r in results) / total
    avg_fake = sum(r["fake_prob"] for r in results) / total

    print()
    print("=" * 45)
    print("  전체 요약")
    print("=" * 45)
    print(f"  총 이미지 : {total}장")
    print(f"  REAL 판정 : {n_real}장  ({n_real/total:.1%})")
    print(f"  FAKE 판정 : {n_fake}장  ({n_fake/total:.1%})")
    print(f"  평균 Real 확률 : {avg_real:.1%}")
    print(f"  평균 Fake 확률 : {avg_fake:.1%}")
    print("=" * 45)


# 단일 이미지 모드 실행
def run_single(model_path: str, image_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[정보] 디바이스: {device}")
    print(f"[정보] 가중치 로드 중: {model_path}")
    model = load_model(model_path, device)

    print(f"[정보] 추론 중: {image_path}")
    result = predict_one(model, device, image_path)

    print()
    print("=" * 45)
    print_result(result, image_path)
    print(f"  {'─' * 43}")


# 폴더 전체 이미지 배치 추론 모드 실행
def run_folder(model_path: str, folder_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[정보] 디바이스: {device}")
    print(f"[정보] 가중치 로드 중: {model_path}")
    model = load_model(model_path, device)  # 모델은 한 번만 로드

    image_paths = collect_images(folder_path)
    print(f"[정보] {len(image_paths)}장 발견 → 추론 시작\n")
    print("=" * 45)

    all_results = []
    for img_path in image_paths:
        try:
            result = predict_one(model, device, str(img_path))
            print_result(result, str(img_path))
            all_results.append(result)
        except Exception as e:
            print(f"  [건너뜀] {img_path.name} — {e}")

    print_summary(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCap Late Fusion 추론 데모")
    parser.add_argument("--model",  type=str, default=MODEL_PATH, help="가중치 파일 경로")
    parser.add_argument("--image",  type=str, default=None,       help="단일 이미지 경로")
    parser.add_argument("--folder", type=str, default=None,       help="폴더 경로 (전체 추론)")
    args = parser.parse_args()

    # --folder 우선, 없으면 --image, 둘 다 없으면 config.py 기본값(단일) 사용
    if args.folder:
        run_folder(args.model, args.folder)
    else:
        run_single(args.model, args.image or IMAGE_PATH)