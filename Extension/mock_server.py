"""
ISY 모의 백엔드 서버
실제 모델 없이 확장 프로그램 UI/동작을 테스트할 때 사용.
서버 구조(보안·캐시·CORS·데모 플랫폼)는 server.py와 동일하게 유지.
"""

import hashlib
import ipaddress
import json
import os
import random
import shutil
import socket
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qs, unquote, urljoin

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# ── 경로 설정 ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
BASE_PATH = Path(BASE_DIR)
DEMO_PLATFORM_DIR = BASE_PATH / "demo_platform"
DEMO_UPLOAD_DIR = DEMO_PLATFORM_DIR / "uploads"
DEMO_DISCLOSURES_PATH = DEMO_PLATFORM_DIR / "disclosures.json"
DEMO_STATIC_DIR = DEMO_PLATFORM_DIR / "static"
DEMO_MAX_VIDEO_BYTES = 500 * 1024 * 1024
_platform_disclosure_lock = threading.Lock()

# ── 보안 설정 ────────────────────────────────────────────────
_BLOCKED_HOSTS = (
    "localhost",
    "127.",
    "0.0.0.0",
    "::1",
    "169.254.",
    "100.64.",
    "10.",
    "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
    "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.",
)

MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_REDIRECTS = 3

# ── LRU 캐시 ────────────────────────────────────────────────
_RESULT_CACHE_MAX = 200
_RESULT_CACHE_TTL_SEC = 30 * 60
_result_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_result_cache_lock = threading.Lock()


# ── Mock 비디오 모델 ─────────────────────────────────────────
class _MockVideoModel:
    def predict_path(self, path: str) -> dict:
        fake_prob = random.uniform(0.1, 0.9)
        return {
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(1 - fake_prob, 4),
            "label": "FAKE" if fake_prob > 0.5 else "REAL",
            "consistency_score": round((1 - fake_prob) * 100),
            "model": "mock-v1",
        }

_video_model = _MockVideoModel()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://studio.youtube.com"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── 요청 모델 ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str
    media_type: Optional[str] = None
    page_url: Optional[str] = None

class TextRequest(BaseModel):
    text: str

class PlatformMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    platform: Optional[str] = None
    video_id: Optional[str] = Field(default=None, alias="videoId")

class VideoRequest(BaseModel):
    url: Optional[str] = None
    platform_meta: Optional[PlatformMeta] = None


# ── 보안 함수 ────────────────────────────────────────────────
def _assert_safe_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"허용되지 않는 URL 스킴: {parsed.scheme}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("호스트를 파싱할 수 없습니다")
    if any(host.startswith(b) for b in _BLOCKED_HOSTS):
        raise ValueError(f"내부/사설 주소로의 요청은 허용되지 않습니다: {host}")

    try:
        addr_literal = ipaddress.ip_address(host)
        if (addr_literal.is_private or addr_literal.is_loopback
                or addr_literal.is_link_local or addr_literal.is_reserved
                or addr_literal.is_multicast or addr_literal.is_unspecified):
            raise ValueError(f"내부/사설 IP: {addr_literal}")
        return
    except ValueError as exc:
        if "내부" in str(exc):
            raise

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"DNS 해석 실패: {host}") from e
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            raise ValueError(f"내부/사설 주소로 해석됨: {host} → {addr}")


def resolve_image_url(url: str, page_url: Optional[str] = None) -> tuple:
    parsed = urlparse(url)

    if "ytimg.com" in parsed.netloc:
        clean = urlunparse(parsed._replace(query="", fragment=""))
        return clean, "https://www.youtube.com/"

    if "fbcdn.net" in parsed.netloc or "cdninstagram.com" in parsed.netloc:
        return url, page_url or "https://www.instagram.com/"

    if "pstatic.net" in parsed.netloc and "dthumb" in parsed.path:
        params = parse_qs(parsed.query)
        if "src" in params:
            actual = params["src"][0].strip('"')
            for _ in range(3):
                decoded = unquote(actual)
                if decoded == actual:
                    break
                actual = decoded
            actual = actual.strip('"')
            return actual, "https://www.naver.com/"
        return url, "https://www.naver.com/"

    referer = f"{parsed.scheme}://{parsed.netloc}/"
    return url, referer


def fetch_image(url: str, page_url: Optional[str] = None) -> Image.Image:
    _assert_safe_url(url)
    resolved_url, referer = resolve_image_url(url, page_url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": referer,
    }

    current_url = resolved_url
    resp = None
    for hop in range(MAX_REDIRECTS + 1):
        _assert_safe_url(current_url)
        resp = requests.get(
            current_url,
            headers=headers,
            timeout=12,
            stream=True,
            allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            resp.close()
            if not location:
                raise ValueError("리다이렉트에 location 헤더 없음")
            current_url = urljoin(current_url, location)
            continue
        break
    else:
        if resp is not None:
            resp.close()
        raise ValueError(f"리다이렉트 횟수 초과 ({MAX_REDIRECTS}회)")

    try:
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"이미지가 아닌 응답 (content-type: {content_type})")

        content_length = resp.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_IMAGE_BYTES:
                    raise ValueError(f"이미지 크기 초과: {content_length} bytes")
            except (TypeError, ValueError):
                pass

        buf = BytesIO()
        total = 0
        for chunk in resp.iter_content(64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError(f"이미지 크기 초과 (스트림 중 {total} bytes)")
            buf.write(chunk)
        buf.seek(0)
    finally:
        resp.close()

    img = Image.open(buf).convert("RGB")
    img.load()
    return img


# ── 캐시 함수 ────────────────────────────────────────────────
def _cache_get(key: str) -> Optional[dict]:
    now = time.monotonic()
    with _result_cache_lock:
        entry = _result_cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if now - ts > _RESULT_CACHE_TTL_SEC:
            _result_cache.pop(key, None)
            return None
        _result_cache.move_to_end(key)
        return result


def _cache_put(key: str, result: dict) -> None:
    with _result_cache_lock:
        _result_cache[key] = (time.monotonic(), result)
        _result_cache.move_to_end(key)
        while len(_result_cache) > _RESULT_CACHE_MAX:
            _result_cache.popitem(last=False)


# ── 데모 플랫폼 헬퍼 ────────────────────────────────────────
def _ensure_demo_platform_dirs() -> None:
    DEMO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_STATIC_DIR.mkdir(parents=True, exist_ok=True)


def _load_platform_disclosures() -> dict:
    _ensure_demo_platform_dirs()
    if not DEMO_DISCLOSURES_PATH.exists():
        return {}
    try:
        with DEMO_DISCLOSURES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        traceback.print_exc()
        return {}


def _save_platform_disclosures(data: dict) -> None:
    _ensure_demo_platform_dirs()
    tmp_path = DEMO_DISCLOSURES_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(DEMO_DISCLOSURES_PATH)


def _get_platform_disclosure(video_id: str) -> Optional[dict]:
    with _platform_disclosure_lock:
        return _load_platform_disclosures().get(video_id)


def _put_platform_disclosure(video_id: str, disclosure: dict) -> None:
    with _platform_disclosure_lock:
        data = _load_platform_disclosures()
        data[video_id] = disclosure
        _save_platform_disclosures(data)


def _result_level(fake_probability: float) -> str:
    if fake_probability >= 0.6:
        return "high"
    if fake_probability >= 0.4:
        return "uncertain"
    return "low"


_LEVEL_VIEWER_LABELS: dict[str, tuple[str, str]] = {
    "high":      ("AI 생성 가능성 높음",    ""),
    "uncertain": ("AI 생성 여부 확인 필요", ""),
    "low":       ("AI 생성 가능성 낮음",    "낮게 "),
}


def _build_platform_disclosure(
    *,
    video_id: str,
    filename: str,
    content_type: Optional[str],
    file_size: int,
    stored_filename: str,
    sha256: str,
    result: dict,
) -> dict:
    fake_probability = float(result.get("fake_probability") or 0)
    real_probability = float(result.get("real_probability") or max(0, 1 - fake_probability))
    percent = round(fake_probability * 100)
    level = _result_level(fake_probability)
    viewer_title, adverb = _LEVEL_VIEWER_LABELS.get(level, ("AI 생성 여부 불명확", ""))
    viewer_summary = f"ISY 자동 분석 결과 AI 생성 가능성이 {percent}%로 {adverb}평가되었습니다."

    return {
        "video_id": video_id,
        "filename": filename,
        "content_type": content_type or "video/mp4",
        "file_size": file_size,
        "stored_filename": stored_filename,
        "sha256": sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "ISY Partner Disclosure API",
        "media_type": "video",
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
        "percent": percent,
        "level": level,
        "model": result.get("model") or "mock-v1",
        "raw_label": result.get("label"),
        "viewer_title": viewer_title,
        "viewer_summary": viewer_summary,
    }


# ── Mock 추론 ────────────────────────────────────────────────
def _mock_image_inference(url: str) -> dict:
    keywords = ["ai", "generated", "fake", "synthetic", "midjourney", "dalle"]
    is_suspicious = any(kw in url.lower() for kw in keywords)
    fake_prob = random.uniform(0.6, 0.95) if is_suspicious else random.uniform(0.05, 0.35)
    return {
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(1 - fake_prob, 4),
        "label": "FAKE" if fake_prob > 0.5 else "REAL",
        "consistency_score": round((1 - fake_prob) * 100),
        "crop_status": "mock",
        "model": "mock-v1",
    }


# NOTE: server.py는 텍스트 모델 미구현 상태에서 503/501을 반환하지만,
# mock_server는 "모델 없이 UI/동작을 테스트"하기 위한 도구이므로
# 텍스트도 그럴듯한 가짜 응답을 그대로 유지한다 (의도된 차이).
def _mock_text_inference(text: str) -> dict:
    ko_patterns = [
        "또한", "뿐만 아니라", "더불어", "아울러", "나아가",
        "따라서", "그러므로", "결론적으로", "궁극적으로", "무엇보다도",
        "살펴보겠습니다", "알아보겠습니다", "중요합니다", "필요합니다",
        "다양한 측면", "여러 가지 측면", "이러한 관점", "핵심적인",
        "효과적으로", "체계적으로", "지속적으로", "포괄적으로",
        "심층적으로", "전반적으로", "구체적으로 살펴",
        "이상으로", "정리하자면", "요약하면", "마지막으로",
    ]
    en_patterns = [
        "delve", "certainly", "moreover", "furthermore",
        "it is worth noting", "in conclusion", "as an ai",
        "in today's world", "in the realm of",
    ]
    ko_hits = sum(1 for p in ko_patterns if p in text)
    en_hits = sum(1 for p in en_patterns if p in text.lower())
    total_hits = ko_hits + en_hits
    base = min(0.35 + total_hits * 0.08, 0.95)
    fake_prob = max(0.0, min(1.0, base + random.uniform(-0.05, 0.05)))
    return {
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(1 - fake_prob, 4),
        "consistency_score": round((1 - fake_prob) * 100),
        "text_length": len(text),
        "model": "mock-v1",
    }


# ── 엔드포인트 ───────────────────────────────────────────────
@app.post("/api/analyze/image")
def analyze_image(req: AnalyzeRequest):
    if not req.url:
        raise HTTPException(status_code=400, detail="url 필드가 필요합니다")

    try:
        cache_key, _ = resolve_image_url(req.url, req.page_url)
    except Exception:
        cache_key = req.url

    cached = _cache_get(cache_key)
    if cached is not None:
        return {"url": req.url, "media_type": "image", "from_cache": True, **cached}

    try:
        fetch_image(req.url, req.page_url)
    except requests.RequestException as e:
        print(f"[ISY] 다운로드 실패 ({req.url}): {e}")
        raise HTTPException(status_code=502, detail=f"이미지 다운로드 실패: {e}")
    except Exception as e:
        print(f"[ISY] 이미지 열기 실패 ({req.url}): {e}")
        raise HTTPException(status_code=400, detail=f"이미지 열기 실패: {e}")

    result = _mock_image_inference(req.url)
    _cache_put(cache_key, result)
    return {"url": req.url, "media_type": "image", **result}


@app.post("/api/analyze/text")
async def analyze_text(req: TextRequest):
    return {"media_type": "text", **_mock_text_inference(req.text)}


@app.post("/api/analyze/video")
def analyze_video(req: VideoRequest):
    if not req.url and (not req.platform_meta or not req.platform_meta.video_id):
        raise HTTPException(status_code=400, detail="url 또는 platform_meta.video_id 중 하나가 필요합니다")

    result = _video_model.predict_path("")
    return {"url": req.url, "media_type": "video", **result}


@app.post("/api/analyze/video-file")
def analyze_video_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    temp_dir = tempfile.mkdtemp(prefix="isy-mock-video-")
    temp_path = os.path.join(temp_dir, "upload" + suffix)
    try:
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        result = _video_model.predict_path(temp_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"video file inference failed: {e}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"url": None, "media_type": "video", **result}


@app.post("/api/platform/demo-upload")
def platform_demo_upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}:
        suffix = ".mp4"

    _ensure_demo_platform_dirs()
    temp_dir = tempfile.mkdtemp(prefix="isy-platform-upload-")
    temp_path = os.path.join(temp_dir, "upload" + suffix)
    hasher = hashlib.sha256()
    total = 0
    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > DEMO_MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="video file is too large for the demo platform")
                hasher.update(chunk)
                out.write(chunk)

        sha256 = hasher.hexdigest()
        video_id = sha256[:16]
        stored_filename = f"{video_id}{suffix}"
        stored_path = DEMO_UPLOAD_DIR / stored_filename
        shutil.copyfile(temp_path, stored_path)

        result = _video_model.predict_path(str(stored_path))
        disclosure = _build_platform_disclosure(
            video_id=video_id,
            filename=file.filename or stored_filename,
            content_type=file.content_type,
            file_size=total,
            stored_filename=stored_filename,
            sha256=sha256,
            result=result,
        )
        _put_platform_disclosure(video_id, disclosure)
        return {
            "ok": True,
            "video_id": video_id,
            "watch_url": f"/demo/watch/{video_id}",
            "disclosure": disclosure,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"demo platform upload failed: {e}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/platform/disclosures")
def list_platform_disclosures():
    with _platform_disclosure_lock:
        return list(_load_platform_disclosures().values())


@app.get("/api/platform/disclosures/{video_id}")
def get_platform_disclosure(video_id: str):
    disclosure = _get_platform_disclosure(video_id)
    if not disclosure:
        raise HTTPException(status_code=404, detail="platform disclosure not found")
    return disclosure


@app.get("/api/platform/videos/{video_id}")
def get_platform_video(video_id: str):
    disclosure = _get_platform_disclosure(video_id)
    if not disclosure:
        raise HTTPException(status_code=404, detail="platform disclosure not found")
    stored_filename = disclosure.get("stored_filename")
    if not stored_filename:
        raise HTTPException(status_code=404, detail="video file not found")
    path = DEMO_UPLOAD_DIR / stored_filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="video file not found")
    return FileResponse(path, media_type=disclosure.get("content_type") or "video/mp4")


@app.get("/demo/browse")
def platform_demo_browse_page():
    path = DEMO_STATIC_DIR / "browse.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="demo browse page not found")
    return FileResponse(path)


@app.get("/demo/upload")
def platform_demo_upload_page():
    path = DEMO_STATIC_DIR / "upload.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="demo upload page not found")
    return FileResponse(path)


@app.get("/demo/watch/{video_id}")
def platform_demo_watch_page(video_id: str):
    path = DEMO_STATIC_DIR / "watch.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="demo watch page not found")
    return FileResponse(path)


@app.get("/demo/static/{filename}")
def platform_demo_static(filename: str):
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(status_code=400, detail="invalid static path")
    base = DEMO_STATIC_DIR.resolve()
    resolved = (DEMO_STATIC_DIR / filename).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid static path")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="static file not found")
    return FileResponse(resolved)


if __name__ == "__main__":
    import uvicorn
    print("ISY 모의 서버 시작: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
