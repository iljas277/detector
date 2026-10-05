import hashlib
import json
import math
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from .config import DATA, MAX_UPLOAD


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1000:] or f"command failed: {args[0]}")
    return result.stdout


def probe(path):
    data = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,start_time,duration", "-show_entries", "format=duration", "-of", "json", str(path)]))
    stream = data.get("streams", [{}])[0]
    if not stream.get("width") or not stream.get("height"):
        raise ValueError("video has no decodable video stream")
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("video duration is unknown or zero")
    return {"width": int(stream["width"]), "height": int(stream["height"]), "duration_s": duration, "stream_start_s": float(stream.get("start_time") or 0)}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_upload(source, original_name):
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise ValueError("supported video formats: mp4, mov, mkv, avi, webm")
    dest = DATA / "originals" / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    h = hashlib.sha256()
    try:
        with dest.open("wb") as f:
            while block := source.read(1024*1024):
                size += len(block)
                if size > MAX_UPLOAD:
                    raise ValueError("upload exceeds VIDEO_SEARCH_MAX_UPLOAD_MB")
                h.update(block)
                f.write(block)
        metadata = probe(dest)
        preview = DATA / "previews" / f"{dest.stem}.jpg"
        extract_frame(dest, min(1.0, metadata["duration_s"] / 2), preview)
        return dest, h.hexdigest(), metadata, preview
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def extract_frame(video_path, seconds, out_path):
    command(["ffmpeg", "-y", "-v", "error", "-ss", str(max(0, seconds)), "-i", str(video_path), "-frames:v", "1", str(out_path)])
    if not Path(out_path).exists():
        raise RuntimeError("decoder produced no frame")
    return out_path


def extract_clip(video_path, start, end, out_path):
    if end <= start:
        raise ValueError("clip end must exceed start")
    command(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(video_path), "-t", str(end-start), "-map", "0:v:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)])
    return probe(out_path)["duration_s"]


def parse_utc(value):
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("recording start requires timezone offset")
    return dt.astimezone(timezone.utc).isoformat()


def absolute_time(start_utc, seconds):
    return (datetime.fromisoformat(start_utc) + timedelta(seconds=seconds)).isoformat() if start_utc else None


def read_telemetry(path):
    if not path:
        return None
    import csv
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf8"))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf8", newline="") as fh:
            data = list(csv.DictReader(fh))
    else:
        raise ValueError("telemetry must be JSON or CSV")
    return validate_telemetry(data)


def validate_telemetry(data):
    if not isinstance(data, list) or len(data) > 100000:
        raise ValueError("telemetry must be a list of at most 100000 samples")
    normalized = []
    for row in data:
        if not isinstance(row, dict) or "time_s" not in row:
            raise ValueError("telemetry sample missing time_s")
        if "latitude" not in row or "longitude" not in row:
            raise ValueError("telemetry sample missing latitude or longitude")
        try:
            seconds, latitude, longitude = float(row["time_s"]), float(row["latitude"]), float(row["longitude"])
            accuracy = float(row["accuracy_m"]) if row.get("accuracy_m") is not None else None
        except (ValueError, TypeError) as exc:
            raise ValueError("telemetry values must be numbers") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("telemetry time_s must be nonnegative")
        if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("telemetry coordinates out of range")
        if accuracy is not None and (not math.isfinite(accuracy) or accuracy < 0):
            raise ValueError("telemetry accuracy_m must be nonnegative")
        normalized.append({"time_s": seconds, "latitude": latitude, "longitude": longitude, "accuracy_m": accuracy})
    return sorted(normalized, key=lambda row: row["time_s"])


def location_at(source, seconds):
    telemetry = source.telemetry or []
    if telemetry:
        nearest = min(telemetry, key=lambda row: abs(float(row["time_s"])-seconds))
        delta = abs(float(nearest["time_s"])-seconds)
        if delta <= 2:
            return {"kind": "source_position", "coordinate_type": "WGS84", "latitude": float(nearest["latitude"]), "longitude": float(nearest["longitude"]), "origin": "telemetry", "sync_error_s": delta, "accuracy_m": float(nearest["accuracy_m"]) if nearest.get("accuracy_m") else None}
    if source.location:
        return {"kind": "source_position", "coordinate_type": "WGS84", **source.location, "origin": "source_metadata", "sync_error_s": None}
    return None


def crop_reference(image: Image.Image, bbox, dest):
    if len(bbox) != 4:
        raise ValueError("bbox requires x1,y1,x2,y2")
    x1, y1, x2, y2 = [round(float(x)) for x in bbox]
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError("bbox must lie within image")
    image.crop((x1, y1, x2, y2)).convert("RGB").save(dest, format="JPEG", quality=95)
    return [x1, y1, x2, y2]
