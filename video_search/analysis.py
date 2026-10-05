import hashlib
import json
import math
import re
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from .config import DATA, DEVICE, MODEL, REVISION, prepare_data
prepare_data()
import av
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import delete, select
from .db import AnalysisJob, CollectionEpisode, Detection, Episode, Target, TargetReference, Track, VideoAsset, VideoSource, session, utcnow
from .detector import COMMON_OBJECTS, Owlv2Adapter, ReferenceMatcher, detect_with_tiles, iou
from .video import absolute_time, extract_frame, location_at


DEFAULTS = {"scan": "sampled", "sample_step_s": 1.0, "refine_step_s": 0.2, "detection_threshold": 0.2, "similarity_threshold": 0.5, "tile_size": None, "tile_overlap": 0.2, "max_gap_s": 1.0, "min_duration_s": 0.0, "include_single": True, "context_before_s": 1.0, "context_after_s": 1.0, "device": DEVICE}
PIPELINE_VERSION = "4"
QUICK_VERSION = "7"
QUICK_FPS = 2
OBJECT_ALIASES = {
    "машина": "car", "машины": "car", "автомобиль": "car", "автомобили": "car", "авто": "car", "легковая машина": "car",
    "человек": "person", "люди": "person", "пешеход": "person", "пешеходы": "person",
    "грузовик": "truck", "грузовики": "truck", "автобус": "bus", "автобусы": "bus",
    "фургон": "van", "фургоны": "van", "велосипед": "bicycle", "велосипеды": "bicycle",
    "мотоцикл": "motorcycle", "мотоциклы": "motorcycle", "лодка": "boat", "лодки": "boat",
    "самолёт": "airplane", "самолет": "airplane", "вертолёт": "helicopter", "вертолет": "helicopter",
    "дрон": "drone", "собака": "dog", "собаки": "dog", "кошка": "cat", "кошки": "cat",
    "птица": "bird", "птицы": "bird", "дерево": "tree", "деревья": "tree",
    "здание": "building", "здания": "building", "палатка": "tent", "палатки": "tent",
    "тент": "awning", "зонт": "umbrella", "зонты": "umbrella",
    "лошадь": "horse", "лошади": "horse", "корова": "cow", "коровы": "cow",
    "рюкзак": "backpack", "рюкзаки": "backpack", "чемодан": "suitcase", "чемоданы": "suitcase",
}


def parse_requested_objects(raw):
    names = [part.strip().lower() for part in re.split(r"[,;]", raw or "") if part.strip()]
    if not names:
        raise ValueError("Укажите один или несколько объектов через запятую")
    labels = []
    for name in names:
        label = OBJECT_ALIASES.get(name, name)
        if not re.fullmatch(r"[a-z][a-z -]{0,39}", label):
            raise ValueError(f"Не знаю объект «{name}». Напишите его название по-английски")
        if label not in labels:
            labels.append(label)
    if len(labels) > 8:
        raise ValueError("Укажите не больше 8 типов объектов за один запуск")
    return sorted(labels)


def enqueue_quick(video_id, objects="", reference_path=None, reference_sha256=None):
    """Queue a visual search for the object types chosen by the user."""
    labels = parse_requested_objects(objects) if (objects or "").strip() else []
    if not labels and not reference_path:
        raise ValueError("Укажите объекты или добавьте фото объекта")
    key = hashlib.sha256(("|".join(labels) + "|" + (reference_sha256 or "")).encode()).hexdigest()[:16]
    target_name = f"__quick_selected_{key}__"
    with session() as db:
        target = db.scalar(select(Target).where(Target.name == target_name))
        if target is None:
            target = Target(name=target_name, mode="category", model_prompt=", ".join(labels) or "visual reference", prompt_confirmed=True)
            db.add(target)
            db.flush()
        target_id = target.id
    return enqueue(video_id, target_id, {"quick_preview": True, "quick_fps": QUICK_FPS,
                                         "quick_version": QUICK_VERSION, "quick_labels": labels,
                                         "quick_reference_path": reference_path,
                                         "quick_reference_sha256": reference_sha256,
                                         "detection_threshold": 0.25})


def _draw_common(image, hits):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, round(image.width / 70)))
    except OSError:
        font = ImageFont.load_default()
    line = max(2, round(image.width / 350))
    for box, score, label in hits:
        x1, y1, x2, y2 = box
        color = "#35f4a0"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        caption = f"{label} {score:.2f}"
        bounds = draw.textbbox((x1, y1), caption, font=font)
        top = max(0, y1 - (bounds[3] - bounds[1]) - 6)
        draw.rectangle((x1, top, x1 + bounds[2] - bounds[0] + 7, top + bounds[3] - bounds[1] + 6), fill="#0d2630")
        draw.text((x1 + 3, top + 2), caption, font=font, fill="white")
    return result


def _run_quick_video(job_id, video, detector, labels, reference_path=None):
    """Encode only frames that were actually inspected, at a fixed review FPS."""
    output = DATA / "annotated" / f"job-{job_id}.mp4"
    temporary = output.with_suffix(".partial.mp4")
    observations_path = DATA / "annotated" / f"job-{job_id}.json"
    observations_temporary = observations_path.with_suffix(".partial.json")
    width, height = video.width + video.width % 2, video.height + video.height % 2
    frame_count = 0
    observations = []
    reference_matcher = None
    if reference_path:
        with Image.open(reference_path) as reference_file:
            reference_matcher = ReferenceMatcher(reference_file)
    try:
        with av.open(str(temporary), "w") as container:
            stream = container.add_stream("libx264", rate=QUICK_FPS)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23", "preset": "veryfast"}
            for seconds, image in selected_frames(video, "sampled", 1 / QUICK_FPS):
                if _check_cancel(job_id):
                    return
                text_hits = detector.detect_common(image, labels=labels, threshold=0.25) if labels else []
                if reference_matcher is not None:
                    photo_hits = reference_matcher.detect(image)
                    if labels:
                        hits = [(box, score, f"{label} + photo") for box, score, label in text_hits
                                if any(iou(box, photo_box) >= 0.25 for photo_box, _, _ in photo_hits)]
                    else:
                        hits = photo_hits
                else:
                    hits = text_hits
                hits.sort(key=lambda item: item[1], reverse=True)
                deduplicated = []
                for hit in hits:
                    if all(iou(hit[0], prior[0]) < 0.5 for prior in deduplicated):
                        deduplicated.append(hit)
                hits = deduplicated[:60]
                observations.append({
                    "frame_index": frame_count,
                    "source_time_s": round(seconds, 6),
                    "output_time_s": frame_count / QUICK_FPS,
                    "detections": [
                        {"label": label, "score": round(score, 6), "bbox_xyxy": [round(v, 2) for v in box]}
                        for box, score, label in hits
                    ],
                })
                annotated = _draw_common(image, hits)
                if annotated.size != (width, height):
                    padded = Image.new("RGB", (width, height))
                    padded.paste(annotated, (0, 0))
                    annotated = padded
                frame = av.VideoFrame.from_image(annotated)
                frame.pts = frame_count
                frame.time_base = Fraction(1, QUICK_FPS)
                for packet in stream.encode(frame):
                    container.mux(packet)
                frame_count += 1
                _progress(job_id, min(0.99, seconds / max(video.duration_s, 0.01)))
            for packet in stream.encode():
                container.mux(packet)
        if not frame_count:
            raise ValueError("video has no decodable frames")
        observations_temporary.write_text(json.dumps(observations, ensure_ascii=False), encoding="utf8")
        temporary.replace(output)
        observations_temporary.replace(observations_path)
        with session() as db:
            done = db.get(AnalysisJob, job_id)
            done.status, done.progress, done.finished_at = "done", 1.0, utcnow()
    finally:
        if temporary.exists():
            temporary.unlink()
        if observations_temporary.exists():
            observations_temporary.unlink()


def validated_config(proposed):
    config = {**DEFAULTS, **(proposed or {})}
    if config["scan"] not in ("full", "sampled"):
        raise ValueError("scan must be full or sampled")
    for key in ("sample_step_s", "refine_step_s", "max_gap_s", "min_duration_s", "context_before_s", "context_after_s"):
        if not isinstance(config[key], (int, float)) or config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if config["sample_step_s"] <= 0 or config["refine_step_s"] <= 0:
        raise ValueError("sample steps must be positive")
    for key in ("detection_threshold", "similarity_threshold", "tile_overlap"):
        if not 0 <= config[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")
    if config["tile_size"] is not None and config["tile_size"] < 64:
        raise ValueError("tile_size must be at least 64")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return config


def enqueue(video_id, target_id, proposed=None):
    config = validated_config(proposed)
    config["pipeline_version"] = PIPELINE_VERSION
    with session() as db:
        video = db.get(VideoAsset, video_id)
        target = db.get(Target, target_id)
        if not video or not target:
            raise ValueError("video or target not found")
        source = db.get(VideoSource, video.source_id)
        if target.mode == "category" and (not target.model_prompt or not target.prompt_confirmed):
            raise ValueError("category target needs a confirmed English model_prompt")
        refs = db.scalars(select(TargetReference).where(TargetReference.target_id == target_id, TargetReference.version == target.version)).all()
        if target.mode == "visual_similarity":
            if not any(not r.negative for r in refs):
                raise ValueError("visual target needs a positive reference")
        payload = [video.sha256, video.original_name, video.start_utc, source.location,
                   source.telemetry, target.id, target.version, target.model_prompt,
                   target.mode, MODEL, REVISION, config]
        cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        old = db.scalar(select(AnalysisJob).where(AnalysisJob.cache_key == cache_key, AnalysisJob.status == "done"))
        if old:
            return old.id, True
        snapshot = {"id": target.id, "name": target.name, "mode": target.mode, "model_prompt": target.model_prompt, "references": [{"id": r.id, "crop_path": r.crop_path, "negative": r.negative} for r in refs]}
        job = AnalysisJob(video_id=video_id, target_id=target_id, target_version=target.version, mode=target.mode, model_id=MODEL, model_revision=REVISION, config=config, target_snapshot=snapshot, cache_key=cache_key)
        db.add(job)
        db.flush()
        return job.id, False


def decoded_frames(path, stream_start_s=0):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = max(0.0, float(frame.pts * stream.time_base) - stream_start_s)
            yield seconds, frame


def selected_frames(video, scan, step, windows=None):
    last = -math.inf
    for seconds, frame in decoded_frames(video.path, video.stream_start_s):
        if windows is not None and not any(a <= seconds <= b for a, b in windows):
            continue
        if scan == "full" or seconds - last + 1e-5 >= step:
            last = seconds
            yield seconds, frame.to_image().convert("RGB")


def merge_windows(times, radius, duration):
    result = []
    for t in sorted(times):
        a, b = max(0, t-radius), min(duration, t+radius)
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def assemble_episodes(observations, config, duration):
    """Pure episode assembly; observations have time, track_id, score."""
    groups = defaultdict(list)
    for row in observations:
        groups[row["track_id"]].append(row)
    episodes = []
    for track_id, rows in groups.items():
        rows.sort(key=lambda r: r["time_s"])
        chunks = []
        for row in rows:
            if not chunks or row["time_s"] - chunks[-1][-1]["time_s"] > config["max_gap_s"]:
                chunks.append([])
            chunks[-1].append(row)
        for chunk in chunks:
            first, last = chunk[0]["time_s"], chunk[-1]["time_s"]
            if last-first < config["min_duration_s"] and not (len(chunk) == 1 and config["include_single"]):
                continue
            scores = [r["score"] for r in chunk if r["score"] is not None]
            episodes.append({"track_id": track_id, "observed_start_s": first, "observed_end_s": last, "start_s": max(0, first-config["context_before_s"]), "end_s": min(duration, max(last, first+0.04)+config["context_after_s"]), "rank_score": max(scores) if scores else None, "rank_rule": "maximum raw model score within observed detections"})
    return sorted(episodes, key=lambda e: (e["start_s"], e["track_id"]))


def _check_cancel(job_id):
    with session() as db:
        job = db.get(AnalysisJob, job_id)
        if job.cancel_requested:
            job.status = "cancelled"
            job.finished_at = utcnow()
            return True
    return False


def _progress(job_id, value):
    with session() as db:
        db.get(AnalysisJob, job_id).progress = min(1.0, value)


def retry_job(job_id):
    with session() as db:
        job = db.get(AnalysisJob, job_id)
        if not job or job.status not in ("failed", "cancelled"):
            raise ValueError("only failed or cancelled jobs can be retried")
        episode_ids = select(Episode.id).where(Episode.job_id == job_id)
        db.execute(delete(CollectionEpisode).where(CollectionEpisode.episode_id.in_(episode_ids)))
        db.execute(delete(Episode).where(Episode.job_id == job_id))
        db.execute(delete(Detection).where(Detection.job_id == job_id))
        db.execute(delete(Track).where(Track.job_id == job_id))
        job.status, job.error, job.progress, job.cancel_requested = "queued", None, 0, False


def recover_interrupted():
    with session() as db:
        for job in db.scalars(select(AnalysisJob).where(AnalysisJob.status == "running")):
            job.status, job.error = "failed", "worker interrupted; use retry to requeue"


def run_job(job_id):
    with session() as db:
        job = db.get(AnalysisJob, job_id)
        if not job or job.status not in ("queued", "failed"):
            return
        job.status, job.started_at, job.progress, job.error = "running", utcnow(), 0, None
        video = db.get(VideoAsset, job.video_id)
        source = db.get(VideoSource, video.source_id)
        target_data = job.target_snapshot
        ref_data = target_data["references"]
        config = job.config
    photo_only = config.get("quick_preview") and config.get("quick_reference_path") and not config.get("quick_labels")
    detector = None if photo_only else Owlv2Adapter()
    try:
        selected_device = "cpu" if photo_only else detector.load(config["device"])
        with session() as db:
            db.get(AnalysisJob, job_id).selected_device = selected_device
        if config.get("quick_preview"):
            _run_quick_video(job_id, video, detector, config.get("quick_labels", COMMON_OBJECTS), config.get("quick_reference_path"))
            return
        detector.prepare_target(target_data, ref_data)
        threshold = config["detection_threshold"] if job.mode == "category" else config["similarity_threshold"]
        if config["scan"] == "sampled":
            candidates = []
            for n, (seconds, image) in enumerate(selected_frames(video, "sampled", config["sample_step_s"])):
                hits = detect_with_tiles(detector, image, threshold, config["tile_size"], config["tile_overlap"])
                if hits:
                    candidates.append(seconds)
                if n % 10 == 0:
                    if _check_cancel(job_id): return
                    _progress(job_id, min(0.4, 0.4*seconds/video.duration_s))
            windows = merge_windows(candidates, config["sample_step_s"], video.duration_s)
            frames = selected_frames(video, "sampled", config["refine_step_s"], windows)
        else:
            frames = selected_frames(video, "full", 0)
        # One tracker for this job. The camera-motion limit is documented.
        tracker = sv.ByteTrack(track_activation_threshold=min(0.25, threshold), minimum_consecutive_frames=1, frame_rate=max(1, round(1 / (config["refine_step_s"] if config["scan"] == "sampled" else 1/30))))
        track_map = {}
        unmatched = -1
        observations = []
        for n, (seconds, image) in enumerate(frames):
            if _check_cancel(job_id): return
            hits = detect_with_tiles(detector, image, threshold, config["tile_size"], config["tile_overlap"])
            valid = [h for h in hits if h.bbox[2] > h.bbox[0] and h.bbox[3] > h.bbox[1]]
            if valid:
                boxes = np.array([h.bbox for h in valid], dtype=np.float32)
                scores = np.array([h.detection_score if h.detection_score is not None else h.similarity_score for h in valid], dtype=np.float32)
                tracked = tracker.update_with_detections(sv.Detections(xyxy=boxes, confidence=scores, class_id=np.zeros(len(valid), dtype=int)))
            else:
                tracked = tracker.update_with_detections(sv.Detections.empty())
            with session() as db:
                for hit in valid:
                    best_id, best_overlap = None, 0
                    for box, _, _, _, tid, _ in tracked:
                        overlap = iou(hit.bbox, box)
                        if tid is not None and overlap > best_overlap:
                            best_id, best_overlap = int(tid), overlap
                    if best_overlap < 0.5:
                        best_id = unmatched
                        unmatched -= 1
                    if best_id not in track_map:
                        track = Track(job_id=job_id, local_id=best_id, target_id=job.target_id)
                        db.add(track)
                        db.flush()
                        track_map[best_id] = track.id
                    row = Detection(job_id=job_id, target_id=job.target_id, track_id=track_map[best_id], time_s=seconds, bbox=list(hit.bbox), detection_score=hit.detection_score, similarity_score=hit.similarity_score, score_origin=hit.score_origin, reference_id=hit.reference_id, observed=True)
                    db.add(row)
                    observations.append({"track_id": track_map[best_id], "time_s": seconds, "score": hit.detection_score if hit.detection_score is not None else hit.similarity_score})
            if n % 10 == 0:
                fraction = seconds / video.duration_s
                _progress(job_id, 0.4+0.6*fraction if config["scan"] == "sampled" else fraction)
        for ep in assemble_episodes(observations, config, video.duration_s):
            preview = DATA / "previews" / f"video-{video.id}-{round(ep['observed_start_s']*1000)}.jpg"
            if not preview.exists():
                extract_frame(video.path, ep["observed_start_s"], preview)
            with session() as db:
                db.add(Episode(job_id=job_id, track_id=ep["track_id"], video_id=video.id, source_id=source.id, target_id=job.target_id, start_s=ep["start_s"], end_s=ep["end_s"], observed_start_s=ep["observed_start_s"], observed_end_s=ep["observed_end_s"], rank_score=ep["rank_score"], rank_rule=ep["rank_rule"], preview_path=str(preview), start_utc=absolute_time(video.start_utc, ep["observed_start_s"]), end_utc=absolute_time(video.start_utc, ep["observed_end_s"]), location=location_at(source, ep["observed_start_s"])))
        with session() as db:
            done = db.get(AnalysisJob, job_id)
            done.status, done.progress, done.finished_at = "done", 1.0, utcnow()
    except Exception as exc:
        with session() as db:
            failed = db.get(AnalysisJob, job_id)
            failed.status, failed.error, failed.finished_at = "failed", f"{type(exc).__name__}: {exc}", utcnow()
        raise
    finally:
        if detector is not None:
            detector.unload()


def worker_loop(poll_s=2):
    import fcntl
    with (DATA / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another worker is already running") from exc
        recover_interrupted()
        while True:
            with session() as db:
                job = db.scalar(select(AnalysisJob).where(AnalysisJob.status == "queued").order_by(AnalysisJob.id))
                job_id = job.id if job else None
            if job_id is None:
                time.sleep(poll_s)
                continue
            try:
                run_job(job_id)
            except Exception as exc:
                print(f"job {job_id} failed: {exc}", flush=True)
