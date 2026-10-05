import csv
import hashlib
import io
import json
import math
import shutil
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import select

from .config import DATA
from .db import (AnalysisJob, Collection, CollectionEpisode, Detection, Episode, ReviewDecision,
                 Target, TargetReference, Track, VideoAsset, VideoSource, session)
from .video import (absolute_time, copy_upload, crop_reference, extract_clip, extract_frame,
                    location_at, parse_utc, read_telemetry, validate_telemetry)


def create_target(name, mode, description="", model_prompt=None, prompt_confirmed=False):
    if not name.strip():
        raise ValueError("target name required")
    if mode not in ("category", "visual_similarity"):
        raise ValueError("mode must be category or visual_similarity")
    if mode == "category" and (not model_prompt or not prompt_confirmed):
        raise ValueError("category requires explicit confirmed model_prompt")
    with session() as db:
        target = Target(name=name.strip(), mode=mode, description=description, model_prompt=model_prompt, prompt_confirmed=prompt_confirmed)
        db.add(target)
        db.flush()
        return target.id


def update_target(target_id, **changes):
    with session() as db:
        target = db.get(Target, target_id)
        if not target:
            raise ValueError("target not found")
        old_version = target.version
        for key in ("name", "description", "model_prompt", "prompt_confirmed"):
            if key in changes:
                setattr(target, key, changes[key])
        target.version += 1
        for ref in db.scalars(select(TargetReference).where(TargetReference.target_id == target_id, TargetReference.version == old_version)).all():
            db.add(TargetReference(target_id=target_id, version=target.version, original_path=ref.original_path, crop_path=ref.crop_path, bbox=ref.bbox, video_id=ref.video_id, time_s=ref.time_s, negative=ref.negative))
        return target.version


def add_reference(target_id, image_path, bbox, video_id=None, time_s=None, negative=False):
    with session() as db:
        target = db.get(Target, target_id)
        if not target or target.mode != "visual_similarity":
            raise ValueError("visual target not found")
        token = uuid.uuid4().hex
        original = DATA / "references" / f"{token}-original.jpg"
        crop = DATA / "references" / f"{token}-crop.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.save(original, quality=95)
            box = crop_reference(image, bbox, crop)
        ref = TargetReference(target_id=target_id, version=target.version+1, original_path=str(original), crop_path=str(crop), bbox=box, video_id=video_id, time_s=time_s, negative=negative)
        old = db.scalars(select(TargetReference).where(TargetReference.target_id == target_id, TargetReference.version == target.version)).all()
        target.version += 1
        for prior in old:
            db.add(TargetReference(target_id=target_id, version=target.version, original_path=prior.original_path, crop_path=prior.crop_path, bbox=prior.bbox, video_id=prior.video_id, time_s=prior.time_s, negative=prior.negative))
        db.add(ref)
        db.flush()
        return ref.id


def save_quick_reference(fileobj):
    data = fileobj.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Фото объекта должно быть меньше 20 МБ")
    digest = hashlib.sha256(data).hexdigest()
    path = DATA / "references" / f"quick-{digest}.jpg"
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width < 16 or opened.height < 16 or opened.width * opened.height > 25_000_000:
                raise ValueError("Размер фото объекта не подходит")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if not path.exists():
                temporary = DATA / "references" / f"quick-{digest}-{uuid.uuid4().hex}.partial.jpg"
                try:
                    image.save(temporary, format="JPEG", quality=95)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Не удалось прочитать фото объекта") from exc
    return str(path), digest


def import_video(fileobj, filename, kind, camera_id=None, flight_id=None, start_utc=None, latitude=None, longitude=None, accuracy_m=None, telemetry_path=None, telemetry=None):
    if kind not in ("stationary_camera", "drone"):
        raise ValueError("kind must be stationary_camera or drone")
    if latitude is not None and (not math.isfinite(latitude) or not -90 <= latitude <= 90):
        raise ValueError("invalid latitude")
    if longitude is not None and (not math.isfinite(longitude) or not -180 <= longitude <= 180):
        raise ValueError("invalid longitude")
    if accuracy_m is not None and (not math.isfinite(accuracy_m) or accuracy_m < 0):
        raise ValueError("invalid accuracy_m")
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be supplied together")
    location = {"latitude": latitude, "longitude": longitude, "accuracy_m": accuracy_m} if latitude is not None else None
    utc = parse_utc(start_utc)
    if telemetry is not None and telemetry_path is not None:
        raise ValueError("supply either telemetry samples or telemetry_path")
    telemetry = validate_telemetry(telemetry) if telemetry is not None else read_telemetry(telemetry_path)
    path, checksum, metadata, preview = copy_upload(fileobj, filename)
    with session() as db:
        source = VideoSource(kind=kind, camera_id=camera_id, flight_id=flight_id, location=location, telemetry=telemetry)
        db.add(source)
        db.flush()
        video = VideoAsset(source_id=source.id, original_name=Path(filename).name, path=str(path), sha256=checksum, start_utc=utc, preview_path=str(preview), **metadata)
        db.add(video)
        db.flush()
        return video.id


def episode_dict(db, ep):
    job = db.get(AnalysisJob, ep.job_id)
    video = db.get(VideoAsset, ep.video_id)
    source = db.get(VideoSource, ep.source_id)
    return {"id": ep.id, "job_id": ep.job_id, "video_id": ep.video_id, "source_id": ep.source_id, "source_kind": source.kind, "camera_id": source.camera_id, "flight_id": source.flight_id, "target_id": ep.target_id, "target_version": job.target_version, "model_id": job.model_id, "model_revision": job.model_revision, "track_id": ep.track_id, "start_s": ep.start_s, "end_s": ep.end_s, "observed_start_s": ep.observed_start_s, "observed_end_s": ep.observed_end_s, "start_utc": ep.start_utc, "end_utc": ep.end_utc, "location": ep.location, "rank_score": ep.rank_score, "rank_rule": ep.rank_rule, "status": ep.status, "preview_url": f"/api/episodes/{ep.id}/preview", "video_url": f"/api/videos/{ep.video_id}/media", "original_name": video.original_name}


def list_episodes(target_id=None, source_id=None, job_id=None, status=None, from_utc=None, to_utc=None):
    from_utc = parse_utc(from_utc) if from_utc else None
    to_utc = parse_utc(to_utc) if to_utc else None
    with session() as db:
        query = select(Episode).order_by(Episode.start_s)
        if target_id: query = query.where(Episode.target_id == target_id)
        if source_id: query = query.where(Episode.source_id == source_id)
        if job_id: query = query.where(Episode.job_id == job_id)
        if status: query = query.where(Episode.status == status)
        if from_utc: query = query.where(Episode.start_utc >= from_utc)
        if to_utc: query = query.where(Episode.start_utc <= to_utc)
        return [episode_dict(db, e) for e in db.scalars(query).all()]


def review(episode_id, status, note=""):
    if status not in ("confirmed", "rejected", "unreviewed"):
        raise ValueError("invalid review status")
    with session() as db:
        ep = db.get(Episode, episode_id)
        if not ep: raise ValueError("episode not found")
        ep.status = status
        db.add(ReviewDecision(episode_id=episode_id, status=status, note=note))


def episode_detections(episode_id):
    with session() as db:
        ep = db.get(Episode, episode_id)
        if not ep: raise ValueError("episode not found")
        rows = db.scalars(select(Detection).where(Detection.job_id == ep.job_id, Detection.track_id == ep.track_id, Detection.time_s >= ep.observed_start_s, Detection.time_s <= ep.observed_end_s).order_by(Detection.time_s)).all()
        return [{"time_s": d.time_s, "bbox": d.bbox, "detection_score": d.detection_score, "similarity_score": d.similarity_score, "score_origin": d.score_origin, "observed": d.observed, "reference_id": d.reference_id} for d in rows]


CSV_COLUMNS = ["id", "job_id", "video_id", "source_id", "source_kind", "camera_id", "flight_id", "target_id", "target_version", "model_id", "model_revision", "track_id", "start_s", "end_s", "observed_start_s", "observed_end_s", "start_utc", "end_utc", "location", "rank_score", "rank_rule", "status", "original_name"]


def export_episodes(rows, fmt):
    if fmt == "json":
        return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf8"), "application/json"
    if fmt == "csv":
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in CSV_COLUMNS})
        return stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8"
    raise ValueError("format must be json or csv")


def export_clip(episode_id):
    with session() as db:
        ep = db.get(Episode, episode_id)
        if not ep: raise ValueError("episode not found")
        video = db.get(VideoAsset, ep.video_id)
        out = DATA / "clips" / f"episode-{episode_id}-job-{ep.job_id}-{round(ep.start_s*1000)}-{round(ep.end_s*1000)}.mp4"
        if not out.exists():
            extract_clip(video.path, ep.start_s, ep.end_s, out)
        return out


def quick_archive_metadata(job_id):
    observations_path = DATA / "annotated" / f"job-{job_id}.json"
    if not observations_path.is_file():
        raise ValueError("frame metadata is unavailable for this analysis")
    with session() as db:
        job = db.get(AnalysisJob, job_id)
        if not job or not job.config.get("quick_preview"):
            raise ValueError("analysis not found")
        if job.status != "done":
            raise ValueError("analysis is not complete")
        video = db.get(VideoAsset, job.video_id)
        source = db.get(VideoSource, video.source_id)
        frames = json.loads(observations_path.read_text(encoding="utf8"))
        for frame in frames:
            seconds = frame["source_time_s"]
            frame["timestamp_utc"] = absolute_time(video.start_utc, seconds)
            frame["source_location"] = location_at(source, seconds)
        labels = job.config.get("quick_labels", [])
        has_reference = bool(job.config.get("quick_reference_path"))
        template_reference = has_reference and int(job.config.get("quick_version", 0)) >= 7
        return {
            "schema_version": 2,
            "job_id": job_id,
            "video": {"name": video.original_name, "sha256": video.sha256,
                      "duration_s": video.duration_s, "width": video.width,
                      "height": video.height, "start_utc": video.start_utc},
            "searched_objects": labels,
            "reference_image_sha256": job.config.get("quick_reference_sha256"),
            "model": {"id": job.model_id, "revision": job.model_revision} if labels or has_reference and not template_reference else None,
            "reference_matcher": ({"id": "opencv_tm_ccoeff_normed_multiscale", "version": job.config.get("quick_version")}
                                  if template_reference else {"id": "owlv2_image_guided", "version": job.config.get("quick_version")}
                                  if has_reference else None),
            "sampling_fps": job.config.get("quick_fps", 2),
            "location_note": "source_location is the camera or drone position, not the detected object's position",
            "frames": frames,
        }


def export_quick_archive(job_id):
    video_path = DATA / "annotated" / f"job-{job_id}.mp4"
    if not video_path.is_file():
        raise ValueError("annotated video is unavailable")
    metadata = quick_archive_metadata(job_id)
    archive_path = DATA / "annotated" / f"job-{job_id}.zip"
    temporary = DATA / "annotated" / f"job-{job_id}-{uuid.uuid4().hex}.partial.zip"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.write(video_path, "annotated.mp4", compress_type=zipfile.ZIP_STORED)
            archive.writestr("detections.json", json.dumps(metadata, ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_DEFLATED)
            with session() as db:
                job = db.get(AnalysisJob, job_id)
                reference_path = job.config.get("quick_reference_path")
            if reference_path:
                archive.write(reference_path, "reference.jpg", compress_type=zipfile.ZIP_STORED)
        temporary.replace(archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return archive_path


def collection_rows(collection_id):
    with session() as db:
        links = db.scalars(select(CollectionEpisode).where(CollectionEpisode.collection_id == collection_id)).all()
        return [episode_dict(db, db.get(Episode, link.episode_id)) for link in links]


def export_collection_zip(collection_id):
    rows = collection_rows(collection_id)
    if not rows:
        raise ValueError("collection empty or not found")
    out = DATA / "clips" / f"collection-{collection_id}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        manifest, _ = export_episodes(rows, "json")
        zf.writestr("episodes.json", manifest)
        for row in rows:
            zf.write(export_clip(row["id"]), f"episode-{row['id']}.mp4")
    return out
