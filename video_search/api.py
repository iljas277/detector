import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select

from .analysis import enqueue, enqueue_quick, parse_requested_objects, retry_job, validated_config
from .config import DATA, MAX_UPLOAD, ROOT
from .db import (AnalysisJob, Collection, CollectionEpisode, Episode, Target, TargetReference,
                 VideoAsset, VideoSource, init_db, session)
from .service import (add_reference, collection_rows, create_target, episode_detections,
                      export_clip, export_collection_zip, export_episodes, export_quick_archive, import_video,
                      list_episodes, review, save_quick_reference, update_target)
from .video import extract_frame

app = FastAPI(title="Локальный поиск объектов в видео")
init_db()


def fail(exc):
    raise HTTPException(status_code=400, detail=str(exc)) from exc


class TargetIn(BaseModel):
    name: str
    mode: str
    description: str = ""
    model_prompt: str | None = None
    prompt_confirmed: bool = False


class TargetEdit(BaseModel):
    name: str | None = None
    description: str | None = None
    model_prompt: str | None = None
    prompt_confirmed: bool | None = None


class AnalyzeIn(BaseModel):
    video_id: int
    target_id: int
    config: dict[str, Any] = {}


class ReviewIn(BaseModel):
    status: str
    note: str = ""


class CollectionIn(BaseModel):
    name: str
    description: str = ""


class CollectionAdd(BaseModel):
    episode_id: int


@app.get("/api/health")
def health():
    model_dir = DATA / "models" / "owlv2"
    return {"ok": True, "model_weights_present": (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists()}


@app.get("/api/quick/capabilities")
def quick_capabilities():
    return {"version": 7, "archive": True, "reference_image": True}


@app.post("/api/quick/analyze")
def quick_analyze(file: UploadFile = File(...), objects: str = Form(""), reference_image: UploadFile | None = File(None),
                  start_utc: str | None = Form(None), latitude: float | None = Form(None),
                  longitude: float | None = Form(None), accuracy_m: float | None = Form(None),
                  telemetry_json: str | None = Form(None)):
    try:
        if objects.strip():
            parse_requested_objects(objects)
        if not objects.strip() and reference_image is None:
            raise ValueError("Укажите объекты или добавьте фото объекта")
        telemetry = json.loads(telemetry_json) if telemetry_json else None
        reference_path, reference_sha256 = save_quick_reference(reference_image.file) if reference_image else (None, None)
        video_id = import_video(file.file, file.filename or "video.mp4", "drone",
                                start_utc=start_utc, latitude=latitude, longitude=longitude,
                                accuracy_m=accuracy_m, telemetry=telemetry)
        job_id, cached = enqueue_quick(video_id, objects, reference_path, reference_sha256)
        return {"job_id": job_id, "cached": cached}
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        fail(exc)


@app.get("/api/quick/jobs")
def quick_jobs():
    with session() as db:
        rows = db.scalars(select(AnalysisJob).order_by(AnalysisJob.id.desc())).all()
        result = []
        for job in rows:
            if not job.config.get("quick_preview"):
                continue
            video = db.get(VideoAsset, job.video_id)
            result.append({
                "id": job.id, "name": video.original_name, "status": job.status,
                "progress": job.progress, "error": job.error, "duration_s": video.duration_s,
                "objects": job.config.get("quick_labels", []),
                "has_reference": bool(job.config.get("quick_reference_path")),
                "preview_url": f"/api/videos/{video.id}/preview",
                "video_url": f"/api/quick/jobs/{job.id}/video" if job.status == "done" else None,
                "archive_url": f"/api/quick/jobs/{job.id}/archive" if job.status == "done" and (DATA / "annotated" / f"job-{job.id}.json").is_file() else None,
            })
        return result


@app.get("/api/quick/jobs/{job_id}/video")
def quick_video(job_id: int):
    with session() as db:
        job = db.get(AnalysisJob, job_id)
        if not job or not job.config.get("quick_preview"):
            raise HTTPException(404, "analysis not found")
        if job.status != "done":
            raise HTTPException(409, "video is not ready")
    path = DATA / "annotated" / f"job-{job_id}.mp4"
    if not path.is_file():
        raise HTTPException(404, "annotated video missing")
    return FileResponse(path, media_type="video/mp4", filename=f"objects-{job_id}.mp4", content_disposition_type="inline")


@app.get("/api/quick/jobs/{job_id}/archive")
def quick_archive(job_id: int):
    try:
        path = export_quick_archive(job_id)
    except ValueError as exc:
        fail(exc)
    return FileResponse(path, media_type="application/zip", filename=f"objects-{job_id}.zip")


@app.get("/api/targets")
def targets():
    with session() as db:
        result = []
        for t in db.scalars(select(Target).order_by(Target.id.desc())):
            refs = db.scalars(select(TargetReference).where(TargetReference.target_id == t.id, TargetReference.version == t.version)).all()
            result.append({"id": t.id, "name": t.name, "description": t.description, "mode": t.mode, "model_prompt": t.model_prompt, "prompt_confirmed": t.prompt_confirmed, "version": t.version, "references": [{"id": r.id, "bbox": r.bbox, "negative": r.negative, "time_s": r.time_s, "video_id": r.video_id, "crop_url": f"/api/references/{r.id}/crop"} for r in refs]})
        return result


@app.post("/api/targets")
def targets_create(data: TargetIn):
    try: return {"id": create_target(**data.model_dump())}
    except ValueError as exc: fail(exc)


@app.patch("/api/targets/{target_id}")
def targets_update(target_id: int, data: TargetEdit):
    try: return {"version": update_target(target_id, **data.model_dump(exclude_unset=True))}
    except ValueError as exc: fail(exc)


@app.post("/api/targets/{target_id}/references/image")
async def reference_image(target_id: int, file: UploadFile = File(...), bbox: str = Form(...), negative: bool = Form(False)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "image required")
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        data = await file.read(20*1024*1024+1)
        if len(data) > 20*1024*1024: raise HTTPException(413, "reference image too large")
        tmp.write(data); tmp.flush()
        try: return {"id": add_reference(target_id, tmp.name, json.loads(bbox), negative=negative)}
        except (ValueError, OSError, json.JSONDecodeError) as exc: fail(exc)


@app.post("/api/targets/{target_id}/references/video")
def reference_video(target_id: int, video_id: int = Form(...), time_s: float = Form(...), bbox: str = Form(...), negative: bool = Form(False)):
    with session() as db:
        video = db.get(VideoAsset, video_id)
        if not video: raise HTTPException(404, "video not found")
        if not 0 <= time_s <= video.duration_s: raise HTTPException(400, "time outside video")
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        try:
            extract_frame(video.path, time_s, tmp.name)
            return {"id": add_reference(target_id, tmp.name, json.loads(bbox), video_id=video_id, time_s=time_s, negative=negative)}
        except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc: fail(exc)


@app.get("/api/references/{ref_id}/crop")
def reference_crop(ref_id: int):
    with session() as db:
        ref = db.get(TargetReference, ref_id)
        if not ref: raise HTTPException(404, "reference not found")
        return FileResponse(ref.crop_path, media_type="image/jpeg")


@app.post("/api/videos")
def videos_create(file: UploadFile = File(...), kind: str = Form(...), camera_id: str | None = Form(None), flight_id: str | None = Form(None), start_utc: str | None = Form(None), latitude: float | None = Form(None), longitude: float | None = Form(None), accuracy_m: float | None = Form(None)):
    try: return {"id": import_video(file.file, file.filename or "video.mp4", kind, camera_id, flight_id, start_utc, latitude, longitude, accuracy_m)}
    except (ValueError, RuntimeError) as exc: fail(exc)


@app.get("/api/videos")
def videos():
    with session() as db:
        rows = []
        for v in db.scalars(select(VideoAsset).order_by(VideoAsset.id.desc())):
            source = db.get(VideoSource, v.source_id)
            rows.append({"id": v.id, "source_id": v.source_id, "kind": source.kind, "camera_id": source.camera_id, "flight_id": source.flight_id, "name": v.original_name, "sha256": v.sha256, "width": v.width, "height": v.height, "duration_s": v.duration_s, "start_utc": v.start_utc, "location": source.location, "preview_url": f"/api/videos/{v.id}/preview", "media_url": f"/api/videos/{v.id}/media"})
        return rows


@app.get("/api/videos/{video_id}/preview")
def video_preview(video_id: int):
    with session() as db:
        v = db.get(VideoAsset, video_id)
        if not v: raise HTTPException(404, "video not found")
        return FileResponse(v.preview_path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/media")
def video_media(video_id: int):
    with session() as db:
        v = db.get(VideoAsset, video_id)
        if not v: raise HTTPException(404, "video not found")
        media_type = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}.get(Path(v.path).suffix.lower(), "application/octet-stream")
        return FileResponse(v.path, media_type=media_type, filename=v.original_name, content_disposition_type="inline")


@app.get("/api/videos/{video_id}/frame")
def video_frame(video_id: int, time_s: float = 0):
    with session() as db:
        v = db.get(VideoAsset, video_id)
        if not v: raise HTTPException(404, "video not found")
        if not 0 <= time_s <= v.duration_s: raise HTTPException(400, "time outside video")
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        extract_frame(v.path, time_s, tmp.name)
        tmp.seek(0)
        return Response(tmp.read(), media_type="image/jpeg")


@app.post("/api/jobs")
def jobs_create(data: AnalyzeIn):
    try:
        job_id, cached = enqueue(data.video_id, data.target_id, data.config)
        return {"id": job_id, "cached": cached}
    except ValueError as exc: fail(exc)


@app.get("/api/jobs")
def jobs():
    with session() as db:
        return [{"id": j.id, "video_id": j.video_id, "target_id": j.target_id, "target_version": j.target_version, "status": j.status, "progress": j.progress, "error": j.error, "selected_device": j.selected_device, "config": j.config} for j in db.scalars(select(AnalysisJob).order_by(AnalysisJob.id.desc()))]


@app.post("/api/jobs/{job_id}/cancel")
def jobs_cancel(job_id: int):
    with session() as db:
        j = db.get(AnalysisJob, job_id)
        if not j: raise HTTPException(404, "job not found")
        if j.status == "queued": j.status = "cancelled"
        elif j.status == "running": j.cancel_requested = True
        return {"status": j.status}


@app.post("/api/jobs/{job_id}/retry")
def jobs_retry(job_id: int):
    try: retry_job(job_id); return {"status": "queued"}
    except ValueError as exc: fail(exc)


@app.get("/api/episodes")
def episodes(target_id: int | None = None, source_id: int | None = None, job_id: int | None = None, status: str | None = None, from_utc: str | None = None, to_utc: str | None = None):
    return list_episodes(target_id, source_id, job_id, status, from_utc, to_utc)


@app.get("/api/episodes/{episode_id}/detections")
def episode_boxes(episode_id: int):
    try: return episode_detections(episode_id)
    except ValueError as exc: fail(exc)


@app.get("/api/episodes/{episode_id}/preview")
def episode_preview(episode_id: int):
    with session() as db:
        ep = db.get(Episode, episode_id)
        if not ep: raise HTTPException(404, "episode not found")
        return FileResponse(ep.preview_path, media_type="image/jpeg")


@app.post("/api/episodes/{episode_id}/review")
def episode_review(episode_id: int, data: ReviewIn):
    try: review(episode_id, data.status, data.note); return {"ok": True}
    except ValueError as exc: fail(exc)


@app.get("/api/episodes/{episode_id}/clip")
def episode_clip(episode_id: int):
    try: return FileResponse(export_clip(episode_id), media_type="video/mp4", filename=f"episode-{episode_id}.mp4")
    except ValueError as exc: fail(exc)


@app.get("/api/export/{fmt}")
def export_all(fmt: str, target_id: int | None = None, source_id: int | None = None, job_id: int | None = None, status: str | None = None, from_utc: str | None = None, to_utc: str | None = None):
    try:
        content, media = export_episodes(list_episodes(target_id, source_id, job_id, status, from_utc, to_utc), fmt)
        return Response(content, media_type=media, headers={"Content-Disposition": f"attachment; filename=episodes.{fmt}"})
    except ValueError as exc: fail(exc)


@app.get("/api/collections")
def collections():
    with session() as db:
        return [{"id": c.id, "name": c.name, "description": c.description, "episode_ids": [e.episode_id for e in db.scalars(select(CollectionEpisode).where(CollectionEpisode.collection_id == c.id))]} for c in db.scalars(select(Collection).order_by(Collection.id.desc()))]


@app.post("/api/collections")
def collections_create(data: CollectionIn):
    with session() as db:
        c = Collection(**data.model_dump()); db.add(c); db.flush(); return {"id": c.id}


@app.post("/api/collections/{collection_id}/episodes")
def collections_add(collection_id: int, data: CollectionAdd):
    with session() as db:
        if not db.get(Collection, collection_id) or not db.get(Episode, data.episode_id): raise HTTPException(404, "collection or episode not found")
        if not db.get(CollectionEpisode, (collection_id, data.episode_id)):
            db.add(CollectionEpisode(collection_id=collection_id, episode_id=data.episode_id))
    return {"ok": True}


@app.get("/api/collections/{collection_id}/export")
def collections_export(collection_id: int):
    try: return FileResponse(export_collection_zip(collection_id), media_type="application/zip", filename=f"collection-{collection_id}.zip")
    except ValueError as exc: fail(exc)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
