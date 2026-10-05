import json
from collections import defaultdict

from sqlalchemy import select

from .db import AnalysisJob, VideoAsset, session
from .detector import iou
from .service import episode_detections, list_episodes


def temporal_iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0


def score_group(truth, predicted, duration_hours, criterion):
    candidates = []
    for pi, pred in enumerate(predicted):
        for ti, target in enumerate(truth):
            if (pred["video_id"], pred["target_id"]) != (target["video_id"], target["target_id"]): continue
            a = (pred["observed_start_s"], max(pred["observed_end_s"], pred["observed_start_s"]+0.04))
            b = (target["start_s"], target["end_s"])
            overlap = temporal_iou(a, b)
            if overlap >= criterion: candidates.append((overlap, pi, ti))
    used_p, used_t, errors, box_ious = set(), set(), [], []
    for _, pi, ti in sorted(candidates, reverse=True):
        if pi in used_p or ti in used_t: continue
        used_p.add(pi); used_t.add(ti)
        pred, target = predicted[pi], truth[ti]
        errors.extend((abs(pred["observed_start_s"]-target["start_s"]), abs(pred["observed_end_s"]-target["end_s"])))
        if target.get("boxes"):
            detections = episode_detections(pred["id"])
            for annotated in target["boxes"]:
                if not detections: continue
                nearest = min(detections, key=lambda d: abs(d["time_s"]-annotated["time_s"]))
                if abs(nearest["time_s"]-annotated["time_s"]) <= annotated.get("tolerance_s", 0.2):
                    box_ious.append(iou(nearest["bbox"], annotated["bbox"]))
    tp, fp, fn = len(used_p), len(predicted)-len(used_p), len(truth)-len(used_t)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None, "false_episodes_per_hour": fp/duration_hours if duration_hours else None, "boundary_mae_s": sum(errors)/len(errors) if errors else None, "mean_box_iou": sum(box_ious)/len(box_ious) if box_ious else None, "box_samples": len(box_ious)}


def evaluate_file(path, criterion=0.5):
    """Input: {videos:[{video_id,duration_s,split,source_kind}], episodes:[{video_id,target_id,start_s,end_s,mode,boxes?}], job_ids?:[]}.

    Only test split is scored. Split is assigned by whole video/source, not frame.
    """
    data = json.loads(open(path, encoding="utf8").read())
    if not 0 < criterion <= 1: raise ValueError("temporal IoU criterion must be in (0,1]")
    videos = {v["video_id"]: v for v in data["videos"] if v["split"] == "test"}
    if len(videos) != sum(v["split"] == "test" for v in data["videos"]): raise ValueError("duplicate test video IDs")
    truth = [e for e in data["episodes"] if e["video_id"] in videos]
    predicted = [e for e in list_episodes() if e["video_id"] in videos and (not data.get("job_ids") or e["job_id"] in data["job_ids"])]
    with session() as db:
        jobs = {j.id: j for j in db.scalars(select(AnalysisJob)).all()}
        assets = {v.id: v for v in db.scalars(select(VideoAsset)).all()}
    eligible_jobs = [j for j in jobs.values() if j.status == "done" and j.video_id in videos and (not data.get("job_ids") or j.id in data["job_ids"])]
    def label(e):
        if "id" in e:
            return e["source_kind"], jobs[e["job_id"]].mode
        return videos[e["video_id"]]["source_kind"], e["mode"]
    groups = defaultdict(lambda: {"truth": [], "pred": [], "videos": set()})
    for e in truth: groups[label(e)]["truth"].append(e)
    for e in predicted: groups[label(e)]["pred"].append(e)
    for job in eligible_jobs:
        groups[(videos[job.video_id]["source_kind"], job.mode)]["videos"].add(job.video_id)
    report = {}
    for (kind, mode), group in groups.items():
        hours = sum(videos[vid]["duration_s"] for vid in group["videos"])/3600
        report[f"{kind}/{mode}"] = score_group(group["truth"], group["pred"], hours, criterion)
    runs = []
    for job in sorted(eligible_jobs, key=lambda j: j.id):
        job_id, video = job.id, assets[job.video_id]
        elapsed = (job.finished_at-job.started_at).total_seconds() if job.finished_at and job.started_at else None
        runs.append({"job_id": job_id, "device": job.selected_device, "resolution": [video.width, video.height], "config": job.config, "processing_seconds_per_video_hour": elapsed*3600/video.duration_s if elapsed else None})
    return {"temporal_iou_criterion": criterion, "by_source_and_mode": report, "runs": runs, "note": "Null means insufficient observations; no model quality is inferred without held-out real annotations."}
