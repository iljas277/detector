import argparse
import json
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from .analysis import enqueue, retry_job, run_job, worker_loop
from .config import DATA, DEVICE
from .db import AnalysisJob, Target, TargetReference, VideoAsset, init_db, session
from .detector import Owlv2Adapter, download_weights
from .service import (add_reference, create_target, export_clip, export_episodes,
                      import_video, list_episodes)


def main():
    parser = argparse.ArgumentParser(prog="video-search")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    weights = sub.add_parser("weights")
    weights.add_argument("action", choices=["download", "status"])
    target = sub.add_parser("target")
    target.add_argument("name")
    target.add_argument("--mode", choices=["category", "visual_similarity"], required=True)
    target.add_argument("--prompt")
    target.add_argument("--confirm-prompt", action="store_true")
    reference = sub.add_parser("reference")
    reference.add_argument("target_id", type=int)
    reference.add_argument("image")
    reference.add_argument("--bbox", nargs=4, type=float, required=True)
    reference.add_argument("--negative", action="store_true")
    imp = sub.add_parser("import")
    imp.add_argument("video")
    imp.add_argument("--kind", choices=["stationary_camera", "drone"], required=True)
    imp.add_argument("--camera-id")
    imp.add_argument("--flight-id")
    imp.add_argument("--start-utc")
    imp.add_argument("--latitude", type=float)
    imp.add_argument("--longitude", type=float)
    imp.add_argument("--accuracy-m", type=float)
    imp.add_argument("--telemetry")
    image = sub.add_parser("image")
    image.add_argument("target_id", type=int)
    image.add_argument("path")
    image.add_argument("--threshold", type=float, default=0.2)
    image.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("video_id", type=int)
    analyze.add_argument("target_id", type=int)
    analyze.add_argument("--scan", choices=["full", "sampled"], default="sampled")
    analyze.add_argument("--step", type=float, default=1.0)
    analyze.add_argument("--refine-step", type=float, default=0.2)
    analyze.add_argument("--threshold", type=float, default=0.2)
    analyze.add_argument("--tile-size", type=int)
    analyze.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEVICE)
    analyze.add_argument("--wait", action="store_true")
    sub.add_parser("worker")
    jobs = sub.add_parser("jobs")
    retry = sub.add_parser("retry")
    retry.add_argument("job_id", type=int)
    sub.add_parser("episodes")
    export = sub.add_parser("export")
    export.add_argument("--format", choices=["json", "csv", "mp4"], required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--episode-id", type=int)
    export.add_argument("--job-id", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("annotations")
    evaluate.add_argument("--temporal-iou", type=float, default=0.5)
    args = parser.parse_args()
    init_db()
    if args.command == "init":
        print(f"database: {DATA / 'archive.sqlite3'}")
    elif args.command == "weights":
        if args.action == "download": print(json.dumps(download_weights(), indent=2))
        else: print("present" if (DATA / "models/owlv2/config.json").exists() and (DATA / "models/owlv2/model.safetensors").exists() else "absent")
    elif args.command == "target":
        print(create_target(args.name, args.mode, model_prompt=args.prompt, prompt_confirmed=args.confirm_prompt))
    elif args.command == "reference":
        print(add_reference(args.target_id, args.image, args.bbox, negative=args.negative))
    elif args.command == "import":
        with open(args.video, "rb") as f:
            print(import_video(f, Path(args.video).name, args.kind, args.camera_id, args.flight_id, args.start_utc, args.latitude, args.longitude, args.accuracy_m, args.telemetry))
    elif args.command == "image":
        with session() as db:
            t = db.get(Target, args.target_id)
            if not t: parser.error("target not found")
            refs = db.scalars(select(TargetReference).where(TargetReference.target_id == t.id, TargetReference.version == t.version)).all()
            target_data = {"id": t.id, "mode": t.mode, "model_prompt": t.model_prompt}
            ref_data = [{"id": r.id, "crop_path": r.crop_path, "negative": r.negative} for r in refs]
        detector = Owlv2Adapter()
        print(f"selected device: {detector.load(args.device)}")
        detector.prepare_target(target_data, ref_data)
        with Image.open(args.path) as im:
            for h in detector.detect(im.convert("RGB"), args.threshold): print(json.dumps(h.__dict__))
        detector.unload()
    elif args.command == "analyze":
        config = {"scan": args.scan, "sample_step_s": args.step, "refine_step_s": args.refine_step, "detection_threshold": args.threshold, "similarity_threshold": args.threshold, "tile_size": args.tile_size, "device": args.device}
        job_id, cached = enqueue(args.video_id, args.target_id, config)
        print(json.dumps({"job_id": job_id, "cached": cached}))
        if args.wait and not cached: run_job(job_id)
    elif args.command == "worker": worker_loop()
    elif args.command == "jobs":
        with session() as db:
            for j in db.scalars(select(AnalysisJob).order_by(AnalysisJob.id)):
                print(json.dumps({"id": j.id, "status": j.status, "progress": j.progress, "error": j.error, "device": j.selected_device}))
    elif args.command == "retry":
        retry_job(args.job_id)
        print(f"job {args.job_id} queued")
    elif args.command == "episodes":
        print(json.dumps(list_episodes(), ensure_ascii=False, indent=2))
    elif args.command == "export":
        if args.format == "mp4":
            if not args.episode_id: parser.error("mp4 requires --episode-id")
            import shutil
            shutil.copy2(export_clip(args.episode_id), args.output)
        else:
            rows = list_episodes(job_id=args.job_id)
            Path(args.output).write_bytes(export_episodes(rows, args.format)[0])
        print(args.output)
    elif args.command == "evaluate":
        from .evaluate import evaluate_file
        print(json.dumps(evaluate_file(args.annotations, args.temporal_iou), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
