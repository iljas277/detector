import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("VIDEO_SEARCH_DATA", str(ROOT / "data-gpu"))).resolve()
os.environ.setdefault("MPLCONFIGDIR", str(DATA / "matplotlib"))
MODEL = os.getenv("VIDEO_SEARCH_MODEL", "google/owlv2-base-patch16-ensemble")
REVISION = os.getenv("VIDEO_SEARCH_REVISION", "57beb61adb5abda3de4a9796bc35ae60bc4b9802")
DEVICE = os.getenv("VIDEO_SEARCH_DEVICE", "cuda")
MAX_UPLOAD = int(os.getenv("VIDEO_SEARCH_MAX_UPLOAD_MB", "2048")) * 1024 * 1024


def prepare_data():
    for part in ("originals", "references", "previews", "clips", "annotated", "models", "matplotlib"):
        (DATA / part).mkdir(parents=True, exist_ok=True)


def local_model_path():
    return DATA / "models" / "owlv2"
