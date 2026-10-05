from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
import cv2
import numpy as np

from .config import DEVICE, MODEL, REVISION, local_model_path


# OWLv2 is open vocabulary, but a finite prompt list cannot identify every
# possible object. These are useful classes for a first visual pass.
COMMON_OBJECTS = (
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "van",
    "boat", "airplane", "helicopter", "dog", "cat", "bird", "building",
    "tree", "traffic light", "road sign", "backpack", "suitcase",
)


@dataclass
class Hit:
    bbox: tuple[float, float, float, float]
    target_id: int
    detection_score: float | None
    similarity_score: float | None
    score_origin: str
    reference_id: int | None = None


class DetectorAdapter(ABC):
    supported_modes = ("category", "visual_similarity")

    @abstractmethod
    def load(self, device: Literal["auto", "cpu", "cuda"] = "auto") -> str: ...

    @abstractmethod
    def prepare_target(self, target: dict, references: list[dict]) -> None: ...

    @abstractmethod
    def detect(self, image: Image.Image, threshold: float) -> list[Hit]: ...

    @abstractmethod
    def unload(self) -> None: ...


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    area = max(0, right-left) * max(0, bottom-top)
    denom = max(0, a[2]-a[0]) * max(0, a[3]-a[1]) + max(0, b[2]-b[0]) * max(0, b[3]-b[1]) - area
    return area / denom if denom else 0.0


def nms(hits: list[Hit], overlap=0.5) -> list[Hit]:
    ordered = sorted(hits, key=lambda h: h.detection_score if h.detection_score is not None else h.similarity_score or 0, reverse=True)
    kept = []
    for hit in ordered:
        if all(iou(hit.bbox, other.bbox) < overlap for other in kept):
            kept.append(hit)
    return kept


def tile_boxes(width: int, height: int, size: int, overlap: float):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile size and overlap must be valid")
    step = max(1, round(size * (1-overlap)))
    xs = list(range(0, max(1, width-size+1), step))
    ys = list(range(0, max(1, height-size+1), step))
    xs.append(max(0, width-size))
    ys.append(max(0, height-size))
    return [(x, y, min(width, x+size), min(height, y+size)) for y in sorted(set(ys)) for x in sorted(set(xs))]


def detect_with_tiles(detector: DetectorAdapter, image: Image.Image, threshold: float, size: int | None, overlap=0.2):
    if size is None:
        return detector.detect(image, threshold)
    hits = []
    for x1, y1, x2, y2 in tile_boxes(*image.size, size, overlap):
        crop = image.crop((x1, y1, x2, y2))
        for h in detector.detect(crop, threshold):
            hits.append(Hit((h.bbox[0]+x1, h.bbox[1]+y1, h.bbox[2]+x1, h.bbox[3]+y1), h.target_id, h.detection_score, h.similarity_score, h.score_origin, h.reference_id))
    return nms(hits)


class ReferenceMatcher:
    """Conservative search for the appearance in a supplied object crop."""

    def __init__(self, reference: Image.Image):
        self.reference = np.asarray(reference.convert("RGB"))
        # A flat crop has no pattern to locate and produces artificial perfect
        # correlations with TM_CCOEFF_NORMED.
        self.usable = bool(np.max(np.std(self.reference, axis=(0, 1))) >= 8)

    def detect(self, image: Image.Image, threshold: float = 0.8):
        if not self.usable:
            return []
        frame = np.asarray(image.convert("RGB"))
        ref_height, ref_width = self.reference.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        candidates = []
        for scale in np.geomspace(0.12, 1.5, 32):
            width, height = round(ref_width * scale), round(ref_height * scale)
            if min(width, height) < 24 or width > frame_width or height > frame_height:
                continue
            template = cv2.resize(self.reference, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            scores = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if score < threshold:
                    break
                candidates.append(((float(x), float(y), float(x + width), float(y + height)), float(score), "совпадение с фото"))
                scores[max(0, y - height // 2):min(scores.shape[0], y + height // 2 + 1),
                       max(0, x - width // 2):min(scores.shape[1], x + width // 2 + 1)] = -1
        candidates.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for candidate in candidates:
            if all(iou(candidate[0], prior[0]) < 0.3 for prior in kept):
                kept.append(candidate)
            if len(kept) >= 10:
                break
        return kept


class Owlv2Adapter(DetectorAdapter):
    def __init__(self):
        self.model = None
        self.processor = None
        self.target = None
        self.references = []
        self.device = None

    def load(self, device=DEVICE):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = "cuda" if device == "cuda" or device == "auto" and torch.cuda.is_available() else "cpu"
        path = local_model_path()
        if not (path / "config.json").exists():
            raise FileNotFoundError("OWLv2 weights absent; run `python -m video_search.cli weights download` explicitly")
        self.processor = Owlv2Processor.from_pretrained(path, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(path, local_files_only=True).to(self.device).eval()
        return self.device

    def prepare_target(self, target, references):
        if target["mode"] not in self.supported_modes:
            raise ValueError("unsupported target mode")
        if target["mode"] == "category" and not target.get("model_prompt"):
            raise ValueError("category target requires confirmed English model_prompt")
        if target["mode"] == "visual_similarity" and not any(not r.get("negative") for r in references):
            raise ValueError("visual target requires at least one positive crop")
        self.target = target
        self.references = references

    def _run(self, image, query, threshold):
        import torch
        if isinstance(query, str):
            inputs = self.processor(text=[[query]], images=image, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(output, threshold=threshold, target_sizes=torch.tensor([(image.height, image.width)]), text_labels=[[query]])[0]
        else:
            inputs = self.processor(images=image, query_images=query, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self.model.image_guided_detection(**inputs)
            # The library's image-guided post-process returns display alphas,
            # not raw model scores. Preserve sigmoid(max logit) instead.
            raw_scores = torch.sigmoid(output.logits.max(dim=-1).values)[0].tolist()
            raw_boxes = output.target_pred_boxes[0].tolist()
            found = []
            for score, (cx, cy, width, height) in zip(raw_scores, raw_boxes):
                if score < threshold: continue
                found.append(((cx-width/2, cy-height/2, cx+width/2, cy+height/2), float(score)))
            found.sort(key=lambda pair: pair[1], reverse=True)
            kept = []
            for box, score in found:
                if all(iou(box, prior[0]) < 0.3 for prior in kept): kept.append((box, score))
            # OWLv2 pads to a square at max(height, width); its official
            # image-guided post-process scales every coordinate by that size.
            scale = max(image.width, image.height)
            result = [((box[0]*scale, box[1]*scale, box[2]*scale, box[3]*scale), score) for box, score in kept]
        if isinstance(query, str):
            result = [(tuple(map(float, box.tolist())), float(score)) for box, score in zip(result["boxes"], result["scores"])]
        clipped_result = []
        for box, score in result:
            x1, y1, x2, y2 = box
            clipped = (max(0, min(image.width, x1)), max(0, min(image.height, y1)), max(0, min(image.width, x2)), max(0, min(image.height, y2)))
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]: clipped_result.append((clipped, score))
        return clipped_result

    def detect(self, image, threshold):
        if self.model is None or self.target is None:
            raise RuntimeError("model and target must be prepared")
        image = image.convert("RGB")
        target_id = self.target["id"]
        if self.target["mode"] == "category":
            return [Hit(box, target_id, score, None, "owlv2_text_score") for box, score in self._run(image, self.target["model_prompt"], threshold)]
        hits, negatives = [], []
        for ref in self.references:
            with Image.open(ref["crop_path"]) as query:
                found = self._run(image, query.convert("RGB"), threshold)
            for box, score in found:
                if ref.get("negative"):
                    negatives.append((box, score))
                else:
                    hits.append(Hit(box, target_id, None, score, "owlv2_image_guided_sigmoid_max_logit", ref["id"]))
        hits = [h for h in hits if not any(iou(h.bbox, box) >= 0.5 and score >= (h.similarity_score or 0) for box, score in negatives)]
        return nms(hits)

    def detect_common(self, image: Image.Image, labels=None, threshold: float = 0.25):
        """Return (box, score, class) for only the requested object types."""
        import torch
        if self.model is None:
            raise RuntimeError("model must be loaded")
        image = image.convert("RGB")
        labels = list(labels or COMMON_OBJECTS)
        prompts = [f"a photo of {'an' if name[0] in 'aeiou' else 'a'} {name}" for name in labels]
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, threshold=threshold,
            target_sizes=torch.tensor([(image.height, image.width)]),
            text_labels=[labels],
        )[0]
        found = []
        for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            x1, y1, x2, y2 = box.tolist()
            clipped = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
            if clipped[2] - clipped[0] >= 4 and clipped[3] - clipped[1] >= 4:
                found.append((clipped, float(score), label))
        found.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for item in found:
            if all(iou(item[0], prior[0]) < 0.45 for prior in kept):
                kept.append(item)
            if len(kept) >= 60:
                break
        return kept

    def unload(self):
        self.model = None
        self.processor = None
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass


def download_weights():
    import hashlib
    import json
    from huggingface_hub import snapshot_download
    path = local_model_path()
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=path, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    hashes = {}
    for file in path.rglob("*"):
        if file.is_file() and ".cache" not in file.parts and file.name != "manifest.json":
            digest = hashlib.sha256()
            with file.open("rb") as fh:
                for block in iter(lambda: fh.read(1024*1024), b""):
                    digest.update(block)
            hashes[str(file.relative_to(path))] = digest.hexdigest()
    manifest = {"repo": MODEL, "revision": REVISION, "sha256": hashes}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return manifest
