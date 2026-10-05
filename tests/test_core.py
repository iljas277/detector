import io
import subprocess

from PIL import Image

from video_search.analysis import assemble_episodes, merge_windows, parse_requested_objects, selected_frames
from video_search.detector import Hit, ReferenceMatcher, detect_with_tiles, tile_boxes
from video_search.service import export_episodes, save_quick_reference
from video_search.video import absolute_time, extract_clip, location_at, parse_utc, probe, validate_telemetry


class FakeDetector:
    def detect(self, image, threshold):
        # This only tests coordinate conversion, not model quality.
        return [Hit((1, 2, image.width-1, image.height-2), 7, 0.8, None, "test")]


def test_tiling_maps_to_original_pixels():
    image = Image.new("RGB", (140, 100))
    boxes = tile_boxes(140, 100, 80, 0)
    assert (60, 20, 140, 100) in boxes
    hits = detect_with_tiles(FakeDetector(), image, 0.2, 80, 0)
    assert all(0 <= h.bbox[0] < h.bbox[2] <= 140 for h in hits)
    assert any(h.bbox[0] >= 60 for h in hits)


def test_photo_matcher_finds_crop_and_rejects_unrelated_frame():
    import numpy as np

    rng = np.random.default_rng(7)
    crop = Image.fromarray(rng.integers(0, 256, (70, 50, 3), dtype=np.uint8))
    frame = Image.new("RGB", (320, 240), "#438d4b")
    frame.paste(crop, (120, 80))
    matcher = ReferenceMatcher(crop)
    hits = matcher.detect(frame)
    assert len(hits) == 1
    assert all(abs(actual - expected) <= 2 for actual, expected in zip(hits[0][0], (120, 80, 170, 150)))
    assert matcher.detect(Image.new("RGB", frame.size, "#438d4b")) == []
    assert ReferenceMatcher(Image.new("RGB", (50, 70), "green")).detect(frame) == []


def test_single_observation_and_gap():
    config = {"max_gap_s": 0.5, "min_duration_s": 2, "include_single": True, "context_before_s": 1, "context_after_s": 1}
    observations = [{"track_id": 1, "time_s": 0.2, "score": 0.7}, {"track_id": 1, "time_s": 4, "score": 0.9}]
    rows = assemble_episodes(observations, config, 6)
    assert len(rows) == 2
    assert rows[0]["start_s"] == 0
    assert rows[1]["observed_start_s"] == 4
    assert merge_windows([1, 1.5, 5], 1, 10) == [(0, 2.5), (4, 6)]


def test_time_and_csv_unknown_values():
    assert absolute_time(None, 20) is None
    assert absolute_time(parse_utc("2026-10-04T12:00:00+03:00"), 20).startswith("2026-10-04T09:00:20")
    row = {"id": 1, "location": {"kind": "source_position", "latitude": 1.2}}
    csv_bytes, _ = export_episodes([row], "csv")
    assert b"source_position" in csv_bytes


def test_requested_objects_are_canonical_and_limited():
    assert parse_requested_objects("машины, человек, car") == ["car", "person"]
    assert parse_requested_objects("drone; палатка") == ["drone", "tent"]
    import pytest
    with pytest.raises(ValueError, match="по-английски"):
        parse_requested_objects("неизвестная вещь")
    with pytest.raises(ValueError, match="не больше 8"):
        parse_requested_objects("car,dog,cat,truck,bus,boat,bird,tree,person")


def test_telemetry_location_is_source_position():
    samples = validate_telemetry([
        {"time_s": 3, "latitude": 55.7, "longitude": 37.6, "accuracy_m": 4},
        {"time_s": 1, "latitude": 55.6, "longitude": 37.5},
    ])
    source = type("Source", (), {"telemetry": samples, "location": None})()
    assert location_at(source, 3.2)["latitude"] == 55.7
    assert location_at(source, 3.2)["kind"] == "source_position"
    assert location_at(source, 9) is None
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        validate_telemetry([{"time_s": 0, "latitude": 95, "longitude": 1}])


def test_quick_reference_accepts_photo_and_reuses_same_content(tmp_path, monkeypatch):
    import video_search.service as service
    monkeypatch.setattr(service, "DATA", tmp_path)
    (tmp_path / "references").mkdir()
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(stream, format="PNG")
    path, digest = save_quick_reference(io.BytesIO(stream.getvalue()))
    again, same_digest = save_quick_reference(io.BytesIO(stream.getvalue()))
    assert path == again and digest == same_digest
    assert Image.open(path).size == (32, 32)


def test_pts_and_reencoded_clip(tmp_path):
    video = tmp_path / "vfr.mp4"
    # select='not(mod(n,2))' makes an irregular PTS timeline after setpts.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=2", "-vf", "select=not(eq(n\\,5)),setpts=PTS-STARTPTS", "-fps_mode", "vfr", "-c:v", "libx264", str(video)], check=True)
    meta = probe(video)
    times = [t for t, _ in selected_frames(type("V", (), {"path": str(video), "stream_start_s": 0})(), "full", 0)]
    assert len(times) >= 15 and times == sorted(times)
    assert times[-1] > 1
    clip = tmp_path / "clip.mp4"
    actual = extract_clip(video, 0.3, 1.2, clip)
    assert clip.exists() and abs(actual-0.9) < 0.25
