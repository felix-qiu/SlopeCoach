from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from slopecoach_ml.benchmark.biomechanics_debug import (
    _containing_turn,
    _draw_temporal_skeleton,
    write_biomechanics_overlay_video,
)
from slopecoach_ml.cli.__main__ import _temporal_collector, build_parser


class FakeCanvas:
    def __init__(self, marker=b"frame", shape=(480, 640, 3)):
        self.marker = marker
        self.shape = shape


class FakeWriter:
    def __init__(self, *, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.marker)

    def release(self):
        self.released = True


class FakeCV2:
    IMREAD_COLOR = 1
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self, *, writer_opened=True):
        self.lines = []
        self.circles = []
        self.rectangles = []
        self.writer = FakeWriter(opened=writer_opened)

    def imdecode(self, encoded, _mode):
        return None if encoded == b"bad" else FakeCanvas(encoded)

    def VideoWriter_fourcc(self, *_codec):
        return 1234

    def VideoWriter(self, *_args):
        return self.writer

    def resize(self, canvas, dimensions):
        return FakeCanvas(canvas.marker, (dimensions[1], dimensions[0], 3))

    def rectangle(self, *args):
        self.rectangles.append(args)

    def line(self, _canvas, start, end, color, thickness, line_type=None):
        self.lines.append((start, end, color, thickness, line_type))

    def circle(self, _canvas, center, radius, color, thickness, line_type=None):
        self.circles.append((center, radius, color, thickness, line_type))

    def putText(self, *_args):
        return None


class FakeNumpy:
    uint8 = object()

    @staticmethod
    def frombuffer(encoded, dtype):
        assert dtype is FakeNumpy.uint8
        return encoded


def _install_video_fakes(monkeypatch, *, writer_opened=True):
    cv2 = FakeCV2(writer_opened=writer_opened)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy)
    return cv2


def _joint(provenance, *, raw=(10, 20), stabilized=(11, 21)):
    return {
        "provenance": provenance,
        "raw_x_px": raw[0],
        "raw_y_px": raw[1],
        "stabilized_x_px": stabilized[0],
        "stabilized_y_px": stabilized[1],
    }


def _report():
    return {
        "_upstream_debug_report": {
            "temporal_trace": [
                {
                    "timestamp_us": 200_000,
                    "frame_index": 2,
                    "target_id": "target-1",
                    "active_track_id": None,
                    "identity_state": "LOST",
                    "joints": {},
                },
                {
                    "timestamp_us": 100_000,
                    "frame_index": 1,
                    "target_id": "target-1",
                    "active_track_id": 7,
                    "identity_state": "LOCKED",
                    "joints": {},
                },
            ],
            "turn_signal_samples": [
                {"timestamp_us": 100_000, "value": 0.2},
                {"timestamp_us": 200_000, "value": None},
            ],
        },
        "turn_segments": [
            {
                "turn_id": "turn-1",
                "status": "VALID",
                "start_timestamp_us": 50_000,
                "end_timestamp_us": 150_000,
            }
        ],
        "biomechanics_result": {"frame_facts": []},
    }


def test_parser_accepts_optional_overlay_video_and_default_is_unchanged():
    parser = build_parser()
    plain = parser.parse_args(["benchmark-biomechanics", "clip.mp4"])
    requested = parser.parse_args(
        ["benchmark-biomechanics", "clip.mp4", "--overlay-video", "debug.mp4"]
    )
    assert plain.overlay_video is None
    assert requested.overlay_video == "debug.mp4"


@pytest.mark.parametrize(
    ("debug_dir", "overlay_video", "expected"),
    [("debug", None, True), (None, "debug.mp4", True), (None, None, False)],
)
def test_temporal_image_retention_switch(debug_dir, overlay_video, expected):
    args = SimpleNamespace(
        command="benchmark-biomechanics",
        debug_dir=debug_dir,
        overlay_video=overlay_video,
    )
    assert _temporal_collector(args).keep_images is expected


def test_skeleton_provenance_skips_missing_coordinates_safely():
    cv2 = FakeCV2()
    joints = {
        "left_hip": _joint("OBSERVED", raw=(1, 1), stabilized=(2, 2)),
        "left_knee": _joint("INTERPOLATED", raw=(3, 3), stabilized=(4, 4)),
        "left_ankle": _joint("MISSING", raw=(5, 5), stabilized=(6, 6)),
    }
    _draw_temporal_skeleton(cv2, FakeCanvas(), joints)
    assert len(cv2.lines) == 1  # stabilized pose only; no raw-pose double line
    assert len(cv2.circles) == 2
    assert cv2.lines[0][2:] == ((0, 255, 0), 2, cv2.LINE_AA)
    assert cv2.circles == [
        ((2, 2), 5, (0, 255, 255), -1, cv2.LINE_AA),
        ((4, 4), 5, (0, 255, 255), 2, cv2.LINE_AA),
    ]


def test_overlay_hides_eye_and_ear_landmarks_but_keeps_one_head_point():
    cv2 = FakeCV2()
    joints = {
        "nose": _joint("OBSERVED"),
        "left_eye": _joint("OBSERVED"),
        "right_eye": _joint("OBSERVED"),
        "left_ear": _joint("OBSERVED"),
        "right_ear": _joint("OBSERVED"),
    }
    _draw_temporal_skeleton(cv2, FakeCanvas(), joints)
    assert len(cv2.circles) == 1
    assert cv2.circles[0] == ((11, 21), 5, (0, 255, 255), -1, cv2.LINE_AA)


def test_turn_label_only_uses_complete_existing_segment_containment():
    turns = [
        {"turn_id": "turn-1", "start_timestamp_us": 100, "end_timestamp_us": 200},
        {"turn_id": "partial", "start_timestamp_us": None, "end_timestamp_us": 500},
    ]
    assert _containing_turn(turns, 150)["turn_id"] == "turn-1"
    assert _containing_turn(turns, 300) is None


def test_overlay_video_fails_closed_without_frames(monkeypatch, tmp_path):
    _install_video_fakes(monkeypatch)
    with pytest.raises(RuntimeError, match="BIOMECHANICS_DEBUG_VIDEO_NO_FRAMES"):
        write_biomechanics_overlay_video(
            tmp_path / "debug.mp4",
            {"_upstream_debug_report": {"temporal_trace": []}},
            SimpleNamespace(images={}, samples=[]),
            fps=5,
        )


def test_overlay_video_fails_closed_when_writer_cannot_open(monkeypatch, tmp_path):
    cv2 = _install_video_fakes(monkeypatch, writer_opened=False)
    with pytest.raises(
        RuntimeError, match="BIOMECHANICS_DEBUG_VIDEO_WRITER_OPEN_FAILED"
    ):
        write_biomechanics_overlay_video(
            tmp_path / "debug.mp4",
            _report(),
            SimpleNamespace(images={1: b"first"}, samples=[]),
            fps=5,
        )
    assert cv2.writer.released


def test_overlay_video_metadata_and_timestamp_order(monkeypatch, tmp_path):
    cv2 = _install_video_fakes(monkeypatch)
    metadata = write_biomechanics_overlay_video(
        tmp_path / "debug.mp4",
        _report(),
        SimpleNamespace(images={2: b"second", 1: b"first"}, samples=[]),
        fps=5,
    )
    assert cv2.writer.frames == [b"first", b"second"]
    assert cv2.rectangles == []
    assert metadata == {
        "path": str(tmp_path / "debug.mp4"),
        "kind": "SAMPLED_DEBUG_VIDEO",
        "fps": 5.0,
        "frame_count": 2,
        "skipped_frame_count": 0,
        "width_px": 640,
        "height_px": 480,
        "source_model_rerun": False,
    }
