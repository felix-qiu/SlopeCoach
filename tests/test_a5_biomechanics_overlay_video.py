from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from slopecoach_ml.benchmark.biomechanics_debug import (
    _containing_turn,
    _draw_raw_target_pose,
    _draw_temporal_skeleton,
    _overlay_point_radius,
    _raw_pose_points,
    write_biomechanics_overlay_video,
)
from slopecoach_ml.cli.__main__ import _temporal_collector, build_parser
from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import (
    BoundingBox2D,
    FrameGeometry,
    Joint,
    Keypoint2D,
    PersonPose2D,
)
from slopecoach_ml.temporal import TargetPoseSample
from slopecoach_ml.temporal import stabilize_target_pose_stream
from slopecoach_ml.turns import TurnSegmentationConfig, build_turn_signal, segment_turns


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
        self.texts = []
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

    def putText(self, _canvas, text, *_args):
        self.texts.append(text)
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


def _raw_sample(*, state=TargetIdentityState.LOCKED, confidence=0.8, keypoints=None):
    geometry = FrameGeometry(640, 480)
    pose = PersonPose2D(
        BoundingBox2D(90, 80, 100, 260),
        0.9,
        keypoints
        or {
            Joint.LEFT_HIP: Keypoint2D(120, 200, 0.9),
            Joint.LEFT_KNEE: Keypoint2D(125, 270, 0.9),
            Joint.LEFT_ANKLE: Keypoint2D(130, 335, 0.9),
        },
        4,
    )
    return TargetPoseSample(
        100_000,
        1,
        "target-1",
        7,
        state,
        confidence,
        geometry,
        pose,
    )


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
        ((2, 2), 2, (0, 255, 255), -1, cv2.LINE_AA),
        ((4, 4), 2, (0, 255, 255), 2, cv2.LINE_AA),
    ]


def test_trusted_target_can_render_raw_and_stabilized_layers():
    cv2 = FakeCV2()
    raw = _raw_sample()
    assert _draw_raw_target_pose(cv2, FakeCanvas(), raw)
    raw_line_count = len(cv2.lines)
    _draw_temporal_skeleton(
        cv2,
        FakeCanvas(),
        {
            "left_hip": _joint("OBSERVED", stabilized=(120, 200)),
            "left_knee": _joint("OBSERVED", stabilized=(125, 270)),
            "left_ankle": _joint("OBSERVED", stabilized=(130, 335)),
        },
    )
    assert raw_line_count == 2
    assert len(cv2.lines) == 4
    assert cv2.lines[0][2:] == ((255, 160, 0), 1, cv2.LINE_AA)
    assert cv2.lines[-1][2:] == ((0, 255, 0), 2, cv2.LINE_AA)


def test_raw_pose_hides_low_confidence_and_out_of_frame_joints():
    raw = _raw_sample(
        keypoints={
            Joint.LEFT_HIP: Keypoint2D(120, 200, 0.9),
            Joint.LEFT_KNEE: Keypoint2D(125, 270, 0.2),
            Joint.LEFT_ANKLE: Keypoint2D(700, 335, 0.9),
        }
    )
    cv2 = FakeCV2()
    assert _raw_pose_points(raw) == {"left_hip": (120, 200)}
    assert _draw_raw_target_pose(cv2, FakeCanvas(), raw)
    assert cv2.lines == []
    assert len(cv2.circles) == 1


def test_lost_without_current_target_pose_draws_no_raw_skeleton():
    raw = _raw_sample(state=TargetIdentityState.LOST, confidence=0.0)
    raw = TargetPoseSample(
        raw.timestamp_us,
        raw.frame_index,
        raw.target_id,
        None,
        raw.identity_state,
        raw.identity_confidence,
        raw.geometry,
        None,
    )
    cv2 = FakeCV2()
    assert not _draw_raw_target_pose(cv2, FakeCanvas(), raw)
    assert cv2.lines == []
    assert cv2.circles == []


def test_suspect_raw_pose_remains_outside_temporal_turn_and_biomechanics_path():
    raw = _raw_sample(state=TargetIdentityState.SUSPECT, confidence=0.55)
    temporal = stabilize_target_pose_stream([raw])
    stabilized = temporal.samples[0]
    assert raw.raw_target_pose is not None
    assert stabilized.temporal_segment_id is None
    assert stabilized.observed_joint_count == 0
    assert all(point.stabilized_x_px is None for point in stabilized.joints.values())

    signal = build_turn_signal([stabilized])
    assert signal[0].value is None
    assert segment_turns(signal, (), (), TurnSegmentationConfig()) == []
    # The benchmark computes facts only for samples admitted to a temporal segment.
    assert [
        sample for sample in temporal.samples if sample.temporal_segment_id is not None
    ] == []


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
    assert cv2.circles[0] == ((11, 21), 2, (0, 255, 255), -1, cv2.LINE_AA)


def test_overlay_point_radius_scales_with_pose_height_and_is_bounded():
    assert _overlay_point_radius({"nose": _joint("OBSERVED", stabilized=(10, 10))}) == 2
    assert (
        _overlay_point_radius(
            {
                "nose": _joint("OBSERVED", stabilized=(10, 10)),
                "left_ankle": _joint("OBSERVED", stabilized=(10, 310)),
            }
        )
        == 4
    )
    assert (
        _overlay_point_radius(
            {
                "nose": _joint("OBSERVED", stabilized=(10, 0)),
                "left_ankle": _joint("OBSERVED", stabilized=(10, 1000)),
            }
        )
        == 5
    )


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
        "raw_target_pose_debug": {
            "enabled": True,
            "raw_pose_frame_count": 0,
            "analysis_gated_raw_pose_frame_count": 0,
            "trusted_stabilized_pose_frame_count": 0,
        },
    }


def test_manual_suspect_overlay_shows_raw_bbox_and_debug_gate_without_rerun(
    monkeypatch, tmp_path
):
    cv2 = _install_video_fakes(monkeypatch)
    raw = _raw_sample(state=TargetIdentityState.SUSPECT, confidence=0.55)
    report = {
        "manual_target_seed": {"selection_source": "MANUAL_SEED"},
        "_upstream_debug_report": {
            "temporal_trace": [
                {
                    "timestamp_us": raw.timestamp_us,
                    "frame_index": raw.frame_index,
                    "temporal_segment_id": None,
                    "target_id": raw.target_id,
                    "active_track_id": raw.active_track_id,
                    "identity_state": raw.identity_state.value,
                    "joints": {},
                }
            ],
            "turn_signal_samples": [],
        },
        "turn_segments": [],
        "biomechanics_result": {"frame_facts": []},
    }

    class PoisonModel:
        def __getattr__(self, name):
            raise AssertionError(f"overlay attempted model access: {name}")

    collector = SimpleNamespace(
        images={raw.frame_index: b"frame"},
        samples=[raw],
        target_bboxes={
            (raw.timestamp_us, raw.frame_index): raw.raw_target_pose.bbox.to_dict()
        },
        identity_debug={
            (raw.timestamp_us, raw.frame_index): {
                "latest_identity_match_score": 0.47,
                "last_observed_age_us": 0,
            }
        },
        detector=PoisonModel(),
        pose_provider=PoisonModel(),
    )
    metadata = write_biomechanics_overlay_video(
        tmp_path / "manual.mp4", report, collector, fps=5
    )
    assert metadata["source_model_rerun"] is False
    assert metadata["raw_target_pose_debug"] == {
        "enabled": True,
        "raw_pose_frame_count": 1,
        "analysis_gated_raw_pose_frame_count": 1,
        "trusted_stabilized_pose_frame_count": 0,
    }
    assert len(cv2.rectangles) == 1
    assert len(cv2.lines) == 2
    assert any("target_source=MANUAL_SEED" in text for text in cv2.texts)
    assert any("match_score=0.47 observed_age_ms=0" in text for text in cv2.texts)
    assert any("analysis=GATED raw_pose=AVAILABLE" in text for text in cv2.texts)
    assert "RAW POSE DEBUG ONLY" in cv2.texts
