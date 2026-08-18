from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from slopecoach_ml.benchmark import (
    TemporalTurnCollector,
    benchmark_biomechanics_frames,
    benchmark_target_identity_frames,
    benchmark_temporal_turns_frames,
)
from slopecoach_ml.cli.__main__ import build_parser
from slopecoach_ml.detection import Detection
from slopecoach_ml.identity import (
    InitialTargetSelectorConfig,
    ManualTargetSeed,
    PersonCandidate,
    TargetIdentityConfig,
    select_manual_target_seed_match,
)
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry, MMPoseRTMWPoseProvider
from slopecoach_ml.tracking import TrackObservation, TrackState
from slopecoach_ml.video import SampledFrame


class TwoPersonDetector:
    name = "fake-two-person-detector"

    def detect(self, image, geometry):
        return (
            Detection(0, BoundingBox2D(20, 40, 100, 240), 0.91),
            Detection(1, BoundingBox2D(300, 40, 100, 240), 0.96),
        )


class SinglePersonDetector:
    name = "fake-single-person-detector"

    def detect(self, image, geometry):
        return (Detection(0, BoundingBox2D(100, 40, 100, 240), 0.96),)


class DistantPersonDetector:
    name = "fake-distant-person-detector"

    def detect(self, image, geometry):
        return (Detection(0, BoundingBox2D(100, 100, 29, 58), 0.20),)


class FragmentingDetector:
    name = "fake-fragmenting-detector"

    def detect(self, image, geometry):
        x_px = 20 if image < 2 else 450
        return (Detection(0, BoundingBox2D(x_px, 40, 100, 240), 0.96),)


class PoseBackend:
    def infer(self, image, boxes):
        results = []
        for left, top, right, bottom in boxes:
            points = [[left + 50, top + 20] for _ in range(133)]
            for index, point in {
                5: (left + 30, top + 30),
                6: (left + 70, top + 30),
                11: (left + 35, top + 100),
                12: (left + 65, top + 100),
                13: (left + 40, top + 170),
                14: (left + 60, top + 170),
                15: (left + 42, top + 230),
                16: (left + 58, top + 230),
            }.items():
                points[index] = point
            results.append((points, [0.9] * 133))
        return results


class CountingPoseBackend(PoseBackend):
    def __init__(self):
        self.call_count = 0

    def infer(self, image, boxes):
        self.call_count += 1
        return super().infer(image, boxes)


class Appearance:
    def encode(self, image, bbox):
        return (1.0, 0.0)


@dataclass
class Clock:
    value: float = 0.0

    def __call__(self):
        self.value += 0.001
        return self.value


def _frames(count=8):
    geometry = FrameGeometry(640, 480)
    return [
        SampledFrame(index, index * 200_000, geometry, index) for index in range(count)
    ]


@pytest.fixture(autouse=True)
def _metadata(monkeypatch):
    monkeypatch.setattr(
        "slopecoach_ml.benchmark.target_identity.inspect_video",
        lambda path: type(
            "Metadata", (), {"to_dict": lambda self: {"path": str(path)}}
        )(),
    )


def _target_report(*, detector=None, seed=None, frames=None, sample_fps=5, **kwargs):
    return benchmark_target_identity_frames(
        input_path="missing.mp4",
        frames=frames or _frames(),
        detector=detector or TwoPersonDetector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        sample_fps=sample_fps,
        appearance_encoder=Appearance(),
        manual_target_seed=seed,
        clock=Clock(),
        **kwargs,
    )


def test_parser_accepts_manual_seed_pair_and_preserves_auto_default():
    parser = build_parser()
    auto = parser.parse_args(["benchmark-biomechanics", "clip.mp4"])
    manual = parser.parse_args(
        [
            "benchmark-biomechanics",
            "clip.mp4",
            "--target-seed-time",
            "1.6",
            "--target-seed-point",
            "820,460",
        ]
    )
    assert auto.target_seed_time is None
    assert auto.target_seed_point is None
    assert manual.target_seed_time == 1.6
    assert manual.target_seed_point == (820.0, 460.0)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--target-seed-time", "1.6"],
        ["--target-seed-point", "820,460"],
    ],
)
def test_parser_rejects_unpaired_manual_seed(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["benchmark-biomechanics", "clip.mp4", *arguments])


@pytest.mark.parametrize("value", ["820", "820,", ",460", "820,460,1", "hello,460"])
def test_parser_rejects_malformed_manual_seed_point(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "benchmark-biomechanics",
                "clip.mp4",
                "--target-seed-time",
                "1.6",
                "--target-seed-point",
                value,
            ]
        )


def test_parser_rejects_negative_manual_seed_time():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "benchmark-biomechanics",
                "clip.mp4",
                "--target-seed-time",
                "-0.1",
                "--target-seed-point",
                "20,20",
            ]
        )


def test_manual_seed_prevents_auto_lock_and_initializes_containing_person():
    report = _target_report(seed=ManualTargetSeed(0.4, 350, 100))
    assert [item["identity_state"] for item in report["frame_observations"][:2]] == [
        "UNINITIALIZED",
        "UNINITIALIZED",
    ]
    applied = report["manual_target_seed"]
    assert applied["selection_source"] == "MANUAL_SEED"
    assert applied["ground_truth_status"] == "NOT_GROUND_TRUTH"
    assert applied["applied_timestamp_us"] == 400_000
    assert applied["applied_frame_index"] == 2
    assert applied["selected_track_id"] == 2
    assert applied["selected_detection_id"] == 1
    assert applied["identity_evidence_confidence"] != 1.0
    assert report["TARGET_IDENTITY_GT_STATUS"] == "NOT_AVAILABLE"
    assert report["identity_accuracy"]["target_frame_accuracy"] is None
    json.dumps(report, allow_nan=False)


def test_auto_path_remains_auto_and_does_not_add_manual_metadata():
    report = _target_report(
        detector=SinglePersonDetector(),
        selector_config=InitialTargetSelectorConfig(
            initialization_window_us=200_000,
            minimum_track_observations=2,
            minimum_lock_score=0.1,
            minimum_winner_margin=0.0,
        ),
    )
    assert "manual_target_seed" not in report
    assert report["target_identity"]["first_lock_timestamp_us"] == 200_000


def test_manual_seed_point_out_of_bounds_fails_closed():
    with pytest.raises(ValueError, match="MANUAL_TARGET_SEED_POINT_OUT_OF_BOUNDS"):
        _target_report(seed=ManualTargetSeed(0.0, 640, 100))


def test_manual_seed_without_containing_candidate_fails_closed():
    with pytest.raises(ValueError, match="MANUAL_TARGET_SEED_NO_MATCH"):
        _target_report(seed=ManualTargetSeed(0.0, 200, 100))


def test_manual_seed_without_eligible_sampled_frame_fails_closed():
    irregular = [
        SampledFrame(0, 0, FrameGeometry(640, 480), 0),
        SampledFrame(1, 500_000, FrameGeometry(640, 480), 1),
    ]
    with pytest.raises(ValueError, match="MANUAL_TARGET_SEED_FRAME_NOT_FOUND"):
        _target_report(seed=ManualTargetSeed(0.25, 50, 100), frames=irregular)


def test_manual_seed_uses_closest_sample_and_prefers_earlier_ties():
    geometry = FrameGeometry(640, 480)
    closer_later = [
        SampledFrame(0, 0, geometry, 0),
        SampledFrame(1, 400_000, geometry, 1),
        SampledFrame(2, 700_000, geometry, 2),
    ]
    later_report = _target_report(
        detector=SinglePersonDetector(),
        seed=ManualTargetSeed(0.25, 120, 100),
        frames=closer_later,
        sample_fps=2,
    )
    assert later_report["manual_target_seed"]["applied_timestamp_us"] == 400_000

    tie = [
        SampledFrame(0, 0, geometry, 0),
        SampledFrame(1, 500_000, geometry, 1),
    ]
    tie_report = _target_report(
        detector=SinglePersonDetector(),
        seed=ManualTargetSeed(0.25, 120, 100),
        frames=tie,
        sample_fps=2,
    )
    assert tie_report["manual_target_seed"]["applied_timestamp_us"] == 0


def _track(track_id, detection_id, bbox, confidence=0.9):
    return TrackObservation(
        track_id,
        detection_id,
        bbox,
        confidence,
        TrackState.CONFIRMED,
        3,
        0,
        0,
        0,
        0.0,
        0.0,
    )


def _candidate(detection_id, bbox, quality):
    return PersonCandidate(detection_id, bbox, 0.9, quality, {})


def test_overlapping_manual_seed_uses_deterministic_quality_then_track_id_ties():
    bbox = BoundingBox2D(100, 100, 100, 200)
    tracks = (_track(9, 9, bbox), _track(4, 4, bbox), _track(7, 7, bbox))
    candidates = {
        9: _candidate(9, bbox, 0.7),
        4: _candidate(4, bbox, 0.9),
        7: _candidate(7, bbox, 0.9),
    }
    match = select_manual_target_seed_match(
        ManualTargetSeed(0, 150, 200), FrameGeometry(640, 480), tracks, candidates
    )
    assert match.track.track_id == 4


def test_manual_seed_uses_existing_manager_recovery_and_does_not_pin_track_id():
    report = _target_report(
        detector=FragmentingDetector(),
        seed=ManualTargetSeed(0.2, 50, 100),
        frames=_frames(5),
        identity_config=TargetIdentityConfig(
            minimum_lock_score=0.1,
            minimum_winner_margin=0.0,
            suspect_timeout_us=0,
            lost_timeout_us=1_500_000,
            recovery_confirmation_observations=1,
        ),
    )
    assert report["manual_target_seed"]["selected_track_id"] == 1
    assert report["target_identity"]["active_track_id_change_count"] == 1
    assert report["target_identity"]["relock_count"] == 1
    assert report["target_identity"]["recovery_events"][0]["new_track_id"] == 2
    assert report["frame_observations"][-1]["active_track_id"] == 2


def test_manual_identity_flows_to_temporal_and_biomechanics_without_becoming_gt():
    collector = TemporalTurnCollector()
    common = dict(
        input_path="missing.mp4",
        frames=_frames(),
        detector=TwoPersonDetector(),
        pose_provider=MMPoseRTMWPoseProvider(PoseBackend()),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        sample_fps=5,
        appearance_encoder=Appearance(),
        collector=collector,
        manual_target_seed=ManualTargetSeed(0.4, 350, 100),
    )
    temporal = benchmark_temporal_turns_frames(**common)
    selected_sample = collector.samples[2]
    assert selected_sample.active_track_id == 2
    assert selected_sample.raw_target_pose is not None
    assert selected_sample.raw_target_pose.bbox.x_px == 300
    assert temporal["manual_target_seed"]["selected_track_id"] == 2
    assert temporal["identity_input"]["target_identity_accuracy_status"] == "UNKNOWN"

    biomechanics_collector = TemporalTurnCollector()
    common["frames"] = _frames()
    common["collector"] = biomechanics_collector
    biomechanics = benchmark_biomechanics_frames(**common)
    assert biomechanics["manual_target_seed"]["selected_track_id"] == 2
    assert biomechanics_collector.samples[2].raw_target_pose.bbox.x_px == 300
    assert biomechanics["ground_truth"]["TARGET_IDENTITY_ACCURACY_STATUS"] == "UNKNOWN"


def test_distant_manual_suspect_keeps_raw_pose_without_trusted_analysis():
    backend = CountingPoseBackend()
    collector = TemporalTurnCollector()
    frames = _frames(4)
    report = benchmark_biomechanics_frames(
        input_path="missing.mp4",
        frames=frames,
        detector=DistantPersonDetector(),
        pose_provider=MMPoseRTMWPoseProvider(backend),
        detector_model={"model_id": "det"},
        pose_model={"model_id": "pose"},
        sample_fps=5,
        appearance_encoder=Appearance(),
        collector=collector,
        manual_target_seed=ManualTargetSeed(0.0, 110, 110),
    )

    assert backend.call_count == len(frames)
    assert collector.samples[0].identity_confidence < 0.62
    assert collector.samples[0].identity_state.value == "LOCKED"
    assert all(sample.raw_target_pose is not None for sample in collector.samples)
    assert any(
        sample.identity_state.value == "SUSPECT" for sample in collector.samples[1:]
    )
    suspect_debug = [
        collector.identity_debug[(sample.timestamp_us, sample.frame_index)]
        for sample in collector.samples
        if sample.identity_state.value == "SUSPECT"
    ]
    assert suspect_debug
    assert all(
        item["latest_identity_match_score"] is not None for item in suspect_debug
    )
    assert all(item["last_observed_age_us"] == 0 for item in suspect_debug)
    assert report["frame_biomechanics"]["trusted_frame_count"] == 0
    assert report["biomechanics_result"]["frame_facts"] == []
    assert report["turn_segments"] == []
    assert report["feature_registry_sha256"] == (
        "2777c3fbf7513e7537122f897f1901e61baf7eeddcee927937decb7476953048"
    )
