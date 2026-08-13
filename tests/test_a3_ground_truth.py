from __future__ import annotations

import json

import pytest

from slopecoach_ml.identity import (
    GT_CONTRACT_VERSION,
    GroundTruthTargetState,
    TargetIdentityGroundTruth,
    evaluate_target_identity_ground_truth,
    load_target_ground_truth,
)


SHA = "a" * 64


def frame(timestamp, state, bbox=None):
    return {
        "timestamp_us": timestamp,
        "frame_index": timestamp // 100_000,
        "target_state": state,
        "bbox": bbox,
        "notes": None,
    }


def bbox(x=10):
    return {
        "x_px": x,
        "y_px": 10,
        "width_px": 100,
        "height_px": 200,
        "coordinate_space": "SourcePixel2D",
    }


def payload(frames):
    return {
        "contract_version": GT_CONTRACT_VERSION,
        "video_sha256": SHA,
        "video_path_hint": "test.mp4",
        "coordinate_space": "SourcePixel2D",
        "annotation_source": "USER_MANUAL",
        "sample_fps": 5.0,
        "width_px": 640,
        "height_px": 480,
        "duration_seconds": 2.0,
        "frames": frames,
    }


def observation(timestamp, state, selected=None, track_id=None):
    return {
        "timestamp_us": timestamp,
        "identity_state": state,
        "selected_bbox": selected,
        "active_track_id": track_id,
    }


def test_valid_present_absent_uncertain_unlabeled_contract() -> None:
    gt = TargetIdentityGroundTruth.from_dict(
        payload(
            [
                frame(0, "PRESENT", bbox()),
                frame(100_000, "ABSENT"),
                frame(200_000, "UNCERTAIN"),
                frame(300_000, "UNLABELED"),
            ]
        )
    )
    assert [item.target_state for item in gt.frames] == list(GroundTruthTargetState)
    json.dumps(gt.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "bad_frames,match",
    [
        ([frame(0, "PRESENT")], "PRESENT"),
        ([frame(0, "ABSENT", bbox())], "ABSENT"),
        ([frame(0, "BAD")], "target_state"),
        ([frame(0, "UNLABELED"), frame(0, "UNLABELED")], "duplicate"),
        ([frame(-1, "UNLABELED")], "non-negative"),
    ],
)
def test_gt_rejects_invalid_frame_semantics(bad_frames, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        TargetIdentityGroundTruth.from_dict(payload(bad_frames))


def test_gt_rejects_bool_nonfinite_and_coordinate_space() -> None:
    for bad in (
        frame(True, "UNLABELED"),
        frame(0, "PRESENT", {**bbox(), "x_px": float("nan")}),
        frame(0, "PRESENT", {**bbox(), "coordinate_space": "ModelCoordinate"}),
    ):
        with pytest.raises((TypeError, ValueError)):
            TargetIdentityGroundTruth.from_dict(payload([bad]))


def test_gt_hash_mismatch_rejected(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"different")
    annotation = tmp_path / "gt.json"
    annotation.write_text(json.dumps(payload([])), encoding="utf-8")
    with pytest.raises(ValueError, match="TARGET_IDENTITY_GT_VIDEO_MISMATCH"):
        load_target_ground_truth(annotation, video)


def test_gt_json_nan_and_inf_rejected(tmp_path) -> None:
    for constant in ("NaN", "Infinity"):
        path = tmp_path / f"{constant}.json"
        path.write_text(
            json.dumps(payload([])).replace(
                '"sample_fps": 5.0', f'"sample_fps": {constant}'
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_target_ground_truth(path)


def test_gt_accuracy_formulas_and_state_conditioning() -> None:
    gt = TargetIdentityGroundTruth.from_dict(
        payload(
            [
                frame(0, "PRESENT", bbox()),
                frame(100_000, "PRESENT", bbox()),
                frame(200_000, "PRESENT", bbox()),
                frame(300_000, "ABSENT"),
                frame(400_000, "ABSENT"),
                frame(500_000, "UNCERTAIN"),
                frame(600_000, "UNLABELED"),
            ]
        )
    )
    observations = [
        observation(0, "LOCKED", bbox(), 1),
        observation(100_000, "LOCKED", bbox(300), 2),
        observation(200_000, "LOST"),
        observation(300_000, "LOST"),
        observation(400_000, "LOCKED", bbox(300), 9),
        observation(500_000, "LOCKED", bbox(), 1),
        observation(600_000, "LOCKED", bbox(), 1),
    ]
    result = evaluate_target_identity_ground_truth(observations, gt)
    metrics = result["identity_accuracy"]
    assert metrics["correct_lock_count"] == 1
    assert metrics["wrong_target_lock_count"] == 1
    assert metrics["target_not_locked_count"] == 1
    assert metrics["false_lock_when_absent_count"] == 1
    assert metrics["target_lock_coverage_when_present"] == pytest.approx(1 / 3)
    assert metrics["wrong_target_rate"] == pytest.approx(2 / 3)
    assert metrics["false_lock_when_absent_rate"] == pytest.approx(1 / 2)
    assert metrics["target_frame_accuracy"] == pytest.approx(2 / 5)
    assert result["target_present_state_metrics"][
        "lost_when_present_ratio"
    ] == pytest.approx(1 / 3)
    assert result["target_absent_state_metrics"][
        "locked_when_absent_ratio"
    ] == pytest.approx(1 / 2)


def test_recovery_opportunity_success_and_reacquisition_time() -> None:
    gt = TargetIdentityGroundTruth.from_dict(
        payload(
            [
                frame(0, "PRESENT", bbox()),
                frame(100_000, "ABSENT"),
                frame(200_000, "ABSENT"),
                frame(300_000, "PRESENT", bbox()),
                frame(400_000, "PRESENT", bbox()),
            ]
        )
    )
    observations = [
        observation(0, "LOCKED", bbox(), 1),
        observation(100_000, "LOST"),
        observation(200_000, "RECOVERING"),
        observation(300_000, "RECOVERING"),
        observation(400_000, "LOCKED", bbox(), 2),
    ]
    recovery = evaluate_target_identity_ground_truth(observations, gt)["recovery"]
    assert recovery["recovery_opportunity_count"] == 1
    assert recovery["successful_recovery_count"] == 1
    assert recovery["recovery_success_rate"] == 1
    assert recovery["median_reacquisition_time_us"] == 100_000


def test_recovery_after_system_loss_while_target_remains_present() -> None:
    gt = TargetIdentityGroundTruth.from_dict(
        payload(
            [
                frame(0, "PRESENT", bbox()),
                frame(100_000, "PRESENT", bbox()),
                frame(200_000, "PRESENT", bbox()),
            ]
        )
    )
    observations = [
        observation(0, "LOCKED", bbox(), 1),
        observation(100_000, "LOST"),
        observation(200_000, "LOCKED", bbox(), 2),
    ]
    recovery = evaluate_target_identity_ground_truth(observations, gt)["recovery"]
    assert recovery["recovery_opportunity_count"] == 1
    assert recovery["successful_recovery_count"] == 1
    assert recovery["median_reacquisition_time_us"] == 100_000


def test_recovery_wrong_target_count_is_event_based() -> None:
    gt = TargetIdentityGroundTruth.from_dict(
        payload(
            [
                frame(0, "PRESENT", bbox()),
                frame(100_000, "ABSENT"),
                frame(200_000, "PRESENT", bbox()),
                frame(300_000, "PRESENT", bbox()),
            ]
        )
    )
    observations = [
        observation(0, "LOCKED", bbox(), 1),
        observation(100_000, "LOST"),
        observation(200_000, "LOCKED", bbox(300), 9),
        observation(300_000, "LOCKED", bbox(300), 9),
    ]
    recovery = evaluate_target_identity_ground_truth(observations, gt)["recovery"]
    assert recovery["recovery_opportunity_count"] == 1
    assert recovery["successful_recovery_count"] == 0
    assert recovery["recovery_wrong_target_count"] == 1
