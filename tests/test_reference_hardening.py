from __future__ import annotations

import math
from pathlib import Path

import pytest

from slopecoach_ml.pose import Joint, Keypoint2D, PoseFrame
from slopecoach_ml.reference import (
    ReferenceAnalysisConfig,
    ReferenceAnalysisContext,
    analyze_pose_frame,
    load_golden_fixture,
)


FIXTURE = Path(__file__).parents[1] / "fixtures/golden_pose_001.json"


def analyze(frame: PoseFrame, config: ReferenceAnalysisConfig | None = None):
    return analyze_pose_frame(
        frame,
        context=ReferenceAnalysisContext(
            "reference-run", "provider-a", "model-a", "2.3"
        ),
        config=config or ReferenceAnalysisConfig(),
    )


def test_zero_person_has_no_target_feature() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    empty = PoseFrame(
        frame.contract_version,
        frame.timestamp_us,
        frame.frame_index,
        frame.geometry,
        frame.joint_schema,
        (),
    )
    result = analyze(empty)
    assert result.features["left_knee_angle_2d_degrees"] is None
    assert "no person pose available" in result.warnings


def test_one_person_analysis_and_explicit_provenance() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    result = analyze(frame)
    assert result.features["left_knee_angle_2d_degrees"] == pytest.approx(90)
    assert result.analysis_id == "reference-run"
    assert result.model_versions == {
        "provider": "provider-a",
        "model_id": "model-a",
        "model_version": "2.3",
    }
    assert "golden" not in result.analysis_id


def test_multiple_persons_are_not_selected_by_order() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    multi = PoseFrame(
        frame.contract_version,
        frame.timestamp_us,
        frame.frame_index,
        frame.geometry,
        frame.joint_schema,
        (frame.persons[0], frame.persons[0]),
    )
    result = analyze(multi)
    assert result.features["left_knee_angle_2d_degrees"] is None
    assert "MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED" in result.warnings
    assert "MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED" in result.limitations


def test_non_square_par_limitation_is_machine_readable() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    geometry = frame.geometry.__class__(640, 480, pixel_aspect_ratio=1.2)
    non_square = PoseFrame(
        frame.contract_version,
        frame.timestamp_us,
        frame.frame_index,
        geometry,
        frame.joint_schema,
        frame.persons,
    )
    result = analyze(non_square)
    assert result.features["left_knee_angle_2d_degrees"] is None
    assert "NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED" in result.limitations


def test_confidence_threshold_comes_from_config() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    person = frame.persons[0]
    points = dict(person.keypoints)
    points[Joint.LEFT_KNEE] = Keypoint2D(200, 250, 0.8)
    altered = person.__class__(
        person.bbox, person.person_confidence, points, person.detection_id
    )
    altered_frame = PoseFrame(
        frame.contract_version,
        frame.timestamp_us,
        frame.frame_index,
        frame.geometry,
        frame.joint_schema,
        (altered,),
    )
    assert (
        analyze(
            altered_frame, ReferenceAnalysisConfig(min_joint_confidence=0.9)
        ).features["left_knee_angle_2d_degrees"]
        is None
    )
    assert analyze(
        altered_frame, ReferenceAnalysisConfig(min_joint_confidence=0.7)
    ).features["left_knee_angle_2d_degrees"] == pytest.approx(90)


@pytest.mark.parametrize(
    "config",
    [
        ReferenceAnalysisConfig(min_joint_confidence=math.nan),
        ReferenceAnalysisConfig(square_pixel_tolerance=math.inf),
        ReferenceAnalysisConfig(min_joint_confidence=True),
    ],
)
def test_reference_config_rejects_nonfinite_or_boolean_numbers(config) -> None:
    with pytest.raises((TypeError, ValueError)):
        config.validate()
