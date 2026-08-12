from __future__ import annotations

import json
from pathlib import Path

import pytest

from slopecoach_ml.reference import (
    ReferenceAnalysisConfig,
    ReferenceAnalysisContext,
    ReferenceAnalysisResult,
    analyze_pose_frame,
    load_golden_fixture,
)


FIXTURE = Path(__file__).parents[1] / "fixtures/golden_pose_001.json"


def analyze(frame):
    return analyze_pose_frame(
        frame,
        context=ReferenceAnalysisContext(
            "golden-pose-001", "golden-fixture", "golden-pose-001", "golden-pose-v1"
        ),
        config=ReferenceAnalysisConfig(),
    )


def test_golden_fixture_is_deterministic() -> None:
    frame, expected = load_golden_fixture(FIXTURE)
    first = analyze(frame)
    second = analyze(frame)
    assert first.to_json() == second.to_json()
    assert first.features["left_knee_angle_2d_degrees"] == pytest.approx(
        expected["left_knee_angle_2d_degrees"], abs=expected["absolute_tolerance"]
    )


def test_reference_result_serialization_preserves_null() -> None:
    frame, _ = load_golden_fixture(FIXTURE)
    result: ReferenceAnalysisResult = analyze(frame)
    data = json.loads(result.to_json())
    assert data["video_metadata"] is None
    assert data["model_versions"]["provider"] == "golden-fixture"
    assert data["limitations"]
