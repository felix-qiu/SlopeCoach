from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from slopecoach_ml.biomechanics import (
    BIOMECHANICS_FEATURE_SCHEMA_VERSION,
    FEATURE_REGISTRY_SHA256,
    FEATURE_REGISTRY_V1,
    FIXED_ML_FEATURE_VECTOR_STATUS,
    FRAME_FEATURE_REGISTRY_V1,
    TEMPORAL_FEATURE_REGISTRY_V1,
    TURN_FEATURE_REGISTRY_V1,
    BiomechanicsFact,
    BiomechanicsFactScope,
    BiomechanicsFactStatus,
    BiomechanicsFeatureConfig,
    BiomechanicsFeatureFamily,
    FeatureAggregate,
    TemporalBiomechanicsResult,
    TurnBiomechanicsResult,
    canonical_feature_registry_json,
    compute_frame_biomechanics,
    compute_turn_biomechanics,
    derivative_aggregates,
)
from slopecoach_ml.pose import Joint
from slopecoach_ml.turns import TurnPhaseSign, TurnSegment, TurnSegmentStatus

from test_a5_biomechanics import sample


EXPECTED_FEATURE_IDS = (
    "left_knee_angle_2d_deg",
    "right_knee_angle_2d_deg",
    "bilateral_knee_mean_angle_2d_deg",
    "bilateral_knee_abs_difference_2d_deg",
    "ankle_separation_body_scale",
    "knee_separation_body_scale",
    "hip_separation_body_scale",
    "shoulder_separation_body_scale",
    "ankle_to_shoulder_separation_ratio_2d",
    "shoulder_to_ankle_screen_lateral_offset_body_scale",
    "torso_screen_inclination_deg",
    "hip_to_ankle_screen_lateral_offset_body_scale",
    "signed_lateral_body_proxy",
    "shoulder_hip_axis_difference_2d_deg",
    "left_knee_angle_abs_velocity_median_deg_per_s",
    "right_knee_angle_abs_velocity_median_deg_per_s",
    "bilateral_knee_mean_angle_abs_velocity_median_deg_per_s",
    "signed_lateral_body_proxy_abs_velocity_median_per_s",
    "turn_duration_us",
    "turn_peak_lateral_proxy",
    "bilateral_knee_mean_angle_at_apex_deg",
    "bilateral_knee_abs_difference_at_apex_deg",
    "ankle_separation_at_apex_body_scale",
    "bilateral_knee_mean_angle_at_start_deg",
    "bilateral_knee_mean_angle_at_end_deg",
    "knee_angle_change_start_to_apex_deg",
    "knee_angle_change_apex_to_end_deg",
    "minimum_mean_knee_angle_timestamp_us",
    "minimum_mean_knee_angle_offset_from_apex_us",
    "minimum_mean_knee_angle_phase_offset",
)


def fact(**overrides):
    values = {
        "feature_id": "feature",
        "family": BiomechanicsFeatureFamily.STANCE_PROXY,
        "scope": BiomechanicsFactScope.FRAME,
        "unit": "deg",
        "value": 1.0,
        "status": BiomechanicsFactStatus.AVAILABLE,
    }
    values.update(overrides)
    return BiomechanicsFact(**values)


def aggregate(**overrides):
    values = {
        "feature_id": "feature",
        "temporal_segment_id": 1,
        "unit": "deg",
        "sample_count": 3,
        "available_count": 2,
        "support_ratio": 2 / 3,
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "range": 2.0,
        "median_support_confidence": 0.8,
        "observed_only_sample_count": 1,
        "interpolated_support_sample_count": 1,
        "observed_only_ratio": 1 / 3,
        "interpolated_support_ratio": 1 / 3,
        "status": BiomechanicsFactStatus.AVAILABLE,
    }
    values.update(overrides)
    return FeatureAggregate(**values)


def turn(*, start=0, apex=500_000, end=1_500_000, run=1, status=None):
    complete = start is not None and end is not None
    return TurnSegment(
        "turn-1",
        1,
        run,
        start,
        apex,
        end,
        TurnPhaseSign.POSITIVE_PHASE,
        0.8,
        0.8,
        end - start if complete else None,
        3,
        0,
        0.0,
        1.0,
        status or (TurnSegmentStatus.VALID if complete else TurnSegmentStatus.PARTIAL),
    )


def frame_facts_at(timestamp, confidence=0.9, interpolated=None):
    return compute_frame_biomechanics(
        sample(timestamp, confidence=confidence, interpolated=interpolated), 100
    )


def by_id(result):
    return {item.feature_id: item for item in result.facts}


def test_registry_snapshot_and_process_independent_fingerprint():
    assert BIOMECHANICS_FEATURE_SCHEMA_VERSION == "biomechanics-feature-schema-v1"
    assert FIXED_ML_FEATURE_VECTOR_STATUS == "NOT_FROZEN"
    assert (
        len(FRAME_FEATURE_REGISTRY_V1),
        len(TEMPORAL_FEATURE_REGISTRY_V1),
        len(TURN_FEATURE_REGISTRY_V1),
    ) == (14, 4, 12)
    assert (
        tuple(item.feature_id for item in FEATURE_REGISTRY_V1) == EXPECTED_FEATURE_IDS
    )
    assert len(set(EXPECTED_FEATURE_IDS)) == len(FEATURE_REGISTRY_V1) == 30
    assert (
        FEATURE_REGISTRY_SHA256
        == "2777c3fbf7513e7537122f897f1901e61baf7eeddcee927937decb7476953048"
    )
    assert (
        hashlib.sha256(canonical_feature_registry_json().encode()).hexdigest()
        == FEATURE_REGISTRY_SHA256
    )
    command = "from slopecoach_ml.biomechanics import FEATURE_REGISTRY_SHA256;print(FEATURE_REGISTRY_SHA256)"
    assert (
        subprocess.check_output([sys.executable, "-c", command], text=True).strip()
        == FEATURE_REGISTRY_SHA256
    )
    payload = json.loads(canonical_feature_registry_json())
    reordered = [{key: item[key] for key in reversed(tuple(item))} for item in payload]
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == json.dumps(
        reordered, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize(
    "config",
    [
        BiomechanicsFeatureConfig(minimum_joint_support_confidence=True),
        BiomechanicsFeatureConfig(square_pixel_tolerance=False),
        BiomechanicsFeatureConfig(minimum_joint_support_confidence="0.3"),
        BiomechanicsFeatureConfig(square_pixel_tolerance=None),
        BiomechanicsFeatureConfig(minimum_joint_support_confidence=math.nan),
        BiomechanicsFeatureConfig(square_pixel_tolerance=math.inf),
        BiomechanicsFeatureConfig(minimum_aggregate_samples=True),
        BiomechanicsFeatureConfig(apex_match_tolerance_us=True),
        BiomechanicsFeatureConfig(boundary_match_tolerance_us=True),
    ],
)
def test_config_rejects_bool_and_nonfinite_numeric_values(config):
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"value": None},
        {"value": math.nan},
        {"value": math.inf},
        {"value": True},
        {"status": BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE},
        {"timestamp_us": -1},
        {"timestamp_us": True},
        {"temporal_segment_id": 0},
        {"signal_run_id": 0},
        {"support_confidence": 1.1},
        {"support_confidence": math.nan},
        {"observed_joint_count": -1},
        {"interpolated_joint_count": -1},
        {"feature_id": ""},
        {"unit": ""},
    ],
)
def test_fact_invariants_reject_invalid_data(overrides):
    with pytest.raises(ValueError):
        fact(**overrides)


def test_fact_available_and_unavailable_invariants_accept_valid_data():
    assert fact(value=0).value == 0
    assert fact(value=1.5).value == 1.5
    unavailable = fact(value=None, status=BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE)
    assert unavailable.value is None
    with pytest.raises(ValueError, match="exceed"):
        fact(required_joints=(Joint.LEFT_KNEE,), observed_joint_count=2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"available_count": 4},
        {"support_ratio": 1.1},
        {"median": None},
        {"status": BiomechanicsFactStatus.INSUFFICIENT_SAMPLES},
        {"minimum": 4.0},
        {"range": -1.0},
        {"median": math.nan},
    ],
)
def test_aggregate_strictness(overrides):
    with pytest.raises(ValueError):
        aggregate(**overrides)
    valid_missing = aggregate(
        status=BiomechanicsFactStatus.INSUFFICIENT_SAMPLES,
        median=None,
        minimum=None,
        maximum=None,
        range=None,
    )
    assert valid_missing.median is None


def test_public_derivative_entrypoint_validates_config():
    with pytest.raises(ValueError):
        derivative_aggregates(
            (), BiomechanicsFeatureConfig(minimum_derivative_dt_us=True)
        )


def test_complete_turn_timing_is_window_and_signal_run_local():
    frames = (
        *frame_facts_at(0),
        *frame_facts_at(500_000),
        *frame_facts_at(1_500_000),
        *frame_facts_at(1_600_000),
    )
    knee_ids = "bilateral_knee_mean_angle_2d_deg"
    modified = tuple(
        replace(
            item,
            value={0: 90, 500_000: 60, 1_500_000: 90, 1_600_000: 1}[item.timestamp_us],
        )
        if item.feature_id == knee_ids
        else item
        for item in frames
    )
    result = by_id(
        compute_turn_biomechanics(
            [turn()],
            modified,
            {1: {0, 500_000, 1_500_000, 1_600_000}},
            BiomechanicsFeatureConfig(),
        )[0]
    )
    assert result["minimum_mean_knee_angle_timestamp_us"].value == 500_000
    assert result["minimum_mean_knee_angle_offset_from_apex_us"].value == 0
    assert result["minimum_mean_knee_angle_phase_offset"].value == 0
    other_run = by_id(
        compute_turn_biomechanics(
            [turn(run=2)],
            modified,
            {1: {0}, 2: {500_000, 1_500_000}},
            BiomechanicsFeatureConfig(),
        )[0]
    )
    assert other_run["bilateral_knee_mean_angle_at_start_deg"].value is None
    assert other_run["minimum_mean_knee_angle_timestamp_us"].value == 500_000


@pytest.mark.parametrize("missing", ["start", "end"])
def test_partial_turn_keeps_local_evidence_but_disables_complete_timing(missing):
    frames = tuple(
        fact
        for timestamp in (0, 500_000, 1_500_000)
        for fact in frame_facts_at(timestamp)
    )
    current = turn(start=None) if missing == "start" else turn(end=None)
    result = by_id(
        compute_turn_biomechanics(
            [current], frames, {1: {0, 500_000, 1_500_000}}, BiomechanicsFeatureConfig()
        )[0]
    )
    assert (
        result["bilateral_knee_mean_angle_at_apex_deg"].status
        is BiomechanicsFactStatus.AVAILABLE
    )
    missing_fact = result[f"bilateral_knee_mean_angle_at_{missing}_deg"]
    assert missing_fact.status is BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE
    missing_delta = (
        "knee_angle_change_start_to_apex_deg"
        if missing == "start"
        else "knee_angle_change_apex_to_end_deg"
    )
    assert (
        result[missing_delta].status is BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE
    )
    present = "end" if missing == "start" else "start"
    assert (
        result[f"bilateral_knee_mean_angle_at_{present}_deg"].status
        is BiomechanicsFactStatus.AVAILABLE
    )
    present_delta = (
        "knee_angle_change_apex_to_end_deg"
        if missing == "start"
        else "knee_angle_change_start_to_apex_deg"
    )
    assert result[present_delta].status is BiomechanicsFactStatus.AVAILABLE
    for feature_id in (
        "minimum_mean_knee_angle_timestamp_us",
        "minimum_mean_knee_angle_offset_from_apex_us",
        "minimum_mean_knee_angle_phase_offset",
    ):
        assert result[feature_id].value is None
        assert (
            result[feature_id].status
            is BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE
        )


def test_matching_tolerance_tie_break_and_known_window():
    frames = tuple(
        fact
        for timestamp in (490_000, 510_000, 1_510_000)
        for fact in frame_facts_at(timestamp)
    )
    settings = BiomechanicsFeatureConfig(
        apex_match_tolerance_us=20_000, boundary_match_tolerance_us=20_000
    )
    result = by_id(
        compute_turn_biomechanics(
            [turn(start=490_000, end=1_500_000)],
            frames,
            {1: {490_000, 510_000, 1_510_000}},
            settings,
        )[0]
    )
    apex = result["bilateral_knee_mean_angle_at_apex_deg"]
    assert apex.status is BiomechanicsFactStatus.AVAILABLE
    # 490k and 510k are equally close; deterministic matching chooses the earlier sample.
    source = next(
        item
        for item in frames
        if item.timestamp_us == 490_000
        and item.feature_id == "bilateral_knee_mean_angle_2d_deg"
    )
    assert apex.value == source.value
    assert result["bilateral_knee_mean_angle_at_end_deg"].value is None
    outside = by_id(
        compute_turn_biomechanics(
            [turn(apex=700_000)], frames, {1: {490_000, 510_000}}, settings
        )[0]
    )
    assert (
        outside["bilateral_knee_mean_angle_at_apex_deg"].status
        is BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE
    )


def test_turn_source_and_delta_evidence_propagates_conservatively():
    frames = tuple(
        fact
        for timestamp in (0, 500_000, 1_500_000)
        for fact in frame_facts_at(
            timestamp, interpolated=Joint.LEFT_KNEE if timestamp == 500_000 else None
        )
    )
    adjusted = tuple(
        replace(item, support_confidence=0.65 if item.timestamp_us == 500_000 else 0.8)
        for item in frames
    )
    result = by_id(
        compute_turn_biomechanics(
            [turn()],
            adjusted,
            {1: {0, 500_000, 1_500_000}},
            BiomechanicsFeatureConfig(),
        )[0]
    )
    apex = result["bilateral_knee_mean_angle_at_apex_deg"]
    assert apex.support_confidence == 0.65
    assert apex.required_joints
    assert apex.interpolated_joint_count == 1
    delta = result["knee_angle_change_start_to_apex_deg"]
    assert delta.support_confidence == 0.65
    assert len(delta.required_joints) == 6
    assert delta.interpolated_joint_count == 1
    assert result["turn_peak_lateral_proxy"].support_confidence is None
    assert result["minimum_mean_knee_angle_timestamp_us"].required_joints


def turn_fact(feature_id="turn_duration_us", **overrides):
    definition = next(
        item for item in TURN_FEATURE_REGISTRY_V1 if item.feature_id == feature_id
    )
    values = {
        "feature_id": feature_id,
        "family": definition.family,
        "scope": definition.scope,
        "unit": definition.unit,
        "value": 1,
        "status": BiomechanicsFactStatus.AVAILABLE,
        "temporal_segment_id": 1,
        "signal_run_id": 1,
        "turn_id": "turn-1",
    }
    values.update(overrides)
    return BiomechanicsFact(**values)


@pytest.mark.parametrize(
    "bad",
    [
        turn_fact(turn_id="wrong"),
        turn_fact(temporal_segment_id=2),
        turn_fact(signal_run_id=2),
        turn_fact(scope=BiomechanicsFactScope.FRAME),
    ],
)
def test_turn_result_rejects_mismatched_fact_metadata(bad):
    with pytest.raises(ValueError):
        TurnBiomechanicsResult("turn-1", 1, 1, "POSITIVE_PHASE", (bad,))
    duplicate = turn_fact()
    with pytest.raises(ValueError, match="unique"):
        TurnBiomechanicsResult("turn-1", 1, 1, "POSITIVE_PHASE", (duplicate, duplicate))


def test_temporal_result_rejects_duplicate_frame_facts_and_serializes_strictly():
    current = fact(timestamp_us=1, temporal_segment_id=1)
    kwargs = {
        "contract_version": "temporal-biomechanics-v2",
        "feature_schema_version": BIOMECHANICS_FEATURE_SCHEMA_VERSION,
        "feature_registry_sha256": FEATURE_REGISTRY_SHA256,
        "config": BiomechanicsFeatureConfig(),
        "frame_facts": (current,),
        "temporal_segment_features": (),
        "turn_features": (),
        "feature_coverage": {},
    }
    result = TemporalBiomechanicsResult(**kwargs)
    assert (
        json.loads(result.to_json())["feature_registry_sha256"]
        == FEATURE_REGISTRY_SHA256
    )
    with pytest.raises(ValueError, match="duplicate"):
        TemporalBiomechanicsResult(**{**kwargs, "frame_facts": (current, current)})
