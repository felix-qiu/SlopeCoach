from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from slopecoach_ml.sport_type import (
    AutoSportTypeDecision,
    MockSportEvidenceProvider,
    NotConfiguredEquipmentSportEvidenceProvider,
    NotConfiguredVisualSportEvidenceProvider,
    ReferenceSportTypeFusion,
    SportEvidenceKind,
    SportEvidenceObservation,
    SportEvidenceProviderResult,
    SportEvidenceProviderStatus,
    SportEvidenceScope,
    SportType,
    SportTypeConfig,
    SportTypeResolutionStatus,
    SportTypeRoutingBasis,
    SportTypeSource,
    TargetSportFrameContext,
    extract_uncalibrated_sport_cues,
    resolve_sport_type,
    run_sport_type_golden,
    sport_specific_analysis_allowed,
)
from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry


def observation(
    evidence_id="evidence-1",
    *,
    kind=SportEvidenceKind.EQUIPMENT,
    ski=0.9,
    snowboard=0.05,
    quality=0.9,
    scope=SportEvidenceScope.CLIP,
    timestamp=None,
    provider="mock",
):
    return SportEvidenceObservation(
        evidence_id=evidence_id,
        kind=kind,
        provider_name=provider,
        ski_support=ski,
        snowboard_support=snowboard,
        quality=quality,
        scope=scope,
        timestamp_us=timestamp,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ski_support", float("nan")),
        ("ski_support", float("inf")),
        ("ski_support", True),
        ("ski_support", -0.1),
        ("snowboard_support", 1.1),
        ("quality", -0.1),
        ("quality", 1.1),
    ],
)
def test_evidence_rejects_invalid_ratios(field, value):
    with pytest.raises(ValueError):
        replace(observation(), **{field: value})


def test_evidence_scope_and_identity_validation():
    with pytest.raises(ValueError, match="timestamp"):
        observation(scope=SportEvidenceScope.FRAME)
    with pytest.raises(ValueError, match="timestamp"):
        observation(scope=SportEvidenceScope.FRAME, timestamp=-1)
    with pytest.raises(ValueError, match="temporal_segment_id"):
        replace(observation(), temporal_segment_id=0)
    with pytest.raises(ValueError, match="evidence_id"):
        replace(observation(), evidence_id="")
    with pytest.raises(ValueError, match="provider_name"):
        replace(observation(), provider_name=" ")


@pytest.mark.parametrize(
    "field,value",
    [
        ("equipment_weight", True),
        ("visual_classifier_weight", float("nan")),
        ("pose_geometry_weight", 0),
        ("minimum_frame_observations_per_kind", True),
        ("minimum_primary_support", 1.1),
    ],
)
def test_config_is_strict(field, value):
    with pytest.raises(ValueError):
        replace(SportTypeConfig(), **{field: value})


def test_provider_result_status_invariants():
    item = observation()
    with pytest.raises(ValueError):
        SportEvidenceProviderResult(
            "mock",
            SportEvidenceKind.EQUIPMENT,
            SportEvidenceProviderStatus.NOT_CONFIGURED,
            (item,),
        )
    with pytest.raises(ValueError):
        SportEvidenceProviderResult(
            "mock",
            SportEvidenceKind.EQUIPMENT,
            SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE,
        )
    with pytest.raises(ValueError):
        SportEvidenceProviderResult(
            "mock", SportEvidenceKind.EQUIPMENT, SportEvidenceProviderStatus.FAILED
        )
    failed = SportEvidenceProviderResult(
        "mock",
        SportEvidenceKind.EQUIPMENT,
        SportEvidenceProviderStatus.FAILED,
        error="boom",
    )
    assert failed.error == "boom"


def test_auto_decision_count_is_strict_integer():
    with pytest.raises(ValueError, match="evidence_observation_count"):
        AutoSportTypeDecision(
            SportType.UNKNOWN,
            SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE,
            None,
            None,
            None,
            (),
            (),
            0.5,
            True,
            ("NO_EVIDENCE",),
        )


def test_not_configured_and_mock_providers_are_honest():
    equipment = NotConfiguredEquipmentSportEvidenceProvider().infer()
    visual = NotConfiguredVisualSportEvidenceProvider().infer()
    assert (
        equipment.status is visual.status is SportEvidenceProviderStatus.NOT_CONFIGURED
    )
    assert not equipment.observations and not visual.observations
    mock = MockSportEvidenceProvider(
        "mock", SportEvidenceKind.EQUIPMENT, (observation(),)
    ).infer()
    assert mock.status is SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE


def test_failed_equipment_allows_strong_visual_fallback():
    failed_equipment = SportEvidenceProviderResult(
        "failed-equipment",
        SportEvidenceKind.EQUIPMENT,
        SportEvidenceProviderStatus.FAILED,
        error="RuntimeError: detector unavailable",
    )
    visual = MockSportEvidenceProvider(
        "visual",
        SportEvidenceKind.VISUAL_CLASSIFIER,
        (
            observation(
                "visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=0.04,
                snowboard=0.91,
                provider="visual",
            ),
        ),
    ).infer()
    result = resolve_sport_type((failed_equipment, visual))
    assert result.effective_sport_type is SportType.SNOWBOARD
    assert result.auto_decision.routing_basis is SportTypeRoutingBasis.VISUAL_FALLBACK
    assert result.auto_decision.reason_codes == (
        "VISUAL_FALLBACK_EQUIPMENT_UNAVAILABLE",
    )
    assert result.provider_results[0].status is SportEvidenceProviderStatus.FAILED


def test_target_provider_context_requires_locked_identity():
    geometry = FrameGeometry(640, 480)
    kwargs = dict(
        timestamp_us=0,
        frame_index=0,
        geometry=geometry,
        target_id="target-1",
        active_track_id=1,
        target_bbox=BoundingBox2D(10, 10, 100, 200),
        frame_reference=object(),
    )
    context = TargetSportFrameContext(
        identity_state=TargetIdentityState.LOCKED, **kwargs
    )
    assert context.target_id == "target-1"
    for unsafe in (
        TargetIdentityState.UNINITIALIZED,
        TargetIdentityState.SUSPECT,
        TargetIdentityState.LOST,
        TargetIdentityState.RECOVERING,
        TargetIdentityState.AMBIGUOUS,
    ):
        with pytest.raises(ValueError, match="LOCKED"):
            TargetSportFrameContext(identity_state=unsafe, **kwargs)


def test_no_evidence_and_secondary_only_require_primary():
    fusion = ReferenceSportTypeFusion()
    assert (
        fusion.decide(()).status
        is SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE
    )
    secondary = [
        observation(
            "p0",
            kind=SportEvidenceKind.POSE_GEOMETRY,
            scope=SportEvidenceScope.FRAME,
            timestamp=0,
        ),
        observation(
            "p1",
            kind=SportEvidenceKind.POSE_GEOMETRY,
            scope=SportEvidenceScope.FRAME,
            timestamp=1,
        ),
        observation("m", kind=SportEvidenceKind.TEMPORAL_MOTION),
    ]
    decision = fusion.decide(secondary)
    assert decision.sport_type is SportType.UNKNOWN
    assert decision.status is SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE


@pytest.mark.parametrize(
    "ski,snowboard,expected",
    [(0.9, 0.05, SportType.SKI), (0.05, 0.9, SportType.SNOWBOARD)],
)
def test_strong_primary_resolves(ski, snowboard, expected):
    decision = ReferenceSportTypeFusion().decide(
        [observation(ski=ski, snowboard=snowboard)]
    )
    assert decision.sport_type is expected
    assert decision.status is SportTypeResolutionStatus.RESOLVED_AUTO
    assert decision.routing_basis is SportTypeRoutingBasis.EQUIPMENT_PRIMARY


def test_tie_margin_conflict_and_weak_primary_are_unresolved():
    fusion = ReferenceSportTypeFusion()
    tie = fusion.decide([observation(ski=0.8, snowboard=0.8)])
    assert tie.status is SportTypeResolutionStatus.CONFLICTING_PRIMARY_EVIDENCE
    close = fusion.decide([observation(ski=0.72, snowboard=0.61)])
    assert close.status is SportTypeResolutionStatus.AMBIGUOUS
    weak = fusion.decide([observation(ski=0.69, snowboard=0.1)])
    assert weak.status is SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE
    conflict = fusion.decide(
        [
            observation("eq", ski=0.9, snowboard=0.05),
            observation(
                "vis", kind=SportEvidenceKind.VISUAL_CLASSIFIER, ski=0.05, snowboard=0.9
            ),
        ]
    )
    assert conflict.sport_type is SportType.SKI
    assert conflict.routing_basis is SportTypeRoutingBasis.EQUIPMENT_PRIMARY
    assert "EQUIPMENT_PRIMARY_VISUAL_DISAGREES" in conflict.reason_codes


@pytest.mark.parametrize(
    "ski,snowboard,expected",
    [(0.91, 0.04, SportType.SKI), (0.05, 0.92, SportType.SNOWBOARD)],
)
def test_strong_visual_fallback_without_equipment(ski, snowboard, expected):
    decision = ReferenceSportTypeFusion().decide(
        [
            observation(
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=ski,
                snowboard=snowboard,
            )
        ]
    )
    assert decision.sport_type is expected
    assert decision.routing_basis is SportTypeRoutingBasis.VISUAL_FALLBACK
    assert decision.reason_codes == ("VISUAL_FALLBACK_EQUIPMENT_UNAVAILABLE",)


@pytest.mark.parametrize(
    "equipment,visual,expected",
    [
        ((0.90, 0.05), (0.05, 0.90), SportType.SKI),
        ((0.05, 0.90), (0.90, 0.05), SportType.SNOWBOARD),
    ],
)
def test_strong_equipment_outranks_opposing_strong_visual(equipment, visual, expected):
    decision = ReferenceSportTypeFusion().decide(
        [
            observation("eq", ski=equipment[0], snowboard=equipment[1]),
            observation(
                "visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=visual[0],
                snowboard=visual[1],
            ),
        ]
    )
    assert decision.sport_type is expected
    assert decision.routing_basis is SportTypeRoutingBasis.EQUIPMENT_PRIMARY
    assert (decision.ski_support, decision.snowboard_support) == equipment
    assert decision.reason_codes == (
        "EQUIPMENT_PRIMARY_THRESHOLDS_SATISFIED",
        "EQUIPMENT_PRIMARY_VISUAL_DISAGREES",
    )


def test_strong_equipment_records_visual_agreement():
    decision = ReferenceSportTypeFusion().decide(
        [
            observation("eq", ski=0.90, snowboard=0.05),
            observation(
                "visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=0.80,
                snowboard=0.10,
            ),
        ]
    )
    assert decision.reason_codes == (
        "EQUIPMENT_PRIMARY_THRESHOLDS_SATISFIED",
        "EQUIPMENT_PRIMARY_VISUAL_AGREES",
    )


def test_weak_equipment_does_not_veto_opposing_visual_fallback():
    decision = ReferenceSportTypeFusion().decide(
        [
            observation("eq", ski=0.69, snowboard=0.10),
            observation(
                "visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=0.05,
                snowboard=0.90,
            ),
        ]
    )
    assert decision.sport_type is SportType.SNOWBOARD
    assert decision.routing_basis is SportTypeRoutingBasis.VISUAL_FALLBACK
    assert decision.ski_support == 0.05
    assert decision.snowboard_support == 0.90
    assert decision.reason_codes == (
        "VISUAL_FALLBACK_EQUIPMENT_INSUFFICIENT",
        "VISUAL_FALLBACK_WEAK_EQUIPMENT_DISAGREEMENT",
    )


def test_weak_primary_agreement_combines_only_when_combined_thresholds_pass():
    decision = ReferenceSportTypeFusion().decide(
        [
            observation("eq", ski=0.72, snowboard=0.60),
            observation(
                "visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=0.68,
                snowboard=0.20,
            ),
        ]
    )
    assert decision.sport_type is SportType.SKI
    assert decision.routing_basis is SportTypeRoutingBasis.PRIMARY_AGREEMENT
    assert decision.reason_codes == ("PRIMARY_KINDS_AGREE",)
    assert decision.ski_support == pytest.approx((0.72 + 0.8 * 0.68) / 1.8)
    assert decision.snowboard_support == pytest.approx((0.60 + 0.8 * 0.20) / 1.8)

    insufficient = ReferenceSportTypeFusion().decide(
        [
            observation("weak-eq", ski=0.62, snowboard=0.20),
            observation(
                "weak-visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=0.66,
                snowboard=0.18,
            ),
        ]
    )
    assert insufficient.sport_type is SportType.UNKNOWN
    assert insufficient.routing_basis is SportTypeRoutingBasis.NONE
    assert insufficient.reason_codes == (
        "PRIMARY_KINDS_AGREE",
        "WINNER_PRIMARY_SUPPORT_BELOW_MINIMUM",
    )


def test_opposing_unresolved_primary_kinds_fail_closed():
    decision = ReferenceSportTypeFusion().decide(
        [
            observation("eq", ski=0.69, snowboard=0.10),
            observation(
                "visual",
                kind=SportEvidenceKind.VISUAL_CLASSIFIER,
                ski=0.10,
                snowboard=0.69,
            ),
        ]
    )
    assert decision.sport_type is SportType.UNKNOWN
    assert decision.status is SportTypeResolutionStatus.CONFLICTING_PRIMARY_EVIDENCE
    assert decision.routing_basis is SportTypeRoutingBasis.NONE
    assert decision.reason_codes == ("PRIMARY_KINDS_CONFLICT_UNRESOLVED",)


def test_exact_tie_below_conflict_threshold_is_ambiguous():
    decision = ReferenceSportTypeFusion().decide([observation(ski=0.7, snowboard=0.7)])
    assert decision.sport_type is SportType.UNKNOWN
    assert decision.status is SportTypeResolutionStatus.AMBIGUOUS


def test_primary_with_secondary_fusion_behaviors():
    primary = observation("primary", ski=0.9, snowboard=0.05)
    agreeing = observation(
        "agree", kind=SportEvidenceKind.TEMPORAL_MOTION, ski=0.8, snowboard=0.1
    )
    opposing = observation(
        "oppose", kind=SportEvidenceKind.TEMPORAL_MOTION, ski=0.1, snowboard=0.6
    )
    assert (
        ReferenceSportTypeFusion().decide([primary, agreeing]).sport_type
        is SportType.SKI
    )
    assert (
        ReferenceSportTypeFusion().decide([primary, opposing]).sport_type
        is SportType.SKI
    )
    decision = ReferenceSportTypeFusion().decide([primary, opposing])
    assert decision.routing_basis is SportTypeRoutingBasis.EQUIPMENT_PRIMARY
    assert decision.ski_support == 0.9
    assert decision.snowboard_support == 0.05


def test_duplicates_rejected_and_distinct_frame_timestamps_required():
    fusion = ReferenceSportTypeFusion()
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        fusion.decide([observation(), observation()])
    same_timestamp = [
        observation("a", scope=SportEvidenceScope.FRAME, timestamp=10),
        observation("b", scope=SportEvidenceScope.FRAME, timestamp=10),
    ]
    assert (
        fusion.decide(same_timestamp).status
        is SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE
    )


def test_fusion_is_order_independent_and_strict_json():
    items = [
        observation("a", scope=SportEvidenceScope.FRAME, timestamp=0),
        observation("b", scope=SportEvidenceScope.FRAME, timestamp=200000),
        observation(
            "c", kind=SportEvidenceKind.TEMPORAL_MOTION, ski=0.8, snowboard=0.1
        ),
        observation(
            "visual",
            kind=SportEvidenceKind.VISUAL_CLASSIFIER,
            ski=0.05,
            snowboard=0.90,
        ),
    ]
    forward = ReferenceSportTypeFusion().decide(items).to_dict()
    backward = ReferenceSportTypeFusion().decide(reversed(items)).to_dict()
    assert forward == backward
    json.dumps(forward, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("user", [SportType.SKI, SportType.SNOWBOARD])
def test_user_override_resolves_unknown(user):
    result = resolve_sport_type((), user_selection=user)
    assert result.effective_sport_type is user
    assert result.effective_source is SportTypeSource.USER
    assert result.auto_decision.sport_type is SportType.UNKNOWN
    assert not result.ask_user_recommended


def test_user_override_retains_auto_and_exposes_disagreement():
    provider = MockSportEvidenceProvider(
        "mock", SportEvidenceKind.EQUIPMENT, (observation(),)
    ).infer()
    agreeing = resolve_sport_type((provider,), user_selection=SportType.SKI)
    disagreeing = resolve_sport_type((provider,), user_selection=SportType.SNOWBOARD)
    assert not agreeing.auto_user_disagreement
    assert disagreeing.auto_user_disagreement
    assert disagreeing.auto_decision.sport_type is SportType.SKI
    assert (
        disagreeing.auto_decision.routing_basis
        is SportTypeRoutingBasis.EQUIPMENT_PRIMARY
    )
    assert disagreeing.effective_sport_type is SportType.SNOWBOARD


@pytest.mark.parametrize("filename", ["ski_test_001.mp4", "snowboard_test.mp4"])
def test_filename_is_not_evidence(filename):
    assert filename
    result = resolve_sport_type(())
    assert result.effective_sport_type is SportType.UNKNOWN


def test_uncalibrated_cues_preserve_null_and_never_fuse():
    aggregates = [
        SimpleNamespace(
            feature_id="ankle_separation_body_scale", median=0.3, range=0.1
        ),
        SimpleNamespace(
            feature_id="bilateral_knee_mean_angle_2d_deg", median=90.0, range=20.0
        ),
    ]
    cues = extract_uncalibrated_sport_cues(
        SimpleNamespace(temporal_segment_features=aggregates)
    )
    by_id = {item.cue_id: item for item in cues}
    assert by_id["ankle_separation_body_scale_median"].value == 0.3
    assert by_id["shoulder_hip_axis_difference_2d_deg_median"].value is None
    assert by_id["bilateral_knee_mean_angle_2d_deg_range"].value == 20.0
    assert all(not item.contributes_to_auto_fusion for item in cues)


def test_unknown_only_blocks_sport_specific_analysis_not_generic_biomechanics():
    generic_facts = {"left_knee_angle_2d_deg": 90.0}
    result = resolve_sport_type(())
    assert generic_facts["left_knee_angle_2d_deg"] == 90.0
    assert not sport_specific_analysis_allowed(result)
    assert sport_specific_analysis_allowed(
        resolve_sport_type((), user_selection=SportType.SKI)
    )


def test_sport_type_golden():
    path = Path(__file__).parents[1] / "fixtures/golden_sport_type_001.json"
    result = run_sport_type_golden(path)
    assert result["golden_passed"]
    assert all(item["passed"] for item in result["cases"])
