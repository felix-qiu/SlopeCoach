from __future__ import annotations

import math
from dataclasses import replace

import pytest

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry
from slopecoach_ml.sport_type import (
    VISUAL_SPORT_PROMPT_SCHEMA_VERSION,
    VISUAL_SPORT_PROMPTS,
    ClipVisualSportEvidenceProvider,
    FakeVisualSportBackend,
    ReferenceSportTypeFusion,
    SportEvidenceKind,
    SportEvidenceObservation,
    SportEvidenceProviderResult,
    SportEvidenceProviderStatus,
    SportEvidenceScope,
    SportType,
    SportTypeResolutionStatus,
    SportTypeRoutingBasis,
    TargetSportFrameContext,
    VisualSportEvidenceConfig,
    VisualSportScores,
    bgr_to_rgb,
    execute_sport_evidence_providers,
    visual_crop_bbox,
    visual_prompt_sha256,
)


class Crop:
    size = 1

    def __init__(self, key=None):
        self.key = key

    def __getitem__(self, key):
        return Crop(key)

    def copy(self):
        return self


def context(timestamp=0, frame_index=0, bbox=None):
    return TargetSportFrameContext(
        timestamp,
        frame_index,
        FrameGeometry(640, 480),
        "target-1",
        1,
        bbox or BoundingBox2D(100, 100, 100, 200),
        TargetIdentityState.LOCKED,
        Crop(),
    )


def scores(ski, snowboard, neutral):
    return VisualSportScores(ski, snowboard, neutral)


@pytest.mark.parametrize(
    "output,expected",
    [
        (scores(0.90, 0.04, 0.06), SportType.SKI),
        (scores(0.04, 0.91, 0.05), SportType.SNOWBOARD),
    ],
)
def test_two_visual_frames_resolve_strong_sport(output, expected):
    provider = ClipVisualSportEvidenceProvider(FakeVisualSportBackend([output, output]))
    result = provider.infer((context(0, 0), context(200000, 1)))
    assert result.observations[0].evidence_id == (
        "VISUAL_CLASSIFIER:openai-clip-vit-b32-visual-sport:0:0"
    )
    decision = ReferenceSportTypeFusion().decide(result.observations)
    assert decision.sport_type is expected
    assert decision.status is SportTypeResolutionStatus.RESOLVED_AUTO
    assert decision.primary_evidence_kinds == (SportEvidenceKind.VISUAL_CLASSIFIER,)


def test_neutral_abstains_and_one_frame_is_inactive():
    neutral = scores(0.10, 0.12, 0.78)
    provider = ClipVisualSportEvidenceProvider(
        FakeVisualSportBackend([neutral, neutral])
    )
    decision = ReferenceSportTypeFusion().decide(
        provider.infer((context(0, 0), context(200000, 1))).observations
    )
    assert decision.sport_type is SportType.UNKNOWN
    assert decision.status is SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE

    strong = ClipVisualSportEvidenceProvider(
        FakeVisualSportBackend([scores(0.02, 0.95, 0.03)])
    )
    one = ReferenceSportTypeFusion().decide(strong.infer((context(),)).observations)
    assert one.sport_type is SportType.UNKNOWN
    assert not one.primary_evidence_kinds


def observation(evidence_id, kind, ski, snowboard, timestamp):
    return SportEvidenceObservation(
        evidence_id,
        kind,
        "provider-" + kind.value.lower(),
        ski,
        snowboard,
        1.0,
        SportEvidenceScope.FRAME,
        timestamp_us=timestamp,
    )


def test_cross_primary_agreement_conflict_and_weak_equipment_math():
    agreeing = [
        observation(f"e{t}", SportEvidenceKind.EQUIPMENT, 0.02, 0.60, t) for t in (0, 1)
    ] + [
        observation(f"v{t}", SportEvidenceKind.VISUAL_CLASSIFIER, 0.03, 0.90, t)
        for t in (0, 1)
    ]
    agreed = ReferenceSportTypeFusion().decide(agreeing)
    assert agreed.sport_type is SportType.SNOWBOARD
    assert agreed.routing_basis is SportTypeRoutingBasis.VISUAL_FALLBACK
    assert math.isclose(agreed.snowboard_support, 0.90)

    conflicting = [
        observation(f"ce{t}", SportEvidenceKind.EQUIPMENT, 0.90, 0.02, t)
        for t in (0, 1)
    ] + [
        observation(f"cv{t}", SportEvidenceKind.VISUAL_CLASSIFIER, 0.02, 0.90, t)
        for t in (0, 1)
    ]
    conflict = ReferenceSportTypeFusion().decide(conflicting)
    assert conflict.sport_type is SportType.SKI
    assert conflict.routing_basis is SportTypeRoutingBasis.EQUIPMENT_PRIMARY

    weak = [
        observation(f"we{t}", SportEvidenceKind.EQUIPMENT, 0.02, 0.40, t)
        for t in (0, 1)
    ] + [
        observation(f"wv{t}", SportEvidenceKind.VISUAL_CLASSIFIER, 0.03, 0.90, t)
        for t in (0, 1)
    ]
    weak_decision = ReferenceSportTypeFusion().decide(weak)
    assert math.isclose(weak_decision.snowboard_support, 0.90)
    assert weak_decision.sport_type is SportType.SNOWBOARD
    assert weak_decision.routing_basis is SportTypeRoutingBasis.VISUAL_FALLBACK


def test_visual_failure_isolated_from_other_provider():
    visual = ClipVisualSportEvidenceProvider(
        FakeVisualSportBackend([RuntimeError("clip inference failed")])
    )

    class Equipment:
        name = "equipment-provider"
        kind = SportEvidenceKind.EQUIPMENT
        execution_scope = "FRAME"

        def infer(self, contexts):
            items = tuple(
                SportEvidenceObservation(
                    f"equipment:{item.timestamp_us}",
                    self.kind,
                    self.name,
                    0.9,
                    0.01,
                    1.0,
                    SportEvidenceScope.FRAME,
                    timestamp_us=item.timestamp_us,
                )
                for item in contexts
            )
            return SportEvidenceProviderResult(
                self.name,
                self.kind,
                SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE,
                items,
            )

    results = execute_sport_evidence_providers(
        (Equipment(), visual), (context(0, 0), context(1, 1))
    )
    assert results[0].status.value == "EXECUTED_WITH_EVIDENCE"
    assert results[1].status.value == "FAILED"
    assert (
        ReferenceSportTypeFusion().decide(results[0].observations).sport_type
        is SportType.SKI
    )


def test_provider_uses_target_crop_not_full_frame():
    backend = FakeVisualSportBackend([scores(0.9, 0.04, 0.06)])
    provider = ClipVisualSportEvidenceProvider(backend)
    item = context()
    expected = visual_crop_bbox(item, VisualSportEvidenceConfig())
    provider.infer((item,))
    crop = backend.calls[0]
    y_slice, x_slice = crop.key
    assert (x_slice.start, x_slice.stop) == (
        round(expected.x_px),
        round(expected.x_px + expected.width_px),
    )
    assert (y_slice.start, y_slice.stop) == (
        round(expected.y_px),
        round(expected.y_px + expected.height_px),
    )
    assert expected.width_px < item.geometry.width_px
    assert expected.height_px < item.geometry.height_px


def test_context_selection_shared_and_deterministic():
    config = replace(VisualSportEvidenceConfig(), max_frame_contexts=4)
    backend = FakeVisualSportBackend([scores(0.9, 0.04, 0.06)] * 4)
    provider = ClipVisualSportEvidenceProvider(backend, config)
    provider.infer(tuple(context(index * 100, index) for index in reversed(range(10))))
    assert [item["timestamp_us"] for item in provider.last_debug_frames] == [
        0,
        300,
        600,
        900,
    ]


def test_prompt_schema_order_and_fingerprint_snapshot():
    assert VISUAL_SPORT_PROMPT_SCHEMA_VERSION == "visual-sport-prompts-v1"
    assert tuple(VISUAL_SPORT_PROMPTS) == ("SKI", "SNOWBOARD", "NEUTRAL")
    assert tuple(map(len, VISUAL_SPORT_PROMPTS.values())) == (4, 4, 4)
    assert (
        visual_prompt_sha256()
        == "6b27ff66445915e37235acd7eca8bb3148be3c20d5802e502171da6a486d1fdf"
    )
    assert visual_prompt_sha256() == visual_prompt_sha256()


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 0.5, 0.5),
        (math.inf, 0.0, 0.0),
        (-0.1, 0.5, 0.6),
        (1.1, 0.0, -0.1),
        (0.3, 0.3, 0.3),
    ],
)
def test_visual_support_contract_rejects_invalid(values):
    with pytest.raises(ValueError):
        VisualSportScores(*values)


class ColorArray:
    ndim = 3
    shape = (1, 1, 3)

    def __init__(self, marker="BGR"):
        self.marker = marker

    def __getitem__(self, key):
        assert key == (Ellipsis, slice(None, None, -1))
        return ColorArray("RGB")

    def copy(self):
        return self


def test_bgr_to_rgb_conversion_path():
    assert bgr_to_rgb(ColorArray()).marker == "RGB"
