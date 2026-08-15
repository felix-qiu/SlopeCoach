from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.benchmark import SportTypeBenchmarkCollector
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry
from slopecoach_ml.sport_type import (
    AutoSportTypeDecision,
    EquipmentSportEvidenceConfig,
    MMDetEquipmentSportEvidenceProvider,
    ReferenceSportTypeFusion,
    SportEvidenceKind,
    SportEvidenceProviderStatus,
    SportType,
    SportTypeResolutionStatus,
    TargetSportFrameContext,
    equipment_crop_bbox,
    execute_sport_evidence_providers,
    select_equipment_contexts,
    summarize_provider_kind,
)


class FakeImage:
    size = 1

    def __getitem__(self, key):
        return self

    def copy(self):
        return FakeImage()


class FakeBackend:
    def __init__(self, classes, outputs=()):
        self.class_names = tuple(classes)
        self.outputs = list(outputs)
        self.calls = 0

    def infer(self, image):
        output = self.outputs[self.calls] if self.calls < len(self.outputs) else ()
        self.calls += 1
        if isinstance(output, Exception):
            raise output
        return tuple(output)


def context(timestamp=0, frame_index=0, bbox=None):
    return TargetSportFrameContext(
        timestamp,
        frame_index,
        FrameGeometry(640, 480),
        "target-1",
        1,
        bbox or BoundingBox2D(100, 100, 100, 200),
        TargetIdentityState.LOCKED,
        FakeImage(),
    )


def detection(label, score=0.9, *, inside=True):
    # Default crop is x=50..250, y=70..390. The mapped center is (130, 200)
    # when inside and (55, 85) when outside the lower-body association zone.
    return (60, 120, 100, 140, score, label) if inside else (0, 0, 10, 30, score, label)


@pytest.mark.parametrize(
    "classes,skis_index,snowboard_index",
    [
        (("person", "snowboard", "dog", "skis"), 3, 1),
        (("skis", "dog", "snowboard", "person"), 0, 2),
    ],
)
def test_dynamic_class_lookup(classes, skis_index, snowboard_index):
    provider = MMDetEquipmentSportEvidenceProvider(FakeBackend(classes))
    assert provider.skis_label_index == skis_index
    assert provider.snowboard_label_index == snowboard_index


def test_missing_class_map_fails_clearly():
    with pytest.raises(RuntimeError, match="EQUIPMENT_CLASS_MAP_UNSUPPORTED"):
        MMDetEquipmentSportEvidenceProvider(FakeBackend(("person", "skis")))


@pytest.mark.parametrize(
    "class_name,score,expected",
    [("skis", 0.90, SportType.SKI), ("snowboard", 0.92, SportType.SNOWBOARD)],
)
def test_two_associated_frames_resolve_through_existing_fusion(
    class_name, score, expected
):
    classes = ("person", "snowboard", "dog", "skis")
    label = classes.index(class_name)
    backend = FakeBackend(
        classes, ((detection(label, score),), (detection(label, score),))
    )
    provider = MMDetEquipmentSportEvidenceProvider(backend)
    result = provider.infer((context(0, 0), context(200000, 1)))
    assert result.status is SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE
    assert len(result.observations) == 2
    decision = ReferenceSportTypeFusion().decide(result.observations)
    assert decision.sport_type is expected
    assert decision.status is SportTypeResolutionStatus.RESOLVED_AUTO


def test_one_frame_remains_insufficient():
    classes = ("skis", "snowboard")
    provider = MMDetEquipmentSportEvidenceProvider(
        FakeBackend(classes, ((detection(0),),))
    )
    result = provider.infer((context(),))
    decision = ReferenceSportTypeFusion().decide(result.observations)
    assert decision.sport_type is SportType.UNKNOWN
    assert decision.status is SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE


def test_no_equipment_and_below_threshold_emit_no_observation():
    classes = ("skis", "snowboard")
    no_equipment = MMDetEquipmentSportEvidenceProvider(FakeBackend(classes, ((), ())))
    result = no_equipment.infer((context(0, 0), context(1, 1)))
    assert result.status is SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE
    assert not result.observations
    below = MMDetEquipmentSportEvidenceProvider(
        FakeBackend(classes, ((detection(0, 0.20),),))
    ).infer((context(),))
    assert below.status is SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE


def test_bystander_rejected_and_associated_detection_maps_to_source():
    classes = ("skis", "snowboard")
    provider = MMDetEquipmentSportEvidenceProvider(
        FakeBackend(classes, ((detection(0, inside=False),), (detection(0),)))
    )
    result = provider.infer((context(0, 0), context(1, 1)))
    assert len(result.observations) == 1
    first, second = provider.last_debug_frames
    assert first["associated_skis_count"] == 0
    assert second["associated_skis_count"] == 1
    assert second["associated_detections"][0]["bbox"]["x_px"] == 110.0
    assert equipment_crop_bbox(context(), EquipmentSportEvidenceConfig()).x_px == 50.0


def test_context_selection_is_deterministic_distinct_and_evenly_spaced():
    config = replace(EquipmentSportEvidenceConfig(), max_frame_contexts=4)
    items = [context(index * 100, index) for index in range(10)]
    first = select_equipment_contexts(items, config)[1]
    reversed_result = select_equipment_contexts(reversed(items), config)[1]
    assert [item.timestamp_us for item in first] == [0, 300, 600, 900]
    assert [item.timestamp_us for item in reversed_result] == [0, 300, 600, 900]


def test_benchmark_collector_retains_only_locked_target_pixels():
    collector = SportTypeBenchmarkCollector()
    frame = SimpleNamespace(
        timestamp_us=0,
        frame_index=0,
        geometry=FrameGeometry(640, 480),
        image=FakeImage(),
    )
    base = {
        "timestamp_us": 0,
        "frame_index": 0,
        "target_id": "target-1",
        "active_track_id": 1,
        "identity_confidence": 0.9,
        "selected_bbox": BoundingBox2D(100, 100, 100, 200).to_dict(),
        "tracks": [{"track_id": 1, "detection_id": 1}],
        "limitations": [],
    }
    collector.observe(frame, {**base, "identity_state": "LOCKED"}, None)
    collector.observe(
        frame,
        {**base, "identity_state": "SUSPECT", "selected_bbox": None},
        None,
    )
    assert len(collector.samples) == 2
    assert len(collector.sport_contexts) == 1
    assert collector.sport_contexts[0].identity_state is TargetIdentityState.LOCKED


def test_provider_failure_isolated_and_other_evidence_survives():
    classes = ("skis", "snowboard")
    failed = MMDetEquipmentSportEvidenceProvider(
        FakeBackend(classes, (RuntimeError("inference boom"),))
    )
    valid = MMDetEquipmentSportEvidenceProvider(
        FakeBackend(classes, ((detection(0),), (detection(0),)))
    )
    valid.name = "second-equipment-provider"
    results = execute_sport_evidence_providers(
        (failed, valid), (context(0, 0), context(1, 1))
    )
    assert results[0].status is SportEvidenceProviderStatus.FAILED
    assert results[0].error
    assert results[1].status is SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE
    assert (
        ReferenceSportTypeFusion().decide(results[1].observations).sport_type
        is SportType.SKI
    )


def test_multiple_same_kind_provider_summary_uses_all_results():
    from slopecoach_ml.sport_type import NotConfiguredEquipmentSportEvidenceProvider

    classes = ("skis", "snowboard")
    configured = MMDetEquipmentSportEvidenceProvider(
        FakeBackend(classes, ((detection(0),), (detection(0),)))
    )
    results = execute_sport_evidence_providers(
        (NotConfiguredEquipmentSportEvidenceProvider(), configured),
        (context(0, 0), context(1, 1)),
    )
    summary = summarize_provider_kind(results, SportEvidenceKind.EQUIPMENT)
    assert summary["overall_status"] == "EXECUTED_WITH_EVIDENCE"
    assert summary["provider_with_evidence_count"] == 1
    assert summary["observation_count"] == 2


def auto_decision(**overrides):
    values = dict(
        sport_type=SportType.SKI,
        status=SportTypeResolutionStatus.RESOLVED_AUTO,
        ski_support=0.9,
        snowboard_support=0.1,
        margin=0.8,
        active_evidence_kinds=(SportEvidenceKind.EQUIPMENT,),
        primary_evidence_kinds=(SportEvidenceKind.EQUIPMENT,),
        evidence_observation_count=2,
        ask_user_recommended=False,
        reason_codes=("AUTO_THRESHOLDS_SATISFIED",),
    )
    values.update(overrides)
    return AutoSportTypeDecision(**values)


def test_auto_decision_hardening():
    with pytest.raises(ValueError, match="RESOLVED_USER"):
        auto_decision(
            sport_type=SportType.UNKNOWN,
            status=SportTypeResolutionStatus.RESOLVED_USER,
            ask_user_recommended=True,
        )
    with pytest.raises(ValueError, match="margin"):
        auto_decision(margin=0.7)
    with pytest.raises(ValueError, match="resolved SKI"):
        auto_decision(ski_support=0.1, snowboard_support=0.9)
    with pytest.raises(ValueError, match="resolved SNOWBOARD"):
        auto_decision(
            sport_type=SportType.SNOWBOARD,
            ski_support=0.9,
            snowboard_support=0.1,
        )
    assert auto_decision().sport_type is SportType.SKI
