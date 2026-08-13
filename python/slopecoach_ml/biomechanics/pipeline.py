"""A5 temporal biomechanics reference pipeline."""

from __future__ import annotations

from slopecoach_ml.temporal import StabilizedPoseSample, segment_body_scales
from slopecoach_ml.turns import TurnSegment, ValidSignalRun

from .contracts import BiomechanicsFeatureConfig, TemporalBiomechanicsResult
from .frame_features import compute_frame_biomechanics
from .registry import BIOMECHANICS_FEATURE_SCHEMA_VERSION, FEATURE_REGISTRY_SHA256
from .temporal_features import aggregate_frame_facts, derivative_aggregates, feature_coverage
from .turn_features import compute_turn_biomechanics


def analyze_temporal_biomechanics(
    samples: list[StabilizedPoseSample],
    turns: list[TurnSegment] | None = None,
    signal_runs: list[ValidSignalRun] | None = None,
    config: BiomechanicsFeatureConfig | None = None,
) -> TemporalBiomechanicsResult:
    settings = config or BiomechanicsFeatureConfig()
    settings.validate()
    scales = segment_body_scales(samples)
    frame_facts = tuple(
        fact
        for sample in samples
        if sample.temporal_segment_id is not None
        for fact in compute_frame_biomechanics(
            sample, scales.get(sample.temporal_segment_id), settings
        )
    )
    aggregates = aggregate_frame_facts(frame_facts, settings) + derivative_aggregates(
        frame_facts, settings
    )
    run_timestamps = {
        run.signal_run_id: {sample.timestamp_us for _, sample in run.indexed_samples}
        for run in signal_runs or []
    }
    turn_features = compute_turn_biomechanics(turns or [], frame_facts, run_timestamps, settings)
    return TemporalBiomechanicsResult(
        contract_version="temporal-biomechanics-v2",
        feature_schema_version=BIOMECHANICS_FEATURE_SCHEMA_VERSION,
        feature_registry_sha256=FEATURE_REGISTRY_SHA256,
        config=settings,
        frame_facts=frame_facts,
        temporal_segment_features=aggregates,
        turn_features=turn_features,
        feature_coverage=feature_coverage(frame_facts),
    )
