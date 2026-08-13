"""A4.1-gated, signal-run-local turn biomechanics facts."""

from __future__ import annotations

from slopecoach_ml.turns import TurnSegment, TurnSegmentStatus

from .contracts import (
    BiomechanicsFact,
    BiomechanicsFactStatus,
    BiomechanicsFeatureConfig,
    TurnBiomechanicsResult,
)
from .registry import FEATURE_BY_ID


def _complete_turn_window(turn: TurnSegment) -> bool:
    """A complete window has ordered start/apex/end, positive duration, and a run ID."""
    return (
        turn.start_timestamp_us is not None
        and turn.end_timestamp_us is not None
        and turn.start_timestamp_us <= turn.apex_timestamp_us <= turn.end_timestamp_us
        and turn.duration_us is not None
        and turn.duration_us > 0
        and isinstance(turn.signal_run_id, int)
        and not isinstance(turn.signal_run_id, bool)
        and turn.signal_run_id > 0
    )


def _source_evidence(source: BiomechanicsFact | None) -> dict[str, object]:
    if source is None:
        return {}
    return {
        "support_confidence": source.support_confidence,
        "required_joints": source.required_joints,
        "observed_joint_count": source.observed_joint_count,
        "interpolated_joint_count": source.interpolated_joint_count,
    }


def _delta_evidence(
    first: BiomechanicsFact | None, second: BiomechanicsFact | None
) -> dict[str, object]:
    if first is None or second is None:
        return {}
    required = tuple(dict.fromkeys(first.required_joints + second.required_joints))
    confidences = (first.support_confidence, second.support_confidence)
    # Counts describe conservative unique-joint support, not observations across timestamps.
    interpolated = min(
        len(required), max(first.interpolated_joint_count, second.interpolated_joint_count)
    )
    observed = min(
        len(required) - interpolated,
        first.observed_joint_count,
        second.observed_joint_count,
    )
    return {
        "support_confidence": min(confidences) if None not in confidences else None,
        "required_joints": required,
        "observed_joint_count": observed,
        "interpolated_joint_count": interpolated,
    }


def compute_turn_biomechanics(
    turns: list[TurnSegment],
    frame_facts: tuple[BiomechanicsFact, ...],
    run_timestamps: dict[int, set[int]],
    config: BiomechanicsFeatureConfig,
) -> tuple[TurnBiomechanicsResult, ...]:
    config.validate()
    results = []
    for turn in sorted(turns, key=lambda item: (item.apex_timestamp_us, item.turn_id)):
        if turn.status not in (TurnSegmentStatus.VALID, TurnSegmentStatus.PARTIAL):
            continue
        allowed = run_timestamps.get(turn.signal_run_id, set())
        facts = [
            fact
            for fact in frame_facts
            if fact.temporal_segment_id == turn.temporal_segment_id
            and fact.timestamp_us in allowed
            and (turn.start_timestamp_us is None or fact.timestamp_us >= turn.start_timestamp_us)
            and (turn.end_timestamp_us is None or fact.timestamp_us <= turn.end_timestamp_us)
        ]
        by_feature: dict[str, list[BiomechanicsFact]] = {}
        for fact in facts:
            by_feature.setdefault(fact.feature_id, []).append(fact)

        def closest(
            feature_id: str,
            timestamp_us: int | None,
            tolerance_us: int,
            feature_map=by_feature,
        ):
            if timestamp_us is None:
                return None
            candidates = [
                fact
                for fact in feature_map.get(feature_id, [])
                if fact.status is BiomechanicsFactStatus.AVAILABLE and fact.timestamp_us is not None
            ]
            match = min(
                candidates,
                key=lambda fact: (abs(fact.timestamp_us - timestamp_us), fact.timestamp_us),
                default=None,
            )
            return (
                match
                if match is not None and abs(match.timestamp_us - timestamp_us) <= tolerance_us
                else None
            )

        def out(
            feature_id: str,
            value: float | int | None,
            status: BiomechanicsFactStatus = BiomechanicsFactStatus.AVAILABLE,
            *,
            evidence: dict[str, object] | None = None,
            current_turn=turn,
        ) -> BiomechanicsFact:
            definition = FEATURE_BY_ID[feature_id]
            return BiomechanicsFact(
                feature_id=feature_id,
                family=definition.family,
                scope=definition.scope,
                unit=definition.unit,
                value=value,
                status=status,
                temporal_segment_id=current_turn.temporal_segment_id,
                signal_run_id=current_turn.signal_run_id,
                turn_id=current_turn.turn_id,
                limitations=definition.limitations,
                **(evidence or {}),
            )

        apex_mean = closest(
            "bilateral_knee_mean_angle_2d_deg",
            turn.apex_timestamp_us,
            config.apex_match_tolerance_us,
        )
        apex_diff = closest(
            "bilateral_knee_abs_difference_2d_deg",
            turn.apex_timestamp_us,
            config.apex_match_tolerance_us,
        )
        apex_ankle = closest(
            "ankle_separation_body_scale",
            turn.apex_timestamp_us,
            config.apex_match_tolerance_us,
        )
        start = closest(
            "bilateral_knee_mean_angle_2d_deg",
            turn.start_timestamp_us,
            config.boundary_match_tolerance_us,
        )
        end = closest(
            "bilateral_knee_mean_angle_2d_deg",
            turn.end_timestamp_us,
            config.boundary_match_tolerance_us,
        )
        unavailable = BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE
        boundary = BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE
        complete = _complete_turn_window(turn) and bool(allowed)
        turn_facts = [
            out(
                "turn_duration_us",
                turn.duration_us if complete else None,
                boundary if not complete else BiomechanicsFactStatus.AVAILABLE,
            ),
            out("turn_peak_lateral_proxy", turn.peak_value),
        ]
        sources = (
            ("bilateral_knee_mean_angle_at_apex_deg", apex_mean, False),
            ("bilateral_knee_abs_difference_at_apex_deg", apex_diff, False),
            ("ankle_separation_at_apex_body_scale", apex_ankle, False),
            ("bilateral_knee_mean_angle_at_start_deg", start, turn.start_timestamp_us is None),
            ("bilateral_knee_mean_angle_at_end_deg", end, turn.end_timestamp_us is None),
        )
        for feature_id, source, missing_boundary in sources:
            turn_facts.append(
                out(
                    feature_id,
                    source.value if source else None,
                    BiomechanicsFactStatus.AVAILABLE
                    if source
                    else boundary
                    if missing_boundary
                    else unavailable,
                    evidence=_source_evidence(source),
                )
            )

        start_delta = float(apex_mean.value) - float(start.value) if apex_mean and start else None
        end_delta = float(end.value) - float(apex_mean.value) if apex_mean and end else None
        turn_facts.extend(
            (
                out(
                    "knee_angle_change_start_to_apex_deg",
                    start_delta,
                    BiomechanicsFactStatus.AVAILABLE
                    if start_delta is not None
                    else boundary
                    if turn.start_timestamp_us is None
                    else unavailable,
                    evidence=_delta_evidence(start, apex_mean),
                ),
                out(
                    "knee_angle_change_apex_to_end_deg",
                    end_delta,
                    BiomechanicsFactStatus.AVAILABLE
                    if end_delta is not None
                    else boundary
                    if turn.end_timestamp_us is None
                    else unavailable,
                    evidence=_delta_evidence(apex_mean, end),
                ),
            )
        )

        knee = (
            [
                fact
                for fact in by_feature.get("bilateral_knee_mean_angle_2d_deg", [])
                if fact.status is BiomechanicsFactStatus.AVAILABLE
            ]
            if complete
            else []
        )
        minimum = min(knee, key=lambda fact: (fact.value, fact.timestamp_us), default=None)
        offset = minimum.timestamp_us - turn.apex_timestamp_us if minimum else None
        timing_status = (
            BiomechanicsFactStatus.AVAILABLE
            if minimum is not None
            else unavailable
            if complete
            else boundary
        )
        minimum_evidence = _source_evidence(minimum)
        turn_facts.extend(
            (
                out(
                    "minimum_mean_knee_angle_timestamp_us",
                    minimum.timestamp_us if minimum else None,
                    timing_status,
                    evidence=minimum_evidence,
                ),
                out(
                    "minimum_mean_knee_angle_offset_from_apex_us",
                    offset,
                    timing_status,
                    evidence=minimum_evidence,
                ),
                out(
                    "minimum_mean_knee_angle_phase_offset",
                    offset / turn.duration_us if offset is not None and complete else None,
                    timing_status,
                    evidence=minimum_evidence,
                ),
            )
        )
        results.append(
            TurnBiomechanicsResult(
                turn.turn_id,
                turn.temporal_segment_id,
                turn.signal_run_id,
                turn.phase_sign.value,
                tuple(turn_facts),
            )
        )
    return tuple(results)
