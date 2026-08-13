"""A4.1-gated turn-local biomechanics facts."""

from __future__ import annotations

from slopecoach_ml.turns import TurnSegment, TurnSegmentStatus

from .contracts import (
    BiomechanicsFact,
    BiomechanicsFactScope,
    BiomechanicsFactStatus,
    BiomechanicsFeatureConfig,
    BiomechanicsFeatureFamily,
    TurnBiomechanicsResult,
)


def compute_turn_biomechanics(
    turns: list[TurnSegment],
    frame_facts: tuple[BiomechanicsFact, ...],
    run_timestamps: dict[int, set[int]],
    config: BiomechanicsFeatureConfig,
) -> tuple[TurnBiomechanicsResult, ...]:
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
        by_feature = {}
        for fact in facts:
            by_feature.setdefault(fact.feature_id, []).append(fact)

        def closest(feature, timestamp, feature_map=by_feature):
            if timestamp is None:
                return None
            candidates = [
                fact
                for fact in feature_map.get(feature, [])
                if fact.status is BiomechanicsFactStatus.AVAILABLE
            ]
            match = min(
                candidates,
                key=lambda fact: (abs(fact.timestamp_us - timestamp), fact.timestamp_us),
                default=None,
            )
            return (
                match
                if match and abs(match.timestamp_us - timestamp) <= config.apex_match_tolerance_us
                else None
            )

        def out(
            fid,
            family,
            unit,
            value,
            status=BiomechanicsFactStatus.AVAILABLE,
            current_turn=turn,
        ):
            return BiomechanicsFact(
                fid,
                family,
                BiomechanicsFactScope.TURN,
                unit,
                value,
                status,
                temporal_segment_id=current_turn.temporal_segment_id,
                signal_run_id=current_turn.signal_run_id,
                turn_id=current_turn.turn_id,
                limitations=("IMAGE_SPACE_2D_ONLY_NOT_PHYSICAL_3D",),
            )

        apex_mean = closest("bilateral_knee_mean_angle_2d_deg", turn.apex_timestamp_us)
        apex_diff = closest("bilateral_knee_abs_difference_2d_deg", turn.apex_timestamp_us)
        apex_ankle = closest("ankle_separation_body_scale", turn.apex_timestamp_us)
        start = closest("bilateral_knee_mean_angle_2d_deg", turn.start_timestamp_us)
        end = closest("bilateral_knee_mean_angle_2d_deg", turn.end_timestamp_us)
        unavailable = BiomechanicsFactStatus.INSUFFICIENT_EVIDENCE
        turn_facts = [
            out(
                "turn_duration_us",
                BiomechanicsFeatureFamily.TIMING_PROXY,
                "us",
                turn.duration_us,
                BiomechanicsFactStatus.AVAILABLE
                if turn.duration_us is not None
                else BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE,
            ),
            out(
                "turn_peak_lateral_proxy",
                BiomechanicsFeatureFamily.EDGE_CONTROL_PROXY,
                "ratio",
                turn.peak_value,
            ),
        ]
        for fid, source, family, unit in (
            (
                "bilateral_knee_mean_angle_at_apex_deg",
                apex_mean,
                BiomechanicsFeatureFamily.STANCE_PROXY,
                "deg",
            ),
            (
                "bilateral_knee_abs_difference_at_apex_deg",
                apex_diff,
                BiomechanicsFeatureFamily.SYMMETRY_PROXY,
                "deg",
            ),
            (
                "ankle_separation_at_apex_body_scale",
                apex_ankle,
                BiomechanicsFeatureFamily.STANCE_PROXY,
                "ratio",
            ),
            (
                "bilateral_knee_mean_angle_at_start_deg",
                start,
                BiomechanicsFeatureFamily.STANCE_PROXY,
                "deg",
            ),
            (
                "bilateral_knee_mean_angle_at_end_deg",
                end,
                BiomechanicsFeatureFamily.STANCE_PROXY,
                "deg",
            ),
        ):
            turn_facts.append(
                out(
                    fid,
                    family,
                    unit,
                    source.value if source else None,
                    BiomechanicsFactStatus.AVAILABLE
                    if source
                    else (
                        BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE
                        if "start" in fid
                        and turn.start_timestamp_us is None
                        or "end" in fid
                        and turn.end_timestamp_us is None
                        else unavailable
                    ),
                )
            )
        turn_facts.append(
            out(
                "knee_angle_change_start_to_apex_deg",
                BiomechanicsFeatureFamily.TIMING_PROXY,
                "deg",
                apex_mean.value - start.value if apex_mean and start else None,
                BiomechanicsFactStatus.AVAILABLE if apex_mean and start else unavailable,
            )
        )
        turn_facts.append(
            out(
                "knee_angle_change_apex_to_end_deg",
                BiomechanicsFeatureFamily.TIMING_PROXY,
                "deg",
                end.value - apex_mean.value if apex_mean and end else None,
                BiomechanicsFactStatus.AVAILABLE if apex_mean and end else unavailable,
            )
        )
        knee = [
            fact
            for fact in by_feature.get("bilateral_knee_mean_angle_2d_deg", [])
            if fact.status is BiomechanicsFactStatus.AVAILABLE
        ]
        minimum = min(knee, key=lambda fact: (fact.value, fact.timestamp_us), default=None)
        offset = minimum.timestamp_us - turn.apex_timestamp_us if minimum else None
        turn_facts.extend(
            (
                out(
                    "minimum_mean_knee_angle_timestamp_us",
                    BiomechanicsFeatureFamily.TIMING_PROXY,
                    "us",
                    minimum.timestamp_us if minimum else None,
                    BiomechanicsFactStatus.AVAILABLE if minimum else unavailable,
                ),
                out(
                    "minimum_mean_knee_angle_offset_from_apex_us",
                    BiomechanicsFeatureFamily.TIMING_PROXY,
                    "us",
                    offset,
                    BiomechanicsFactStatus.AVAILABLE if offset is not None else unavailable,
                ),
                out(
                    "minimum_mean_knee_angle_phase_offset",
                    BiomechanicsFeatureFamily.TIMING_PROXY,
                    "ratio",
                    offset / turn.duration_us if offset is not None and turn.duration_us else None,
                    BiomechanicsFactStatus.AVAILABLE
                    if offset is not None and turn.duration_us
                    else BiomechanicsFactStatus.TURN_BOUNDARY_UNAVAILABLE,
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
