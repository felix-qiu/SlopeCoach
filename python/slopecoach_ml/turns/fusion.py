"""Deterministic multi-signal turn evidence fusion for research-only V2 turns."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from slopecoach_ml.pose import Joint
from slopecoach_ml.temporal import StabilizedPoseSample

from .contracts import (
    PeakCandidate,
    TurnSegment,
    TurnSegmentationConfig,
    TurnSignalSample,
    ZeroCrossing,
)
from .peaks import ReferencePeakDetector
from .segmentation import detect_zero_crossings, segment_turns
from .signal import build_turn_signal


@dataclass(frozen=True)
class TurnEvidenceFusionConfig:
    enabled: bool = True
    lateral_weight: float = 0.45
    orientation_weight: float = 0.20
    trajectory_weight: float = 0.20
    lower_body_weight: float = 0.15
    minimum_signal_confidence: float = 0.30
    minimum_fused_evidence_threshold: float = 0.02
    lateral_normalization: float = 0.18
    orientation_rotation_delta_deg_normalization: float = 8.0
    orientation_axis_difference_delta_deg_normalization: float = 6.0
    orientation_rotation_weight: float = 0.70
    orientation_axis_difference_weight: float = 0.30
    trajectory_lateral_velocity_delta_normalization: float = 0.80
    trajectory_curvature_normalization: float = 0.35
    trajectory_lateral_velocity_weight: float = 0.55
    trajectory_curvature_weight: float = 0.45
    lower_body_offset_velocity_normalization: float = 0.55
    lower_body_knee_velocity_delta_normalization: float = 40.0
    lower_body_offset_weight: float = 0.55
    lower_body_knee_weight: float = 0.45
    require_full_transition_context: bool = True
    minimum_transition_crossings: int = 2

    def validate(self) -> None:
        numeric_fields = (
            "lateral_weight",
            "orientation_weight",
            "trajectory_weight",
            "lower_body_weight",
            "minimum_signal_confidence",
            "minimum_fused_evidence_threshold",
            "lateral_normalization",
            "orientation_rotation_delta_deg_normalization",
            "orientation_axis_difference_delta_deg_normalization",
            "orientation_rotation_weight",
            "orientation_axis_difference_weight",
            "trajectory_lateral_velocity_delta_normalization",
            "trajectory_curvature_normalization",
            "trajectory_lateral_velocity_weight",
            "trajectory_curvature_weight",
            "lower_body_offset_velocity_normalization",
            "lower_body_knee_velocity_delta_normalization",
            "lower_body_offset_weight",
            "lower_body_knee_weight",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.minimum_signal_confidence <= 1:
            raise ValueError("minimum_signal_confidence must be in [0, 1]")
        if (
            self.lateral_weight
            + self.orientation_weight
            + self.trajectory_weight
            + self.lower_body_weight
            <= 0
        ):
            raise ValueError("top-level evidence weights must sum to > 0")
        if self.orientation_rotation_weight + self.orientation_axis_difference_weight <= 0:
            raise ValueError("orientation subweights must sum to > 0")
        if self.trajectory_lateral_velocity_weight + self.trajectory_curvature_weight <= 0:
            raise ValueError("trajectory subweights must sum to > 0")
        if self.lower_body_offset_weight + self.lower_body_knee_weight <= 0:
            raise ValueError("lower-body subweights must sum to > 0")
        if (
            isinstance(self.minimum_transition_crossings, bool)
            or not isinstance(self.minimum_transition_crossings, int)
            or self.minimum_transition_crossings < 0
        ):
            raise ValueError("minimum_transition_crossings must be a non-negative integer")


@dataclass(frozen=True)
class TurnEvidenceSample:
    timestamp_us: int
    temporal_segment_id: int | None
    support_confidence: float | None
    lateral_score: float | None
    orientation_score: float | None
    trajectory_score: float | None
    lower_body_score: float | None
    fused_turn_evidence_score: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TurnDetectionResult:
    baseline_signal: tuple[TurnSignalSample, ...]
    signal: tuple[TurnSignalSample, ...]
    raw_peaks: tuple[PeakCandidate, ...]
    peaks: tuple[PeakCandidate, ...]
    rejected_peaks: tuple[PeakCandidate, ...]
    crossings: tuple[ZeroCrossing, ...]
    segments: tuple[TurnSegment, ...]
    evidence_samples: tuple[TurnEvidenceSample, ...]
    fusion_summary: dict[str, object]
    fusion_config: TurnEvidenceFusionConfig | None


def detect_turns_with_reference_pipeline(
    samples: list[StabilizedPoseSample],
    turn_config: TurnSegmentationConfig,
) -> TurnDetectionResult:
    turn_config.validate()
    baseline_signal = tuple(
        build_turn_signal(samples, minimum_confidence=turn_config.minimum_signal_confidence)
    )
    raw_peaks = tuple(ReferencePeakDetector().detect(list(baseline_signal), turn_config))
    crossings = tuple(
        detect_zero_crossings(
            list(baseline_signal),
            turn_config.zero_crossing_tolerance,
            minimum_signal_confidence=turn_config.minimum_signal_confidence,
        )
    )
    segments = tuple(
        segment_turns(list(baseline_signal), list(raw_peaks), list(crossings), turn_config)
    )
    evidence_samples = tuple(
        TurnEvidenceSample(
            timestamp_us=item.timestamp_us,
            temporal_segment_id=item.temporal_segment_id,
            support_confidence=item.support_confidence,
            lateral_score=item.value,
            orientation_score=None,
            trajectory_score=None,
            lower_body_score=None,
            fused_turn_evidence_score=item.value,
        )
        for item in baseline_signal
    )
    return TurnDetectionResult(
        baseline_signal=baseline_signal,
        signal=baseline_signal,
        raw_peaks=raw_peaks,
        peaks=raw_peaks,
        rejected_peaks=(),
        crossings=crossings,
        segments=segments,
        evidence_samples=evidence_samples,
        fusion_summary=_fusion_summary(
            evidence_samples=evidence_samples,
            config=None,
            candidate_count=len(raw_peaks),
            rejected_count=0,
            enabled=False,
        ),
        fusion_config=None,
    )


def detect_turns_with_evidence_fusion(
    samples: list[StabilizedPoseSample],
    turn_config: TurnSegmentationConfig,
    fusion_config: TurnEvidenceFusionConfig | None = None,
) -> TurnDetectionResult:
    turn_config.validate()
    settings = fusion_config or TurnEvidenceFusionConfig()
    settings.validate()
    if not settings.enabled:
        return detect_turns_with_reference_pipeline(samples, turn_config)

    baseline_signal = tuple(
        build_turn_signal(samples, minimum_confidence=turn_config.minimum_signal_confidence)
    )
    observations = tuple(_sample_observation(sample, settings) for sample in samples)
    evidence_samples = tuple(
        _evidence_sample(
            index,
            baseline_signal=baseline_signal,
            observations=observations,
            config=settings,
        )
        for index in range(len(samples))
    )
    fused_signal = tuple(
        TurnSignalSample(
            timestamp_us=item.timestamp_us,
            temporal_segment_id=item.temporal_segment_id,
            value=item.fused_turn_evidence_score,
            support_confidence=item.support_confidence,
            provenance=_signal_provenance(baseline_signal[index], item),
        )
        for index, item in enumerate(evidence_samples)
    )
    raw_peaks = tuple(ReferencePeakDetector().detect(list(fused_signal), turn_config))
    crossings = tuple(
        detect_zero_crossings(
            list(fused_signal),
            turn_config.zero_crossing_tolerance,
            minimum_signal_confidence=turn_config.minimum_signal_confidence,
        )
    )
    peaks, rejected_peaks = _filter_peaks_with_transition_context(raw_peaks, crossings, settings)
    segments = tuple(segment_turns(list(fused_signal), list(peaks), list(crossings), turn_config))
    return TurnDetectionResult(
        baseline_signal=baseline_signal,
        signal=fused_signal,
        raw_peaks=raw_peaks,
        peaks=peaks,
        rejected_peaks=rejected_peaks,
        crossings=crossings,
        segments=segments,
        evidence_samples=evidence_samples,
        fusion_summary=_fusion_summary(
            evidence_samples=evidence_samples,
            config=settings,
            candidate_count=len(raw_peaks),
            rejected_count=len(rejected_peaks),
            enabled=True,
        ),
        fusion_config=settings,
    )


@dataclass(frozen=True)
class _Observation:
    timestamp_us: int
    temporal_segment_id: int | None
    shoulder_axis_deg: float | None
    hip_axis_deg: float | None
    torso_inclination_deg: float | None
    shoulder_hip_axis_difference_deg: float | None
    body_center: tuple[float, float] | None
    body_scale: float | None
    hip_to_ankle_offset: float | None
    left_knee_angle_deg: float | None
    right_knee_angle_deg: float | None


def _sample_observation(
    sample: StabilizedPoseSample,
    config: TurnEvidenceFusionConfig,
) -> _Observation:
    shoulders = _joint_pair(sample, Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER, config)
    hips = _joint_pair(sample, Joint.LEFT_HIP, Joint.RIGHT_HIP, config)
    ankles = _joint_pair(sample, Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE, config)
    left_knee = _joint_point(sample, Joint.LEFT_KNEE, config)
    right_knee = _joint_point(sample, Joint.RIGHT_KNEE, config)

    body_center = None
    body_scale = None
    torso_inclination_deg = None
    shoulder_axis_deg = None
    hip_axis_deg = None
    shoulder_hip_axis_difference_deg = None
    hip_to_ankle_offset = None
    left_knee_angle_deg = None
    right_knee_angle_deg = None

    if shoulders is not None:
        shoulder_axis_deg = _axis_angle_deg(*shoulders)
    if hips is not None:
        hip_axis_deg = _axis_angle_deg(*hips)
    if shoulders is not None and hips is not None:
        shoulder_center = _midpoint_2d(*shoulders)
        hip_center = _midpoint_2d(*hips)
        body_center = (
            (shoulder_center[0] + hip_center[0]) / 2,
            (shoulder_center[1] + hip_center[1]) / 2,
        )
        torso_inclination_deg = _signed_screen_angle_from_vertical(shoulder_center, hip_center)
        shoulder_hip_axis_difference_deg = _undirected_axis_difference_deg(*shoulders, *hips)
    if shoulders is not None and ankles is not None:
        shoulder_center = _midpoint_2d(*shoulders)
        ankle_center = _midpoint_2d(*ankles)
        raw_scale = math.hypot(
            ankle_center[0] - shoulder_center[0], ankle_center[1] - shoulder_center[1]
        )
        if math.isfinite(raw_scale) and raw_scale > 1e-9:
            body_scale = raw_scale
    if body_scale is not None and hips is not None and ankles is not None:
        hip_to_ankle_offset = _normalized_screen_x_offset(
            _midpoint_2d(*hips),
            _midpoint_2d(*ankles),
            body_scale,
        )
    if hips is not None and left_knee is not None and ankles is not None:
        left_knee_angle_deg = _angle_three_points_2d(hips[0], left_knee, ankles[0])
    if hips is not None and right_knee is not None and ankles is not None:
        right_knee_angle_deg = _angle_three_points_2d(hips[1], right_knee, ankles[1])

    return _Observation(
        timestamp_us=sample.timestamp_us,
        temporal_segment_id=sample.temporal_segment_id,
        shoulder_axis_deg=shoulder_axis_deg,
        hip_axis_deg=hip_axis_deg,
        torso_inclination_deg=torso_inclination_deg,
        shoulder_hip_axis_difference_deg=shoulder_hip_axis_difference_deg,
        body_center=body_center,
        body_scale=body_scale,
        hip_to_ankle_offset=hip_to_ankle_offset,
        left_knee_angle_deg=left_knee_angle_deg,
        right_knee_angle_deg=right_knee_angle_deg,
    )


def _evidence_sample(
    index: int,
    *,
    baseline_signal: tuple[TurnSignalSample, ...],
    observations: tuple[_Observation, ...],
    config: TurnEvidenceFusionConfig,
) -> TurnEvidenceSample:
    base = baseline_signal[index]
    if (
        base.temporal_segment_id is None
        or base.value is None
        or base.support_confidence is None
        or base.support_confidence < config.minimum_signal_confidence
    ):
        return TurnEvidenceSample(
            timestamp_us=base.timestamp_us,
            temporal_segment_id=base.temporal_segment_id,
            support_confidence=base.support_confidence,
            lateral_score=base.value,
            orientation_score=None,
            trajectory_score=None,
            lower_body_score=None,
            fused_turn_evidence_score=None,
        )

    lateral_score = _normalize_signed(base.value, config.lateral_normalization)
    orientation_score = _orientation_score(index, baseline_signal, observations, config)
    trajectory_score = _trajectory_score(index, observations, config)
    lower_body_score = _lower_body_score(index, observations, config)
    fused_score = _blend(
        (
            (lateral_score, config.lateral_weight),
            (orientation_score, config.orientation_weight),
            (trajectory_score, config.trajectory_weight),
            (lower_body_score, config.lower_body_weight),
        )
    )
    if abs(fused_score) < config.minimum_fused_evidence_threshold:
        fused_score = 0.0

    return TurnEvidenceSample(
        timestamp_us=base.timestamp_us,
        temporal_segment_id=base.temporal_segment_id,
        support_confidence=base.support_confidence,
        lateral_score=lateral_score,
        orientation_score=orientation_score,
        trajectory_score=trajectory_score,
        lower_body_score=lower_body_score,
        fused_turn_evidence_score=fused_score,
    )


def _orientation_score(
    index: int,
    baseline_signal: tuple[TurnSignalSample, ...],
    observations: tuple[_Observation, ...],
    config: TurnEvidenceFusionConfig,
) -> float:
    previous = _previous_segment_index(
        index,
        observations,
        lambda item: any(
            value is not None
            for value in (item.shoulder_axis_deg, item.hip_axis_deg, item.torso_inclination_deg)
        ),
    )
    if previous is None:
        return 0.0
    current_obs = observations[index]
    previous_obs = observations[previous]
    rotation_deltas = [
        _angle_delta_deg(current_obs.shoulder_axis_deg, previous_obs.shoulder_axis_deg),
        _angle_delta_deg(current_obs.hip_axis_deg, previous_obs.hip_axis_deg),
        _angle_delta_deg(current_obs.torso_inclination_deg, previous_obs.torso_inclination_deg),
    ]
    finite_rotations = [value for value in rotation_deltas if value is not None]
    signed_rotation = sum(finite_rotations) / len(finite_rotations) if finite_rotations else 0.0
    rotation_score = _normalize_signed(
        signed_rotation,
        config.orientation_rotation_delta_deg_normalization,
    )
    axis_difference_delta = _difference_delta(
        current_obs.shoulder_hip_axis_difference_deg,
        previous_obs.shoulder_hip_axis_difference_deg,
    )
    axis_difference_score = 0.0
    if axis_difference_delta is not None:
        sign = _signed_unit(signed_rotation)
        if sign == 0.0:
            sign = _signed_unit(baseline_signal[index].value or 0.0)
        axis_difference_score = sign * min(
            1.0,
            abs(axis_difference_delta)
            / max(config.orientation_axis_difference_delta_deg_normalization, 1e-12),
        )
    return _blend(
        (
            (rotation_score, config.orientation_rotation_weight),
            (axis_difference_score, config.orientation_axis_difference_weight),
        )
    )


def _trajectory_score(
    index: int,
    observations: tuple[_Observation, ...],
    config: TurnEvidenceFusionConfig,
) -> float:
    previous = _previous_segment_index(
        index, observations, lambda item: item.body_center is not None
    )
    if previous is None:
        return 0.0
    earlier = _previous_segment_index(
        previous, observations, lambda item: item.body_center is not None
    )
    if earlier is None:
        return 0.0

    current_velocity = _normalized_velocity(previous, index, observations)
    previous_velocity = _normalized_velocity(earlier, previous, observations)
    if current_velocity is None or previous_velocity is None:
        return 0.0

    lateral_velocity_delta = current_velocity[0] - previous_velocity[0]
    lateral_score = _normalize_signed(
        lateral_velocity_delta,
        config.trajectory_lateral_velocity_delta_normalization,
    )
    norm = math.hypot(*previous_velocity) * math.hypot(*current_velocity)
    curvature = 0.0
    if norm > 1e-12:
        curvature = (
            previous_velocity[0] * current_velocity[1] - previous_velocity[1] * current_velocity[0]
        ) / norm
    curvature_score = _normalize_signed(curvature, config.trajectory_curvature_normalization)
    return _blend(
        (
            (lateral_score, config.trajectory_lateral_velocity_weight),
            (curvature_score, config.trajectory_curvature_weight),
        )
    )


def _lower_body_score(
    index: int,
    observations: tuple[_Observation, ...],
    config: TurnEvidenceFusionConfig,
) -> float:
    previous = _previous_segment_index(index, observations)
    if previous is None:
        return 0.0
    delta_us = _index_delta_us(previous, index, observations)
    if delta_us is None:
        return 0.0
    seconds = delta_us / 1_000_000
    if seconds <= 0:
        return 0.0

    current_obs = observations[index]
    previous_obs = observations[previous]
    offset_score = 0.0
    if current_obs.hip_to_ankle_offset is not None and previous_obs.hip_to_ankle_offset is not None:
        offset_score = _normalize_signed(
            (current_obs.hip_to_ankle_offset - previous_obs.hip_to_ankle_offset) / seconds,
            config.lower_body_offset_velocity_normalization,
        )

    knee_score = 0.0
    if (
        current_obs.left_knee_angle_deg is not None
        and current_obs.right_knee_angle_deg is not None
        and previous_obs.left_knee_angle_deg is not None
        and previous_obs.right_knee_angle_deg is not None
    ):
        left_velocity = (
            current_obs.left_knee_angle_deg - previous_obs.left_knee_angle_deg
        ) / seconds
        right_velocity = (
            current_obs.right_knee_angle_deg - previous_obs.right_knee_angle_deg
        ) / seconds
        knee_score = _normalize_signed(
            right_velocity - left_velocity,
            config.lower_body_knee_velocity_delta_normalization,
        )

    return _blend(
        (
            (offset_score, config.lower_body_offset_weight),
            (knee_score, config.lower_body_knee_weight),
        )
    )


def _filter_peaks_with_transition_context(
    peaks: tuple[PeakCandidate, ...],
    crossings: tuple[ZeroCrossing, ...],
    config: TurnEvidenceFusionConfig,
) -> tuple[tuple[PeakCandidate, ...], tuple[PeakCandidate, ...]]:
    if not config.require_full_transition_context:
        return peaks, ()

    by_run: dict[int, list[ZeroCrossing]] = {}
    for crossing in crossings:
        by_run.setdefault(crossing.signal_run_id, []).append(crossing)

    accepted: list[PeakCandidate] = []
    rejected: list[PeakCandidate] = []
    for peak in peaks:
        run_crossings = by_run.get(peak.signal_run_id, [])
        before = [item for item in run_crossings if item.timestamp_us < peak.timestamp_us]
        after = [item for item in run_crossings if item.timestamp_us > peak.timestamp_us]
        if (
            len(before) + len(after) < config.minimum_transition_crossings
            or not before
            or not after
        ):
            rejected.append(peak)
            continue
        accepted.append(peak)
    return tuple(accepted), tuple(rejected)


def _joint_point(
    sample: StabilizedPoseSample,
    joint: Joint,
    config: TurnEvidenceFusionConfig,
) -> tuple[float, float] | None:
    point = sample.joint(joint)
    if point is None:
        return None
    if (
        point.support_confidence is None
        or point.support_confidence < config.minimum_signal_confidence
        or point.stabilized_x_px is None
        or point.stabilized_y_px is None
    ):
        return None
    if not math.isfinite(point.stabilized_x_px) or not math.isfinite(point.stabilized_y_px):
        return None
    return (point.stabilized_x_px, point.stabilized_y_px)


def _joint_pair(
    sample: StabilizedPoseSample,
    left: Joint,
    right: Joint,
    config: TurnEvidenceFusionConfig,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    left_point = _joint_point(sample, left, config)
    right_point = _joint_point(sample, right, config)
    if left_point is None or right_point is None:
        return None
    return left_point, right_point


def _axis_angle_deg(start: tuple[float, float], end: tuple[float, float]) -> float | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.hypot(dx, dy) <= 1e-12:
        return None
    return math.degrees(math.atan2(dy, dx))


def _midpoint_2d(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)


def _angle_three_points_2d(
    first: tuple[float, float],
    vertex: tuple[float, float],
    third: tuple[float, float],
) -> float | None:
    a = (first[0] - vertex[0], first[1] - vertex[1])
    b = (third[0] - vertex[0], third[1] - vertex[1])
    denominator = math.hypot(*a) * math.hypot(*b)
    if denominator <= 1e-12:
        return None
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _signed_screen_angle_from_vertical(
    start: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.hypot(dx, dy) <= 1e-12:
        return None
    return math.degrees(math.atan2(dx, dy))


def _undirected_axis_difference_deg(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float | None:
    first_dx = first_end[0] - first_start[0]
    first_dy = first_end[1] - first_start[1]
    second_dx = second_end[0] - second_start[0]
    second_dy = second_end[1] - second_start[1]
    if math.hypot(first_dx, first_dy) <= 1e-12 or math.hypot(second_dx, second_dy) <= 1e-12:
        return None
    first = math.degrees(math.atan2(first_dy, first_dx))
    second = math.degrees(math.atan2(second_dy, second_dx))
    difference = abs(first - second) % 180
    return min(difference, 180 - difference)


def _normalized_screen_x_offset(
    first: tuple[float, float],
    second: tuple[float, float],
    scale: float,
) -> float | None:
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    return (first[0] - second[0]) / scale


def _angle_delta_deg(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    delta = current - previous
    while delta <= -180:
        delta += 360
    while delta > 180:
        delta -= 360
    return delta


def _difference_delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _normalize_signed(value: float, scale: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(scale) or scale <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, value / scale))


def _blend(items: Iterable[tuple[float, float]]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in items:
        if not math.isfinite(weight) or weight <= 0:
            continue
        total_weight += weight
        total_value += value * weight
    if total_weight <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, total_value / total_weight))


def _previous_segment_index(index: int, items: tuple, predicate=None) -> int | None:
    predicate = predicate or (lambda item: True)
    current_segment_id = items[index].temporal_segment_id
    for previous in range(index - 1, -1, -1):
        candidate = items[previous]
        if candidate.temporal_segment_id != current_segment_id:
            break
        if predicate(candidate):
            return previous
    return None


def _normalized_velocity(
    previous_index: int,
    current_index: int,
    observations: tuple[_Observation, ...],
) -> tuple[float, float] | None:
    previous = observations[previous_index]
    current = observations[current_index]
    delta_us = _index_delta_us(previous_index, current_index, observations)
    if (
        previous.body_center is None
        or current.body_center is None
        or previous.body_scale is None
        or current.body_scale is None
        or delta_us is None
        or delta_us <= 0
    ):
        return None
    scale = (previous.body_scale + current.body_scale) / 2
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    seconds = delta_us / 1_000_000
    return (
        (current.body_center[0] - previous.body_center[0]) / (seconds * scale),
        (current.body_center[1] - previous.body_center[1]) / (seconds * scale),
    )


def _index_delta_us(
    previous_index: int,
    current_index: int,
    observations: tuple[_Observation, ...],
) -> int | None:
    previous = observations[previous_index]
    current = observations[current_index]
    if previous.temporal_segment_id != current.temporal_segment_id:
        return None
    delta_us = current.timestamp_us - previous.timestamp_us
    return delta_us if delta_us > 0 else None


def _signal_provenance(base: TurnSignalSample, evidence: TurnEvidenceSample) -> str:
    if evidence.fused_turn_evidence_score is None:
        return "MISSING"
    return f"{base.provenance}+TURN_EVIDENCE_FUSION_V2"


def _fusion_summary(
    *,
    evidence_samples: tuple[TurnEvidenceSample, ...],
    config: TurnEvidenceFusionConfig | None,
    candidate_count: int,
    rejected_count: int,
    enabled: bool,
) -> dict[str, object]:
    component_weight_abs_sum = {
        "lateral_score": _sum_abs(item.lateral_score for item in evidence_samples),
        "orientation_score": _sum_abs(item.orientation_score for item in evidence_samples),
        "trajectory_score": _sum_abs(item.trajectory_score for item in evidence_samples),
        "lower_body_score": _sum_abs(item.lower_body_score for item in evidence_samples),
    }
    if config is not None:
        weighted = {
            "lateral_score": component_weight_abs_sum["lateral_score"] * config.lateral_weight,
            "orientation_score": component_weight_abs_sum["orientation_score"]
            * config.orientation_weight,
            "trajectory_score": component_weight_abs_sum["trajectory_score"]
            * config.trajectory_weight,
            "lower_body_score": component_weight_abs_sum["lower_body_score"]
            * config.lower_body_weight,
        }
    else:
        weighted = {"lateral_score": component_weight_abs_sum["lateral_score"]}
    dominant_source = max(weighted, key=lambda name: (weighted[name], name))
    fused_values = [
        sample.fused_turn_evidence_score
        for sample in evidence_samples
        if sample.fused_turn_evidence_score is not None
    ]
    return {
        "enabled": enabled,
        "candidate_count": candidate_count,
        "rejected_count": rejected_count,
        "component_weight_abs_sum": weighted,
        "dominant_evidence_source": dominant_source,
        "max_fused_score": max(fused_values) if fused_values else None,
        "min_fused_score": min(fused_values) if fused_values else None,
    }


def _signed_unit(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _sum_abs(values: Iterable[float | None]) -> float:
    return sum(abs(value) for value in values if value is not None and math.isfinite(value))
