"""A3 target identity research benchmark; independent of the A2.2 raw baseline."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from slopecoach_ml.biomechanics import knee_angle_2d
from slopecoach_ml.detection import Detection, DetectorProvider
from slopecoach_ml.identity import (
    AutoInitialTargetSelector,
    CandidateFilterConfig,
    GroundTruthEvaluationConfig,
    HSVHistogramAppearanceEncoder,
    InitialTargetSelectorConfig,
    ManualTargetSeed,
    PoseSchedulingConfig,
    TargetIdentityConfig,
    TargetIdentityGroundTruth,
    TargetIdentityManager,
    TargetIdentityState,
    evaluate_candidates,
    evaluate_target_identity_ground_truth,
    manual_seed_frame_is_eligible,
    manual_seed_window_has_passed,
    schedule_pose_track_ids,
    select_manual_target_seed_match,
    target_biomechanics_allowed,
)
from slopecoach_ml.pose import PoseFrame
from slopecoach_ml.pose.mmpose_provider import MMPoseRTMWPoseProvider
from slopecoach_ml.reference import ReferenceAnalysisConfig
from slopecoach_ml.tracking import ReferenceMotionIoUTracker, TrackingConfig
from slopecoach_ml.video import SampledFrame, inspect_video

MINIMUM_P95_SAMPLES = 20


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _p95(values: list[float]) -> float | None:
    if len(values) < MINIMUM_P95_SAMPLES:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _manual_seed_frame_annotations(
    frames: Iterable[SampledFrame], seed: ManualTargetSeed | None, sample_fps: float
) -> Iterable[tuple[SampledFrame, bool]]:
    """Mark the closest eligible sample, preferring the earlier sample on a tie."""

    if seed is None:
        for sampled in frames:
            yield sampled, False
        return
    iterator = iter(frames)
    try:
        current = next(iterator)
    except StopIteration:
        return
    requested = seed.requested_timestamp_us
    for following in iterator:
        current_eligible = manual_seed_frame_is_eligible(seed, current.timestamp_us, sample_fps)
        following_eligible = manual_seed_frame_is_eligible(seed, following.timestamp_us, sample_fps)
        current_is_closest = current_eligible and (
            not following_eligible
            or abs(current.timestamp_us - requested) <= abs(following.timestamp_us - requested)
        )
        yield current, current_is_closest
        current = following
    yield current, manual_seed_frame_is_eligible(seed, current.timestamp_us, sample_fps)


def _config_dict(candidate, tracking, selector, identity, scheduling, ground_truth):
    from dataclasses import asdict

    return {
        "profile": "RESEARCH_DEFAULTS_A3_1",
        "candidate_filter": asdict(candidate),
        "tracking": tracking.to_dict(),
        "initial_target_selector": asdict(selector),
        "target_identity": asdict(identity),
        "pose_scheduling": asdict(scheduling),
        "ground_truth_evaluation": asdict(ground_truth),
    }


def benchmark_target_identity_frames(
    *,
    input_path: str | Path,
    frames: Iterable[SampledFrame],
    detector: DetectorProvider,
    pose_provider: MMPoseRTMWPoseProvider,
    detector_model: dict[str, Any],
    pose_model: dict[str, Any],
    device: str = "cpu",
    sample_fps: float = 2.0,
    model_load: dict[str, float] | None = None,
    warmup_frames: int = 0,
    candidate_config: CandidateFilterConfig | None = None,
    tracking_config: TrackingConfig | None = None,
    selector_config: InitialTargetSelectorConfig | None = None,
    identity_config: TargetIdentityConfig | None = None,
    scheduling_config: PoseSchedulingConfig | None = None,
    frame_observer: Callable[[SampledFrame, dict[str, Any], PoseFrame | None], None] | None = None,
    appearance_encoder: Any | None = None,
    target_ground_truth: TargetIdentityGroundTruth | None = None,
    ground_truth_config: GroundTruthEvaluationConfig | None = None,
    ground_truth_status: str = "NOT_AVAILABLE",
    manual_target_seed: ManualTargetSeed | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    candidate_settings = candidate_config or CandidateFilterConfig()
    tracking_settings = tracking_config or TrackingConfig()
    selector_settings = selector_config or InitialTargetSelectorConfig()
    identity_settings = identity_config or TargetIdentityConfig()
    scheduling_settings = scheduling_config or PoseSchedulingConfig()
    ground_truth_settings = ground_truth_config or GroundTruthEvaluationConfig()
    if manual_target_seed is not None:
        manual_target_seed.validate()
    for settings in (
        candidate_settings,
        tracking_settings,
        selector_settings,
        identity_settings,
        scheduling_settings,
        ground_truth_settings,
    ):
        settings.validate()
    tracker = ReferenceMotionIoUTracker(tracking_settings)
    selector = AutoInitialTargetSelector(selector_settings)
    manager = TargetIdentityManager(identity_settings)
    appearance = appearance_encoder or HSVHistogramAppearanceEncoder()
    analysis_config = ReferenceAnalysisConfig()
    metadata = inspect_video(input_path)
    started = clock()
    totals = Counter()
    rejection_reasons: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    observations = []
    active_track_counts = []
    detector_latencies: list[float] = []
    tracking_identity_latencies: list[float] = []
    pose_latencies: list[float] = []
    pipeline_latencies: list[float] = []
    locked_confidences: list[float] = []
    pose_candidates_per_frame: list[int] = []
    first_timestamp = first_lock_timestamp = None
    stage_totals = Counter()
    relock_frames: list[int] = []
    pose_quality_by_track: dict[int, float] = {}
    confirmed_track_ids: set[int] = set()
    manual_seed_application: dict[str, Any] | None = None
    for sampled, apply_manual_seed in _manual_seed_frame_annotations(
        frames, manual_target_seed, sample_fps
    ):
        if (
            manual_target_seed is not None
            and manual_seed_application is None
            and manual_seed_window_has_passed(manual_target_seed, sampled.timestamp_us, sample_fps)
        ):
            raise ValueError("MANUAL_TARGET_SEED_FRAME_NOT_FOUND")
        identity_matches = ()
        if first_timestamp is None:
            first_timestamp = sampled.timestamp_us
        pipeline_started = clock()
        stage = clock()
        raw = detector.detect(sampled.image, sampled.geometry)
        detector_latency = clock() - stage
        detector_latencies.append(detector_latency)
        stage_totals["detector"] += detector_latency
        totals["raw_detections"] += len(raw)
        stage = clock()
        candidates = evaluate_candidates(raw, sampled.geometry, candidate_settings)
        candidate_latency = clock() - stage
        stage_totals["candidate"] += candidate_latency
        viable = [item for item in candidates if item.hard_rejection_reason is None]
        totals["viable_candidates"] += len(viable)
        totals["rejected_candidates"] += len(candidates) - len(viable)
        rejection_reasons.update(
            item.hard_rejection_reason for item in candidates if item.hard_rejection_reason
        )
        viable_detections = tuple(
            Detection(item.detection_id, item.bbox, item.detection_confidence) for item in viable
        )
        candidates_by_detection = {item.detection_id: item for item in viable}
        stage = clock()
        tracking_frame = tracker.update(
            viable_detections,
            sampled.timestamp_us,
            sampled.frame_index,
            sampled.geometry,
            preferred_track_id=manager.identity.active_track_id,
        )
        tracking_latency = clock() - stage
        stage_totals["tracking"] += tracking_latency
        current_tracks = tuple(
            track for track in tracking_frame.tracks if track.missed_duration_us == 0
        )
        confirmed_track_ids.update(
            track.track_id for track in tracking_frame.tracks if track.state.value == "CONFIRMED"
        )
        active_track_counts.append(len(tracking_frame.tracks))
        stage = clock()
        descriptors = {}
        for track in current_tracks:
            descriptor = appearance.encode(sampled.image, track.bbox)
            if descriptor is not None:
                descriptors[track.track_id] = descriptor
        appearance_latency = clock() - stage
        stage_totals["appearance"] += appearance_latency
        stage = clock()
        selection = None
        if manual_target_seed is not None and manual_seed_application is None:
            if apply_manual_seed:
                manual_match = select_manual_target_seed_match(
                    manual_target_seed,
                    sampled.geometry,
                    current_tracks,
                    candidates_by_detection,
                )
                selected = manual_match.track
                initialization_score = manual_match.candidate.quality_score
                manager.initialize(
                    selected,
                    initialization_score,
                    sampled.timestamp_us,
                    descriptors.get(selected.track_id),
                )
                first_lock_timestamp = sampled.timestamp_us
                manual_seed_application = {
                    "selection_source": "MANUAL_SEED",
                    "ground_truth_status": "NOT_GROUND_TRUTH",
                    "requested_time_seconds": manual_target_seed.time_seconds,
                    "requested_point_source_pixel_2d": {
                        "x_px": manual_target_seed.x_px,
                        "y_px": manual_target_seed.y_px,
                    },
                    "applied_timestamp_us": sampled.timestamp_us,
                    "applied_frame_index": sampled.frame_index,
                    "selected_track_id": selected.track_id,
                    "selected_detection_id": selected.detection_id,
                    "selected_bbox": selected.bbox.to_dict(),
                    "identity_evidence_confidence": initialization_score,
                }
        elif (
            manager.identity.state
            in {TargetIdentityState.UNINITIALIZED, TargetIdentityState.AMBIGUOUS}
            and manager.identity.initial_selection_score is None
        ):
            selection = selector.observe(
                current_tracks,
                candidates_by_detection,
                sampled.geometry,
                sampled.timestamp_us,
                pose_quality=pose_quality_by_track,
            )
            if (
                selection.state is TargetIdentityState.LOCKED
                and selection.selected_track_id is not None
            ):
                selected = next(
                    track
                    for track in current_tracks
                    if track.track_id == selection.selected_track_id
                )
                manager.initialize(
                    selected,
                    selection.score or 0.0,
                    sampled.timestamp_us,
                    descriptors.get(selected.track_id),
                )
                if first_lock_timestamp is None:
                    first_lock_timestamp = sampled.timestamp_us
            else:
                manager.identity.state = selection.state
        elif manager.identity.initial_selection_score is not None:
            before_relocks = manager.relock_count
            identity_matches = manager.update(
                current_tracks,
                candidates_by_detection,
                sampled.timestamp_us,
                descriptors=descriptors,
            )
            if manager.relock_count > before_relocks:
                relock_frames.append(sampled.frame_index)
        else:
            pass
        identity_latency = clock() - stage
        stage_totals["identity"] += identity_latency
        ranked_track_ids = [
            track.track_id
            for track in sorted(
                current_tracks,
                key=lambda item: (
                    -candidates_by_detection[item.detection_id].quality_score,
                    item.track_id,
                ),
            )
            if track.detection_id in candidates_by_detection
        ]
        scheduled_ids = schedule_pose_track_ids(
            manager.identity.state,
            manager.identity.active_track_id,
            ranked_track_ids,
            scheduling_settings,
        )
        tracks_by_id = {track.track_id: track for track in current_tracks}
        selected_track = tracks_by_id.get(manager.identity.active_track_id)
        scheduled_tracks = [tracks_by_id[item] for item in scheduled_ids if item in tracks_by_id]
        scheduled_detections = tuple(
            Detection(track.detection_id, track.bbox, track.confidence)
            for track in scheduled_tracks
            if track.detection_id is not None
        )
        pose_candidates_per_frame.append(len(scheduled_detections))
        totals["pose_person_inferences"] += len(scheduled_detections)
        pose_frame = None
        stage = clock()
        if scheduled_detections:
            pose_frame = pose_provider.estimate_detections(
                sampled.image,
                scheduled_detections,
                sampled.geometry,
                timestamp_us=sampled.timestamp_us,
                frame_index=sampled.frame_index,
            )
            pose_latencies.append(pose_provider.last_backend_seconds)
            stage_totals["pose"] += pose_provider.last_backend_seconds
            stage_totals["adapter"] += pose_provider.last_adapter_seconds
            pose_quality_by_track = {
                track.track_id: mean(point.confidence for point in person.keypoints.values())
                for track in scheduled_tracks
                for person in pose_frame.persons
                if person.detection_id == track.detection_id and person.keypoints
            }
        else:
            pose_quality_by_track = {}
        pose_stage_elapsed = clock() - stage
        if not scheduled_detections:
            pose_latencies.append(0.0)
        target_person = None
        if pose_frame and manager.identity.active_track_id is not None:
            active_track = tracks_by_id.get(manager.identity.active_track_id)
            if active_track:
                target_person = next(
                    (
                        person
                        for person in pose_frame.persons
                        if person.detection_id == active_track.detection_id
                    ),
                    None,
                )
        if target_person is not None:
            totals["target_pose_frames"] += 1
        stage = clock()
        angle = None
        limitations = ["IMAGE_2D_ONLY_NOT_PHYSICAL_3D"]
        if target_person is not None and target_biomechanics_allowed(
            manager.identity.state,
            manager.identity.confidence,
            identity_settings.safe_biomechanics_confidence,
        ):
            angle = knee_angle_2d(
                target_person,
                sampled.geometry,
                side="left",
                minimum_confidence=analysis_config.min_joint_confidence,
                square_pixel_tolerance=analysis_config.square_pixel_tolerance,
            )
        else:
            limitations.append("TARGET_IDENTITY_UNCERTAIN")
        if angle is not None:
            totals["target_knee_angles"] += 1
        stage_totals["biomechanics"] += clock() - stage
        state = manager.identity.state
        state_counts[state.value] += 1
        if state is TargetIdentityState.LOCKED:
            locked_confidences.append(manager.identity.confidence)
        warning = {
            TargetIdentityState.UNINITIALIZED: "TARGET_INITIALIZATION_INSUFFICIENT",
            TargetIdentityState.SUSPECT: "TARGET_IDENTITY_SUSPECT",
            TargetIdentityState.LOST: "TARGET_IDENTITY_LOST",
            TargetIdentityState.RECOVERING: "TARGET_IDENTITY_RECOVERING",
            TargetIdentityState.AMBIGUOUS: "TARGET_SELECTION_AMBIGUOUS",
        }.get(state)
        observation = {
            "timestamp_us": sampled.timestamp_us,
            "frame_index": sampled.frame_index,
            "raw_detection_count": len(raw),
            "candidate_count": len(viable),
            "track_count": len(tracking_frame.tracks),
            "target_id": manager.identity.target_id,
            "active_track_id": manager.identity.active_track_id,
            "identity_state": state.value,
            "identity_confidence": manager.identity.confidence,
            "latest_identity_match_score": manager.identity.latest_identity_match_score,
            "last_observed_timestamp_us": manager.identity.last_observed_us,
            "last_observed_age_us": (
                sampled.timestamp_us - manager.identity.last_observed_us
                if manager.identity.last_observed_us is not None
                else None
            ),
            "selected_bbox": (
                selected_track.bbox.to_dict()
                if selected_track is not None
                and manager.identity.state is TargetIdentityState.LOCKED
                else None
            ),
            "identity_match_evidence": [
                {
                    "track_id": match.track_id,
                    "fused_score": match.fused_score,
                    "evidence": match.evidence.__dict__,
                }
                for match in sorted(identity_matches, key=lambda item: -item.fused_score)[:3]
            ],
            "initial_selection_score": selection.score if selection else None,
            "initial_selection_margin": selection.margin if selection else None,
            "candidate_scores": {str(item.detection_id): item.quality_score for item in candidates},
            "pose_inference_candidate_count": len(scheduled_detections),
            "target_pose_available": target_person is not None,
            "left_knee_angle_2d_degrees": angle,
            "warnings": [warning] if warning else [],
            "limitations": limitations,
            "timing": {
                "detector_seconds": detector_latency,
                "candidate_filter_seconds": candidate_latency,
                "tracking_identity_seconds": tracking_latency + identity_latency,
                "pose_seconds": pose_stage_elapsed,
            },
        }
        observations.append(observation)
        pipeline_latencies.append(clock() - pipeline_started)
        tracking_identity_latencies.append(tracking_latency + identity_latency)
        if frame_observer:
            frame_observer(
                sampled,
                {
                    **observation,
                    "tracks": [track.to_dict() for track in tracking_frame.tracks],
                },
                pose_frame,
            )
    if manual_target_seed is not None and manual_seed_application is None:
        raise ValueError("MANUAL_TARGET_SEED_FRAME_NOT_FOUND")
    finished = clock()
    sampled_count = len(observations)
    total_raw = totals["raw_detections"]
    baseline = _load_a2_baseline(input_path)
    report = {
        "benchmark_contract_version": "ski-bench-target-identity-v2",
        "input_kind": "REAL_VIDEO",
        "runtime": {
            "device": device,
            "model_load": model_load or {"detector_seconds": None, "pose_seconds": None},
            "warmup_frames": warmup_frames,
            "warmup_included_in_per_frame_timing": False,
        },
        "models": {"detector": detector_model, "pose": pose_model},
        "config": _config_dict(
            candidate_settings,
            tracking_settings,
            selector_settings,
            identity_settings,
            scheduling_settings,
            ground_truth_settings,
        ),
        "video": metadata.to_dict(),
        "sampling": {
            "sample_fps": sample_fps,
            "sampled_frame_count": sampled_count,
            "timestamps_us": [item["timestamp_us"] for item in observations],
            "source_frame_indices": [item["frame_index"] for item in observations],
        },
        "detections": {
            "raw_detection_person_count": total_raw,
            "mean_raw_detections_per_frame": _ratio(total_raw, sampled_count),
        },
        "candidates": {
            "viable_candidate_count": totals["viable_candidates"],
            "rejected_candidate_count": totals["rejected_candidates"],
            "candidate_reduction_ratio": 1 - totals["viable_candidates"] / total_raw
            if total_raw
            else None,
            "mean_candidates_per_frame": _ratio(totals["viable_candidates"], sampled_count),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "candidate_precision": None,
        },
        "tracking": {
            "implementation": tracker.implementation,
            "total_tracks_created": tracker.total_tracks_created,
            "confirmed_tracks": len(confirmed_track_ids),
            "terminated_track_count": tracker.total_tracks_terminated,
            "preferred_association_count": tracker.preferred_association_count,
            "preferred_association_override_count": (tracker.preferred_association_override_count),
            "track_fragmentation_gt_status": "NOT_AVAILABLE",
            "track_fragmentation_count": None,
            "active_track_count_mean": mean(active_track_counts) if active_track_counts else None,
            "active_track_count_max": max(active_track_counts) if active_track_counts else None,
        },
        "target_identity": {
            "target_id": manager.identity.target_id,
            "initialization_duration_us": first_lock_timestamp - first_timestamp
            if first_lock_timestamp is not None and first_timestamp is not None
            else None,
            "initial_selection_score": manager.identity.initial_selection_score,
            "first_lock_timestamp_us": first_lock_timestamp,
            **{
                f"{name.lower()}_frame_ratio": _ratio(state_counts[name], sampled_count)
                for name in TargetIdentityState.__members__
            },
            "mean_identity_confidence_when_locked": mean(locked_confidences)
            if locked_confidences
            else None,
            "active_track_id_change_count": manager.active_track_id_change_count,
            "recovery_event_count": len(manager.recovery_events),
            "relock_count": manager.relock_count,
            "mean_lost_duration_us": mean(
                [item.lost_duration_us for item in manager.recovery_events]
            )
            if manager.recovery_events
            else None,
            "max_lost_duration_us": max(
                [item.lost_duration_us for item in manager.recovery_events], default=None
            ),
            "recovery_events": [item.__dict__ for item in manager.recovery_events],
        },
        "pose_efficiency": {
            "raw_detection_person_count": total_raw,
            "pose_person_inference_count": totals["pose_person_inferences"],
            "pose_inference_reduction_ratio": 1 - totals["pose_person_inferences"] / total_raw
            if total_raw
            else None,
            "mean_pose_candidates_per_frame": mean(pose_candidates_per_frame)
            if pose_candidates_per_frame
            else None,
            "max_pose_candidates_per_frame": max(pose_candidates_per_frame, default=None),
        },
        "target_pose": {
            "target_pose_coverage": _ratio(totals["target_pose_frames"], sampled_count)
        },
        "biomechanics": {
            "target_knee_angle_coverage": _ratio(totals["target_knee_angles"], sampled_count)
        },
        "baseline_comparison": baseline,
        "performance": {
            "detector_total_seconds": stage_totals["detector"],
            "candidate_filter_total_seconds": stage_totals["candidate"],
            "tracking_total_seconds": stage_totals["tracking"],
            "identity_total_seconds": stage_totals["identity"],
            "appearance_total_seconds": stage_totals["appearance"],
            "pose_total_seconds": stage_totals["pose"],
            "canonical_adapter_total_seconds": stage_totals["adapter"],
            "biomechanics_total_seconds": stage_totals["biomechanics"],
            "total_processing_seconds": finished - started,
            "mean_detector_seconds": mean(detector_latencies) if detector_latencies else None,
            "p95_detector_seconds": _p95(detector_latencies),
            "mean_tracking_identity_seconds": mean(tracking_identity_latencies)
            if tracking_identity_latencies
            else None,
            "p95_tracking_identity_seconds": _p95(tracking_identity_latencies),
            "mean_pose_seconds": mean(pose_latencies) if pose_latencies else None,
            "p95_pose_seconds": _p95(pose_latencies),
            "mean_pipeline_seconds": mean(pipeline_latencies) if pipeline_latencies else None,
            "p95_pipeline_seconds": _p95(pipeline_latencies),
            "ground_truth_evaluation_seconds": 0.0,
        },
        "frame_observations": observations,
        "relock_frame_indices": relock_frames,
        "warnings": sorted({warning for item in observations for warning in item["warnings"]}),
        "limitations": [
            "IMAGE_2D_ONLY_NOT_PHYSICAL_3D",
            "TARGET_IDENTITY_RESEARCH_REFERENCE_ONLY",
            "NO_TEMPORAL_POSE_SMOOTHING",
            "DEEP_REID_NOT_CONFIGURED",
        ],
        "ground_truth": {
            "status": ground_truth_status,
            "contract_version": target_ground_truth.contract_version
            if target_ground_truth
            else None,
            "video_sha256": target_ground_truth.video_sha256 if target_ground_truth else None,
            "annotated_frame_count": len(target_ground_truth.frames)
            if target_ground_truth
            else None,
        },
        "identity_accuracy": {
            "correct_lock_count": None,
            "wrong_target_lock_count": None,
            "target_not_locked_count": None,
            "false_lock_when_absent_count": None,
            "target_lock_coverage_when_present": None,
            "wrong_target_rate": None,
            "false_lock_when_absent_rate": None,
            "target_frame_accuracy": None,
        },
        "target_present_state_metrics": {
            "target_present_frame_count": None,
            "target_present_and_correctly_locked_count": None,
            **{
                f"{state.lower()}_when_present_ratio": None
                for state in TargetIdentityState.__members__
            },
        },
        "target_absent_state_metrics": {
            "target_absent_frame_count": None,
            **{
                f"{state.lower()}_when_absent_ratio": None
                for state in TargetIdentityState.__members__
            },
        },
        "recovery": {
            "REAL_RECOVERY_STATUS": "NOT_EXERCISED_NO_REENTRY_VIDEO",
            "recovery_opportunity_count": None,
            "successful_recovery_count": None,
            "recovery_success_rate": None,
            "median_reacquisition_time_us": None,
            "max_reacquisition_time_us": None,
            "recovery_wrong_target_count": None,
        },
        "TARGET_IDENTITY_GT_STATUS": ground_truth_status,
        "DEEP_REID_STATUS": "NOT_CONFIGURED",
        "USER_TARGET_CORRECTION_STATUS": "DEFERRED",
        "BYTETRACK_STATUS": "NOT_INTEGRATED",
    }
    if target_ground_truth is not None and ground_truth_status == "AVAILABLE":
        gt_started = clock()
        evaluation = evaluate_target_identity_ground_truth(
            observations, target_ground_truth, ground_truth_settings
        )
        report["performance"]["ground_truth_evaluation_seconds"] = clock() - gt_started
        for key in (
            "ground_truth",
            "identity_accuracy",
            "target_present_state_metrics",
            "target_absent_state_metrics",
            "recovery",
        ):
            report[key] = evaluation[key]
        report["tracking"].update(evaluation["tracking_gt"])
        report["frame_gt_classifications"] = evaluation["frame_classifications"]
        report["TARGET_IDENTITY_GT_STATUS"] = "AVAILABLE"
    if manual_seed_application is not None:
        report["manual_target_seed"] = manual_seed_application
    return report


def _load_a2_baseline(input_path: str | Path) -> dict[str, Any]:
    path = Path("artifacts/benchmarks/a2_2") / f"{Path(input_path).stem}_2fps.json"
    if not path.is_file():
        return {"status": "NOT_AVAILABLE"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "EXECUTED",
            "raw_detection_person_count": data["pose"]["attempted_persons"],
            "pose_person_inference_count": data["pose"]["attempted_persons"],
            "pose_total_seconds": data["performance"]["pose_total_seconds"],
            "total_processing_seconds": data["performance"]["total_processing_seconds"],
            "target_knee_angle_coverage": data["analysis"]["left_knee_angle_coverage"],
        }
    except (KeyError, OSError, json.JSONDecodeError):
        return {"status": "NOT_AVAILABLE"}
