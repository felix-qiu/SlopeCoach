from __future__ import annotations

import json

import pytest

from slopecoach_ml.detection import Detection
from slopecoach_ml.identity import (
    AutoInitialTargetSelector,
    CandidateFilterConfig,
    InitialTargetSelectorConfig,
    PersonCandidate,
    PoseSchedulingConfig,
    TargetIdentityConfig,
    TargetIdentityManager,
    TargetIdentityState,
    clipped_crop_bounds,
    descriptor_similarity,
    evaluate_candidates,
    schedule_pose_track_ids,
    target_biomechanics_allowed,
    update_gallery,
    weighted_available,
)
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry
from slopecoach_ml.tracking import (
    ReferenceMotionIoUTracker,
    TrackObservation,
    TrackState,
    TrackingConfig,
)


GEOMETRY = FrameGeometry(1000, 600)


def detection(
    detection_id: int, x: float, *, y=100.0, width=100.0, height=200.0, confidence=0.9
):
    return Detection(detection_id, BoundingBox2D(x, y, width, height), confidence)


def track(
    track_id: int, x: float, *, detection_id=None, hits=3, timestamp=0, missing=0
):
    return TrackObservation(
        track_id,
        track_id if detection_id is None else detection_id,
        BoundingBox2D(x, 100, 100, 200),
        0.9,
        TrackState.MISSING if missing else TrackState.CONFIRMED,
        hits,
        0,
        timestamp - missing,
        missing,
        20.0,
        0.0,
    )


def candidate(item: TrackObservation, quality=0.9):
    return PersonCandidate(
        item.detection_id,
        item.bbox,
        item.confidence,
        quality,
        {"area_fraction": item.bbox.width_px * item.bbox.height_px / 600_000},
    )


def test_candidate_quality_keeps_raw_order_independent_and_conservatively_rejects() -> (
    None
):
    detections = (
        detection(1, 100),
        detection(2, 200, width=2, height=4),
        detection(3, -500),
    )
    result = evaluate_candidates(detections, GEOMETRY)
    assert result[0].hard_rejection_reason is None
    assert result[1].hard_rejection_reason == "TOO_SMALL"
    assert result[2].hard_rejection_reason == "NEARLY_INVISIBLE"
    assert all(0 <= item.quality_score <= 1 for item in result)


def test_candidate_config_validation() -> None:
    with pytest.raises(ValueError):
        CandidateFilterConfig(minimum_area_fraction=-1).validate()


def test_tracker_stable_linear_motion_velocity_and_shuffle() -> None:
    first = ReferenceMotionIoUTracker()
    second = ReferenceMotionIoUTracker()
    for tracker, ordered in (
        (first, (detection(1, 100), detection(2, 600))),
        (second, (detection(2, 600), detection(1, 100))),
    ):
        frame = tracker.update(ordered, 0, 0, GEOMETRY)
        assert [item.track_id for item in frame.tracks] == [1, 2]
        frame = tracker.update(
            (detection(9, 110), detection(8, 590)), 500_000, 1, GEOMETRY
        )
        assert [round(item.velocity_x_px_per_s) for item in frame.tracks] == [20, -20]
    assert [
        item.bbox.x_px for item in first.update((), 600_000, 2, GEOMETRY).tracks
    ] == [110, 590]
    assert [
        item.bbox.x_px for item in second.update((), 600_000, 2, GEOMETRY).tracks
    ] == [110, 590]


def test_tracker_short_miss_termination_and_new_id_use_timestamps() -> None:
    tracker = ReferenceMotionIoUTracker(
        TrackingConfig(maximum_missed_duration_us=500_000)
    )
    assert tracker.update((detection(1, 100),), 0, 0, GEOMETRY).tracks[0].track_id == 1
    assert (
        tracker.update((), 400_000, 200, GEOMETRY).tracks[0].state is TrackState.MISSING
    )
    assert tracker.update((), 500_001, 201, GEOMETRY).tracks == ()
    assert tracker.total_tracks_terminated == 1
    assert (
        tracker.update((detection(2, 800),), 600_000, 202, GEOMETRY).tracks[0].track_id
        == 2
    )


def test_tracker_impossible_jump_and_scale_change_do_not_silently_match() -> None:
    tracker = ReferenceMotionIoUTracker()
    tracker.update((detection(1, 0),), 0, 0, GEOMETRY)
    frame = tracker.update(
        (detection(2, 850, width=300, height=500),), 100_000, 1, GEOMETRY
    )
    assert {item.track_id for item in frame.tracks} == {1, 2}


def test_tracker_crossing_is_deterministic_not_detection_order_based() -> None:
    tracker = ReferenceMotionIoUTracker()
    tracker.update((detection(1, 200), detection(2, 600)), 0, 0, GEOMETRY)
    tracker.update((detection(3, 300), detection(4, 500)), 500_000, 1, GEOMETRY)
    result = tracker.update(
        (detection(6, 400), detection(5, 400)), 1_000_000, 2, GEOMETRY
    )
    assert [item.track_id for item in result.tracks] == sorted(
        item.track_id for item in result.tracks
    )


def test_initial_selector_temporal_motion_beats_static_background() -> None:
    selector = AutoInitialTargetSelector(
        InitialTargetSelectorConfig(
            initialization_window_us=1_000_000,
            minimum_track_observations=3,
            minimum_winner_margin=0.05,
        )
    )
    result = None
    for index, timestamp in enumerate((0, 500_000, 1_000_000)):
        moving = track(1, 350 + 40 * index, timestamp=timestamp)
        static = track(2, 450, timestamp=timestamp)
        candidates = {1: candidate(moving), 2: candidate(static)}
        result = selector.observe((static, moving), candidates, GEOMETRY, timestamp)
    assert result.state is TargetIdentityState.LOCKED
    assert result.selected_track_id == 1


def test_initial_selector_equal_candidates_are_ambiguous_under_shuffle() -> None:
    config = InitialTargetSelectorConfig(
        initialization_window_us=0, minimum_track_observations=1
    )
    decisions = []
    for ordering in ((track(1, 300), track(2, 500)), (track(2, 500), track(1, 300))):
        selector = AutoInitialTargetSelector(config)
        result = selector.observe(
            ordering,
            {item.detection_id: candidate(item) for item in ordering},
            GEOMETRY,
            0,
        )
        decisions.append((result.state, result.selected_track_id))
    assert decisions == [(TargetIdentityState.AMBIGUOUS, None)] * 2


def test_initial_selector_no_person_remains_uninitialized() -> None:
    selector = AutoInitialTargetSelector(
        InitialTargetSelectorConfig(initialization_window_us=0)
    )
    result = selector.observe((), {}, GEOMETRY, 10_000_000)
    assert result.state is TargetIdentityState.UNINITIALIZED
    assert result.selected_track_id is None


def test_giant_one_frame_transient_does_not_beat_persistent_track() -> None:
    selector = AutoInitialTargetSelector(
        InitialTargetSelectorConfig(
            initialization_window_us=1_000_000,
            minimum_track_observations=3,
            minimum_winner_margin=0.04,
        )
    )
    persistent = None
    for index, timestamp in enumerate((0, 500_000, 1_000_000)):
        persistent = track(1, 350 + 30 * index, timestamp=timestamp)
        tracks = (
            (persistent, track(9, 200, timestamp=timestamp))
            if index == 2
            else (persistent,)
        )
        result = selector.observe(
            tracks,
            {item.detection_id: candidate(item) for item in tracks},
            GEOMETRY,
            timestamp,
        )
    assert result.selected_track_id == 1


def test_selector_missing_pose_and_motion_are_null_and_weights_renormalize() -> None:
    selector = AutoInitialTargetSelector(
        InitialTargetSelectorConfig(
            initialization_window_us=0, minimum_track_observations=1
        )
    )
    item = track(1, 400)
    result = selector.observe((item,), {1: candidate(item)}, GEOMETRY, 0)
    evidence = result.evidence[1]
    assert evidence.motion_score is None
    assert evidence.pose_quality_score is None
    assert result.score is not None
    assert weighted_available({"a": 1.0, "b": None}, {"a": 1.0, "b": 99.0}) == 1.0


def test_appearance_similarity_gallery_quality_and_bounds() -> None:
    assert descriptor_similarity((1, 0), (1, 0)) == pytest.approx(1)
    assert descriptor_similarity((1, 0), (0, 1)) == pytest.approx(0)
    assert descriptor_similarity((), ()) is None
    gallery = []
    update_gallery(gallery, (1, 0), quality=0.1, maximum_length=2)
    assert gallery == []
    for descriptor in ((1, 0), (0.5, 0.5), (0, 1)):
        update_gallery(gallery, descriptor, quality=0.9, maximum_length=2)
    assert gallery == [(0.5, 0.5), (0.0, 1.0)]


def test_appearance_crop_clips_partial_bbox_without_changing_canonical_bbox() -> None:
    bbox = BoundingBox2D(-10, -5, 30, 20)
    assert clipped_crop_bounds(100, 100, bbox) == (0, 0, 20, 15)
    assert bbox.x_px == -10
    assert clipped_crop_bounds(100, 100, BoundingBox2D(-10, -10, 5, 5)) is None


def test_target_and_track_identity_are_structurally_distinct_and_relock_new_track() -> (
    None
):
    manager = TargetIdentityManager(
        TargetIdentityConfig(
            suspect_timeout_us=100_000,
            lost_timeout_us=500_000,
            recovery_confirmation_observations=2,
        ),
        target_id="filmed-skier",
    )
    original = track(7, 300)
    manager.initialize(original, 0.9, 0)
    assert manager.identity.target_id == "filmed-skier"
    assert manager.identity.active_track_id == 7
    manager.update((), {}, 150_000)
    assert manager.identity.state is TargetIdentityState.SUSPECT
    manager.update((), {}, 300_000)
    assert manager.identity.state is TargetIdentityState.LOST
    replacement = track(12, 305, timestamp=350_000)
    manager.update((replacement,), {12: candidate(replacement)}, 350_000)
    assert manager.identity.state is TargetIdentityState.RECOVERING
    manager.update((replacement,), {12: candidate(replacement)}, 450_000)
    assert manager.identity.state is TargetIdentityState.LOCKED
    assert manager.identity.active_track_id == 12
    assert manager.identity.target_id == "filmed-skier"
    assert manager.relock_count == 1


def test_brief_miss_is_suspect_then_same_track_locks_again() -> None:
    manager = TargetIdentityManager(TargetIdentityConfig(suspect_timeout_us=500_000))
    original = track(4, 300)
    manager.initialize(original, 0.9, 0)
    manager.update((), {}, 100_000)
    assert manager.identity.state is TargetIdentityState.SUSPECT
    manager.update((original,), {4: candidate(original)}, 200_000)
    # Strong evidence for the same active track can safely clear SUSPECT.
    assert manager.identity.state is TargetIdentityState.LOCKED


def test_similar_recovery_candidates_are_ambiguous_and_do_not_relock() -> None:
    manager = TargetIdentityManager(
        TargetIdentityConfig(suspect_timeout_us=0, lost_timeout_us=1)
    )
    original = track(1, 400)
    manager.initialize(original, 0.9, 0)
    manager.update((), {}, 1)
    manager.update((), {}, 2)
    first, second = track(8, 395), track(9, 405)
    manager.update((first, second), {8: candidate(first), 9: candidate(second)}, 3)
    assert manager.identity.state is TargetIdentityState.AMBIGUOUS
    assert manager.identity.active_track_id is None


@pytest.mark.parametrize(
    "state",
    [
        TargetIdentityState.UNINITIALIZED,
        TargetIdentityState.SUSPECT,
        TargetIdentityState.LOST,
        TargetIdentityState.RECOVERING,
        TargetIdentityState.AMBIGUOUS,
    ],
)
def test_non_locked_states_suppress_biomechanics(state) -> None:
    assert not target_biomechanics_allowed(state, 1.0, 0.6)


def test_locked_safe_confidence_permits_biomechanics() -> None:
    assert target_biomechanics_allowed(TargetIdentityState.LOCKED, 0.7, 0.6)
    assert not target_biomechanics_allowed(TargetIdentityState.LOCKED, 0.5, 0.6)


def test_pose_scheduler_is_bounded_and_state_aware() -> None:
    config = PoseSchedulingConfig(
        max_initial_pose_probe_candidates=2, max_identity_pose_candidates_per_frame=2
    )
    assert schedule_pose_track_ids(
        TargetIdentityState.UNINITIALIZED, None, [1, 2, 3], config
    ) == (1, 2)
    assert schedule_pose_track_ids(
        TargetIdentityState.LOCKED, 7, [1, 2, 7], config
    ) == (7,)
    assert schedule_pose_track_ids(
        TargetIdentityState.SUSPECT, 7, [1, 2, 7], config
    ) == (7, 1)
    assert schedule_pose_track_ids(TargetIdentityState.LOST, None, [1, 2], config) == ()


def test_config_and_identity_serialization_are_deterministic() -> None:
    manager = TargetIdentityManager(target_id="stable-target")
    rendered = json.dumps(manager.to_dict(), sort_keys=True, allow_nan=False)
    assert rendered == json.dumps(manager.to_dict(), sort_keys=True, allow_nan=False)
    with pytest.raises(ValueError):
        TargetIdentityConfig(minimum_winner_margin=2).validate()
