from __future__ import annotations

from .contracts import PoseSchedulingConfig, TargetIdentityState


def schedule_pose_track_ids(state, active_track_id, ranked_track_ids, config=None):
    settings = config or PoseSchedulingConfig()
    settings.validate()
    ranked = list(dict.fromkeys(ranked_track_ids))
    if state is TargetIdentityState.LOCKED:
        return (active_track_id,) if active_track_id is not None else ()
    cap = settings.max_identity_pose_candidates_per_frame
    if state is TargetIdentityState.UNINITIALIZED:
        cap = min(cap, settings.max_initial_pose_probe_candidates)
        return tuple(ranked[:cap])
    if state in {
        TargetIdentityState.SUSPECT,
        TargetIdentityState.RECOVERING,
        TargetIdentityState.AMBIGUOUS,
    }:
        if active_track_id is not None:
            ranked = [active_track_id, *[item for item in ranked if item != active_track_id]]
        return tuple(ranked[:cap])
    return ()


def target_biomechanics_allowed(state, confidence, safe_threshold):
    return state is TargetIdentityState.LOCKED and confidence >= safe_threshold
