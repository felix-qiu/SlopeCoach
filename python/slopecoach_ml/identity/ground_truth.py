"""Strict A3.1 target-identity ground truth contract and benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any

from slopecoach_ml.pose import BoundingBox2D, CoordinateSpace, FrameGeometry
from slopecoach_ml.tracking import bbox_iou

GT_CONTRACT_VERSION = "target-identity-gt-v1"


class GroundTruthTargetState(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNCERTAIN = "UNCERTAIN"
    UNLABELED = "UNLABELED"


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def video_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetGroundTruthFrame:
    timestamp_us: int
    frame_index: int
    target_state: GroundTruthTargetState
    bbox: BoundingBox2D | None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetGroundTruthFrame:
        timestamp = _integer(data["timestamp_us"], "timestamp_us")
        frame_index = _integer(data["frame_index"], "frame_index")
        if timestamp < 0 or frame_index < 0:
            raise ValueError("GT timestamp/frame index must be non-negative")
        try:
            state = GroundTruthTargetState(data["target_state"])
        except ValueError as error:
            raise ValueError("invalid target_state") from error
        raw_bbox = data.get("bbox")
        bbox = BoundingBox2D.from_dict(raw_bbox) if raw_bbox is not None else None
        if state is GroundTruthTargetState.PRESENT and bbox is None:
            raise ValueError("PRESENT requires bbox")
        if state is GroundTruthTargetState.ABSENT and bbox is not None:
            raise ValueError("ABSENT requires bbox=null")
        if bbox is not None:
            if bbox.coordinate_space is not CoordinateSpace.SOURCE_PIXEL_2D:
                raise ValueError("GT coordinate_space must be SourcePixel2D")
            for value in (bbox.x_px, bbox.y_px, bbox.width_px, bbox.height_px):
                if not math.isfinite(value):
                    raise ValueError("GT bbox must be finite")
            if bbox.width_px <= 0 or bbox.height_px <= 0:
                raise ValueError("GT bbox dimensions must be positive")
        notes = data.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise TypeError("notes must be string or null")
        return cls(timestamp, frame_index, state, bbox, notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_us": self.timestamp_us,
            "frame_index": self.frame_index,
            "target_state": self.target_state.value,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TargetIdentityGroundTruth:
    contract_version: str
    video_sha256: str
    video_path_hint: str
    coordinate_space: str
    annotation_source: str
    sample_fps: float
    width_px: int
    height_px: int
    duration_seconds: float | None
    frames: tuple[TargetGroundTruthFrame, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetIdentityGroundTruth:
        if data.get("contract_version") != GT_CONTRACT_VERSION:
            raise ValueError("unsupported GT contract_version")
        if data.get("coordinate_space") != "SourcePixel2D":
            raise ValueError("GT coordinate_space must be SourcePixel2D")
        sha = data.get("video_sha256")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("video_sha256 must be lowercase SHA256")
        sample_fps = data.get("sample_fps")
        if isinstance(sample_fps, bool) or not isinstance(sample_fps, int | float):
            raise TypeError("sample_fps must be numeric")
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        width = _integer(data["width_px"], "width_px")
        height = _integer(data["height_px"], "height_px")
        if width <= 0 or height <= 0:
            raise ValueError("GT dimensions must be positive")
        duration = data.get("duration_seconds")
        if duration is not None:
            if isinstance(duration, bool) or not isinstance(duration, int | float):
                raise TypeError("duration_seconds must be numeric or null")
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("duration_seconds must be finite and non-negative")
        frames = tuple(TargetGroundTruthFrame.from_dict(item) for item in data.get("frames", []))
        timestamps = [item.timestamp_us for item in frames]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("duplicate GT timestamp_us")
        for frame in frames:
            if frame.bbox is not None:
                frame.bbox.validate(FrameGeometry(width, height))
        source = data.get("annotation_source")
        if not isinstance(source, str) or not source:
            raise ValueError("annotation_source must be non-empty")
        hint = data.get("video_path_hint")
        if not isinstance(hint, str):
            raise TypeError("video_path_hint must be a string")
        return cls(
            GT_CONTRACT_VERSION,
            sha,
            hint,
            "SourcePixel2D",
            source,
            float(sample_fps),
            width,
            height,
            float(duration) if duration is not None else None,
            frames,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["frames"] = [item.to_dict() for item in self.frames]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def validate_video(self, video_path: str | Path) -> None:
        if video_sha256(video_path) != self.video_sha256:
            raise ValueError("TARGET_IDENTITY_GT_VIDEO_MISMATCH")


@dataclass(frozen=True)
class GroundTruthEvaluationConfig:
    target_match_iou_threshold: float = 0.5
    maximum_timestamp_delta_us: int = 50_000

    def validate(self) -> None:
        if isinstance(self.target_match_iou_threshold, bool) or not isinstance(
            self.target_match_iou_threshold, int | float
        ):
            raise TypeError("target_match_iou_threshold must be numeric")
        if not math.isfinite(self.target_match_iou_threshold):
            raise ValueError("target_match_iou_threshold must be finite")
        if not 0 <= self.target_match_iou_threshold <= 1:
            raise ValueError("target_match_iou_threshold must be in [0, 1]")
        _integer(self.maximum_timestamp_delta_us, "maximum_timestamp_delta_us")
        if self.maximum_timestamp_delta_us < 0:
            raise ValueError("maximum_timestamp_delta_us must be non-negative")


def load_target_ground_truth(path: str | Path, video_path: str | Path | None = None):
    data = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")),
    )
    gt = TargetIdentityGroundTruth.from_dict(data)
    if video_path is not None:
        gt.validate_video(video_path)
    return gt


def evaluate_target_identity_ground_truth(
    observations: list[dict[str, Any]],
    ground_truth: TargetIdentityGroundTruth,
    config: GroundTruthEvaluationConfig | None = None,
) -> dict[str, Any]:
    settings = config or GroundTruthEvaluationConfig()
    settings.validate()
    unused = set(range(len(ground_truth.frames)))
    classifications = []
    counts = Counter()
    state_present = Counter()
    state_absent = Counter()
    target_track_ids = []
    matched = []
    for observation in observations:
        candidates = [
            (abs(frame.timestamp_us - observation["timestamp_us"]), index, frame)
            for index, frame in enumerate(ground_truth.frames)
            if index in unused
            and abs(frame.timestamp_us - observation["timestamp_us"])
            <= settings.maximum_timestamp_delta_us
        ]
        if not candidates:
            continue
        delta, index, frame = min(candidates)
        unused.remove(index)
        matched.append(frame)
        state = observation["identity_state"]
        locked = state == "LOCKED" and observation.get("selected_bbox") is not None
        iou = None
        if frame.target_state in {
            GroundTruthTargetState.UNCERTAIN,
            GroundTruthTargetState.UNLABELED,
        }:
            classification = "EXCLUDED_FROM_GT_METRICS"
        elif frame.target_state is GroundTruthTargetState.PRESENT:
            state_present[state] += 1
            if locked:
                system_bbox = BoundingBox2D.from_dict(observation["selected_bbox"])
                iou = bbox_iou(frame.bbox, system_bbox)
                classification = (
                    "CORRECT_LOCK"
                    if iou >= settings.target_match_iou_threshold
                    else "WRONG_TARGET_LOCK"
                )
                if (
                    classification == "CORRECT_LOCK"
                    and observation.get("active_track_id") is not None
                ):
                    target_track_ids.append(observation["active_track_id"])
            else:
                classification = "TARGET_NOT_LOCKED"
        else:
            state_absent[state] += 1
            classification = (
                "FALSE_LOCK_WHEN_TARGET_ABSENT" if locked else "CORRECT_TARGET_ABSENT_HANDLING"
            )
        counts[classification] += 1
        classifications.append(
            {
                "timestamp_us": observation["timestamp_us"],
                "gt_timestamp_us": frame.timestamp_us,
                "timestamp_delta_us": delta,
                "gt_target_state": frame.target_state.value,
                "classification": classification,
                "selected_target_iou": iou,
            }
        )
    annotated = Counter(frame.target_state.value for frame in ground_truth.frames)
    present = annotated["PRESENT"]
    absent = annotated["ABSENT"]
    evaluable = present + absent
    system_locked = (
        counts["CORRECT_LOCK"]
        + counts["WRONG_TARGET_LOCK"]
        + counts["FALSE_LOCK_WHEN_TARGET_ABSENT"]
    )
    wrong = counts["WRONG_TARGET_LOCK"] + counts["FALSE_LOCK_WHEN_TARGET_ABSENT"]
    correctly_handled = counts["CORRECT_LOCK"] + counts["CORRECT_TARGET_ABSENT_HANDLING"]
    track_changes = sum(
        a != b for a, b in zip(target_track_ids, target_track_ids[1:], strict=False)
    )
    recovery = _recovery_metrics(observations, ground_truth, classifications)
    return {
        "ground_truth": {
            "status": "AVAILABLE",
            "contract_version": ground_truth.contract_version,
            "video_sha256": ground_truth.video_sha256,
            "annotation_source": ground_truth.annotation_source,
            "annotated_frame_count": len(ground_truth.frames),
            "present_frame_count": present,
            "absent_frame_count": absent,
            "uncertain_frame_count": annotated["UNCERTAIN"],
            "unlabeled_frame_count": annotated["UNLABELED"],
            "matched_frame_count": len(matched),
        },
        "identity_accuracy": {
            "correct_lock_count": counts["CORRECT_LOCK"],
            "wrong_target_lock_count": counts["WRONG_TARGET_LOCK"],
            "target_not_locked_count": counts["TARGET_NOT_LOCKED"],
            "false_lock_when_absent_count": counts["FALSE_LOCK_WHEN_TARGET_ABSENT"],
            "target_lock_coverage_when_present": counts["CORRECT_LOCK"] / present
            if present
            else None,
            "wrong_target_rate": wrong / system_locked if system_locked else None,
            "false_lock_when_absent_rate": counts["FALSE_LOCK_WHEN_TARGET_ABSENT"] / absent
            if absent
            else None,
            "target_frame_accuracy": correctly_handled / evaluable if evaluable else None,
            "formulas": {
                "target_lock_coverage_when_present": "CORRECT_LOCK / PRESENT",
                "wrong_target_rate": (
                    "(WRONG_TARGET_LOCK + FALSE_LOCK_WHEN_TARGET_ABSENT) / "
                    "system LOCKED on evaluable GT"
                ),
                "target_frame_accuracy": (
                    "(CORRECT_LOCK + CORRECT_TARGET_ABSENT_HANDLING) / (PRESENT + ABSENT)"
                ),
            },
        },
        "target_present_state_metrics": {
            "target_present_frame_count": present,
            "target_present_and_correctly_locked_count": counts["CORRECT_LOCK"],
            **_state_metrics(state_present, present, "when_present"),
        },
        "target_absent_state_metrics": {
            "target_absent_frame_count": absent,
            **_state_metrics(state_absent, absent, "when_absent"),
        },
        "recovery": recovery,
        "tracking_gt": {
            "track_fragmentation_gt_status": "AVAILABLE"
            if counts["CORRECT_LOCK"]
            else "NOT_AVAILABLE",
            "track_fragmentation_count": track_changes if counts["CORRECT_LOCK"] else None,
            "target_track_id_change_count_gt": track_changes if counts["CORRECT_LOCK"] else None,
        },
        "frame_classifications": classifications,
    }


def _state_metrics(counts: Counter, denominator: int, suffix: str) -> dict[str, float | None]:
    return {
        f"{state.lower()}_{suffix}_ratio": counts[state] / denominator if denominator else None
        for state in ("LOCKED", "SUSPECT", "LOST", "RECOVERING", "AMBIGUOUS", "UNINITIALIZED")
    }


def _recovery_metrics(observations, ground_truth, classifications):
    by_timestamp = {item["gt_timestamp_us"]: item for item in classifications}
    opportunities = successes = wrong = 0
    times = []
    had_correct_lock = False
    interruption: str | None = None
    reentry_timestamp = None
    wrong_during_event = False
    for frame in ground_truth.frames:
        item = by_timestamp.get(frame.timestamp_us)
        classification = item["classification"] if item else None
        if frame.target_state is GroundTruthTargetState.PRESENT:
            if interruption == "ABSENT":
                opportunities += 1
                reentry_timestamp = frame.timestamp_us
                interruption = "RECOVERING"
            elif had_correct_lock and interruption is None and classification != "CORRECT_LOCK":
                # A GT-backed system loss while the target remains present is also a
                # recovery opportunity. The first non-correct sample starts its clock.
                opportunities += 1
                reentry_timestamp = frame.timestamp_us
                interruption = "RECOVERING"
            if interruption and classification == "CORRECT_LOCK":
                successes += 1
                times.append(frame.timestamp_us - reentry_timestamp)
                interruption = None
                wrong_during_event = False
            elif interruption and classification == "WRONG_TARGET_LOCK":
                if not wrong_during_event:
                    wrong += 1
                    wrong_during_event = True
            if classification == "CORRECT_LOCK":
                had_correct_lock = True
        elif frame.target_state is GroundTruthTargetState.ABSENT and had_correct_lock:
            interruption = "ABSENT"
            reentry_timestamp = None
            wrong_during_event = False
    status = "EXERCISED" if opportunities else "NOT_EXERCISED_NO_TARGET_REENTRY"
    return {
        "REAL_RECOVERY_STATUS": status,
        "recovery_opportunity_count": opportunities,
        "successful_recovery_count": successes,
        "recovery_success_rate": successes / opportunities if opportunities else None,
        "reacquisition_time_us_per_event": times,
        "median_reacquisition_time_us": median(times) if times else None,
        "max_reacquisition_time_us": max(times) if times else None,
        "recovery_wrong_target_count": wrong,
    }
