"""A5 debug artifacts rendered only from already-sampled benchmark evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from slopecoach_ml.pose import Joint
from slopecoach_ml.pose.overlay import COCO17_EDGES
from slopecoach_ml.temporal import TemporalPoseConfig

from .temporal_debug import write_temporal_debug_artifacts

_BODY_OVERLAY_JOINTS = frozenset(joint.value for edge in COCO17_EDGES for joint in edge) | {
    Joint.NOSE.value
}
_STABILIZED_LINE_COLOR = (0, 255, 0)
_STABILIZED_JOINT_COLOR = (0, 255, 255)
_TARGET_BBOX_COLOR = (0, 255, 0)


def write_biomechanics_debug_artifacts(output_dir, report, collector, *, max_frames=12):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "frame_biomechanics": report["biomechanics_result"]["frame_facts"],
        "segment_biomechanics": report["temporal_segment_features"],
        "turn_biomechanics": report["turn_biomechanics"],
        "feature_coverage": report["frame_biomechanics"]["feature_coverage"],
        "feature_schema": {
            "feature_schema_version": report["feature_schema_version"],
            "feature_registry_sha256": report["feature_registry_sha256"],
            "FIXED_ML_FEATURE_VECTOR_STATUS": report["config"]["FIXED_ML_FEATURE_VECTOR_STATUS"],
            "feature_registry": report["feature_registry"],
        },
    }
    paths = {}
    for name, payload in files.items():
        path = destination / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths[name] = str(path)
    upstream = report["_upstream_debug_report"]
    paths.update(
        write_temporal_debug_artifacts(destination, upstream, collector, max_frames=max_frames)
    )
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python/numpy") from error
    frame_facts = {}
    for fact in report["biomechanics_result"]["frame_facts"]:
        frame_facts.setdefault(fact["timestamp_us"], {})[fact["feature_id"]] = fact["value"]
    trace_by_timestamp = {item["timestamp_us"]: item for item in upstream["temporal_trace"]}
    selected = paths["selected_frame_indices"]
    tiles = []
    for frame_index in selected:
        encoded = collector.images.get(frame_index)
        sample = next(
            (item for item in trace_by_timestamp.values() if item["frame_index"] == frame_index),
            None,
        )
        if encoded is None or sample is None:
            continue
        canvas = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        joints = sample["joints"]
        for first, second in COCO17_EDGES:
            a, b = joints[first.value], joints[second.value]
            if None not in (
                a["stabilized_x_px"],
                a["stabilized_y_px"],
                b["stabilized_x_px"],
                b["stabilized_y_px"],
            ):
                cv2.line(
                    canvas,
                    (round(a["stabilized_x_px"]), round(a["stabilized_y_px"])),
                    (round(b["stabilized_x_px"]), round(b["stabilized_y_px"])),
                    (0, 220, 255),
                    2,
                )
        values = frame_facts.get(sample["timestamp_us"], {})
        labels = (
            f"2D L/R knee={_fmt(values.get('left_knee_angle_2d_deg'))}/"
            f"{_fmt(values.get('right_knee_angle_2d_deg'))} deg",
            f"2D knee delta={_fmt(values.get('bilateral_knee_abs_difference_2d_deg'))} "
            f"ankle norm={_fmt(values.get('ankle_separation_body_scale'))}",
            f"image lateral proxy={_fmt(values.get('signed_lateral_body_proxy'))}",
        )
        for row, label in enumerate(labels):
            cv2.putText(
                canvas,
                label,
                (10, 22 + 18 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )
        width = 360
        height = max(1, round(canvas.shape[0] * width / canvas.shape[1]))
        tiles.append(cv2.resize(canvas, (width, height)))
    if tiles:
        tile_height = max(tile.shape[0] for tile in tiles)
        rows = []
        for offset in range(0, len(tiles), 3):
            row = [
                cv2.copyMakeBorder(tile, 0, tile_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                for tile in tiles[offset : offset + 3]
            ]
            while len(row) < 3:
                row.append(np.zeros_like(row[0]))
            rows.append(cv2.hconcat(row))
        contact = destination / "contact_sheet.jpg"
        if not cv2.imwrite(str(contact), cv2.vconcat(rows)):
            raise RuntimeError("BIOMECHANICS_DEBUG_CONTACT_SHEET_WRITE_FAILED")
        paths["contact_sheet"] = str(contact)
    return paths


def write_biomechanics_overlay_video(
    output_path: str | Path,
    report: dict[str, Any],
    collector: Any,
    *,
    fps: float,
) -> dict[str, Any]:
    """Write a sampled debug MP4 without reading the source or rerunning a model."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python/numpy") from error

    upstream = report.get("_upstream_debug_report")
    trace = upstream.get("temporal_trace") if isinstance(upstream, dict) else None
    images = getattr(collector, "images", None)
    if not trace or not images:
        raise RuntimeError("BIOMECHANICS_DEBUG_VIDEO_NO_FRAMES")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered_trace = sorted(trace, key=lambda item: (item["timestamp_us"], item["frame_index"]))
    raw_samples = {
        (sample.timestamp_us, sample.frame_index): sample
        for sample in getattr(collector, "samples", ())
    }
    signal_by_timestamp = {
        sample["timestamp_us"]: sample
        for sample in upstream.get("turn_signal_samples", [])
        if isinstance(sample, dict) and "timestamp_us" in sample
    }
    facts_by_timestamp: dict[int, dict[str, Any]] = {}
    for fact in report.get("biomechanics_result", {}).get("frame_facts", []):
        timestamp = fact.get("timestamp_us")
        if timestamp is not None:
            facts_by_timestamp.setdefault(timestamp, {})[fact["feature_id"]] = fact.get("value")
    turns = report.get("turn_segments", [])

    writer = None
    written = 0
    skipped = 0
    raw_pose_frames = 0
    gated_raw_pose_frames = 0
    trusted_stabilized_pose_frames = 0
    width = height = None
    selection_source = (
        report.get("manual_target_seed", {}).get("selection_source", "AUTO")
        if isinstance(report.get("manual_target_seed"), dict)
        else "AUTO"
    )
    target_bboxes = getattr(collector, "target_bboxes", {})
    identity_debug = getattr(collector, "identity_debug", {})
    try:
        for sample in ordered_trace:
            frame_index = sample["frame_index"]
            encoded = images.get(frame_index)
            if encoded is None:
                skipped += 1
                continue
            canvas = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if canvas is None:
                skipped += 1
                continue
            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    str(destination),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("BIOMECHANICS_DEBUG_VIDEO_WRITER_OPEN_FAILED")
            elif canvas.shape[1] != width or canvas.shape[0] != height:
                canvas = cv2.resize(canvas, (width, height))

            raw = raw_samples.get((sample["timestamp_us"], frame_index))
            sample_key = (sample["timestamp_us"], frame_index)
            bbox = target_bboxes.get(sample_key)
            if bbox is None and raw is not None and raw.raw_target_pose is not None:
                bbox = raw.raw_target_pose.bbox.to_dict()
            _draw_target_bbox(cv2, canvas, bbox)
            raw_available, stabilized_available = _draw_pose_layer(
                cv2, canvas, raw, sample.get("joints", {})
            )
            analysis_trusted = sample.get("temporal_segment_id") is not None
            raw_pose_frames += int(raw_available)
            gated_raw_pose_frames += int(raw_available and not analysis_trusted)
            trusted_stabilized_pose_frames += int(analysis_trusted and stabilized_available)
            signal = signal_by_timestamp.get(sample["timestamp_us"])
            turn = _containing_turn(turns, sample["timestamp_us"])
            facts = facts_by_timestamp.get(sample["timestamp_us"], {})
            _draw_angle_panel(
                cv2,
                canvas,
                _angle_panel_values(raw, sample.get("joints", {}), facts),
            )
            _draw_hud(
                cv2,
                canvas,
                sample,
                raw,
                signal,
                turn,
                facts,
                selection_source=selection_source,
                raw_available=raw_available,
                stabilized_available=stabilized_available,
                analysis_trusted=analysis_trusted,
                identity_debug=identity_debug.get(sample_key, {}),
            )
            writer.write(canvas)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    if written == 0 or width is None or height is None:
        raise RuntimeError("BIOMECHANICS_DEBUG_VIDEO_NO_FRAMES")
    return {
        "path": str(destination),
        "kind": "SAMPLED_DEBUG_VIDEO",
        "fps": float(fps),
        "frame_count": written,
        "skipped_frame_count": skipped,
        "width_px": width,
        "height_px": height,
        "source_model_rerun": False,
        "raw_target_pose_debug": {
            "enabled": True,
            "raw_pose_frame_count": raw_pose_frames,
            "analysis_gated_raw_pose_frame_count": gated_raw_pose_frames,
            "trusted_stabilized_pose_frame_count": trusted_stabilized_pose_frames,
        },
    }


def _draw_target_bbox(cv2: Any, canvas: Any, bbox: Any) -> None:
    if not isinstance(bbox, dict):
        return
    values = tuple(bbox.get(key) for key in ("x_px", "y_px", "width_px", "height_px"))
    if any(not isinstance(value, int | float) for value in values):
        return
    x_px, y_px, width_px, height_px = values
    if width_px <= 0 or height_px <= 0:
        return
    cv2.rectangle(
        canvas,
        (round(x_px), round(y_px)),
        (round(x_px + width_px), round(y_px + height_px)),
        _TARGET_BBOX_COLOR,
        1,
        cv2.LINE_AA,
    )


def _raw_pose_points(raw_sample: Any) -> dict[str, tuple[int, int]]:
    pose = getattr(raw_sample, "raw_target_pose", None)
    geometry = getattr(raw_sample, "geometry", None)
    if pose is None or geometry is None:
        return {}
    minimum_confidence = TemporalPoseConfig().minimum_joint_confidence
    points = {}
    for joint in Joint:
        if joint.value not in _BODY_OVERLAY_JOINTS:
            continue
        point = pose.joint(joint)
        if (
            point is None
            or point.confidence < minimum_confidence
            or not point.is_inside_frame(geometry)
        ):
            continue
        points[joint.value] = (round(point.x_px), round(point.y_px))
    return points


def _draw_raw_target_pose(cv2: Any, canvas: Any, raw_sample: Any) -> bool:
    """Draw only current-frame model output; never interpolation or biomechanics."""
    points = _raw_pose_points(raw_sample)
    if not points:
        return False
    point_radius = _point_radius_from_y(point[1] for point in points.values())
    for first, second in COCO17_EDGES:
        start, end = points.get(first.value), points.get(second.value)
        if start is not None and end is not None:
            cv2.line(canvas, start, end, _STABILIZED_LINE_COLOR, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(
            canvas,
            point,
            point_radius,
            _STABILIZED_JOINT_COLOR,
            -1,
            cv2.LINE_AA,
        )
    return True


def _draw_pose_layer(
    cv2: Any, canvas: Any, raw_sample: Any, joints: dict[str, Any]
) -> tuple[bool, bool]:
    """Render exactly one skeleton, preferring trusted temporal coordinates."""
    raw_available = bool(_raw_pose_points(raw_sample))
    stabilized_available = _stabilized_pose_available(joints)
    if stabilized_available:
        _draw_temporal_skeleton(cv2, canvas, joints)
    elif raw_available:
        _draw_raw_target_pose(cv2, canvas, raw_sample)
    return raw_available, stabilized_available


def _draw_temporal_skeleton(cv2: Any, canvas: Any, joints: dict[str, Any]) -> None:
    point_radius = _overlay_point_radius(joints)
    for first, second in COCO17_EDGES:
        a, b = joints.get(first.value), joints.get(second.value)
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        stable_a = _point(a, "stabilized")
        stable_b = _point(b, "stabilized")
        if stable_a is None or stable_b is None:
            continue
        cv2.line(canvas, stable_a, stable_b, _STABILIZED_LINE_COLOR, 2, cv2.LINE_AA)
    for joint_name, point in joints.items():
        if joint_name not in _BODY_OVERLAY_JOINTS:
            continue
        stable = _point(point, "stabilized") if isinstance(point, dict) else None
        if stable is None:
            continue
        thickness = 2 if point.get("provenance") == "INTERPOLATED" else -1
        cv2.circle(canvas, stable, point_radius, _STABILIZED_JOINT_COLOR, thickness, cv2.LINE_AA)


def _stabilized_pose_available(joints: dict[str, Any]) -> bool:
    return any(
        joint_name in _BODY_OVERLAY_JOINTS
        and isinstance(point, dict)
        and _point(point, "stabilized") is not None
        for joint_name, point in joints.items()
    )


def _overlay_point_radius(joints: dict[str, Any]) -> int:
    y_coordinates = [
        point["stabilized_y_px"]
        for joint_name, point in joints.items()
        if joint_name in _BODY_OVERLAY_JOINTS
        and isinstance(point, dict)
        and _point(point, "stabilized") is not None
    ]
    return _point_radius_from_y(y_coordinates)


def _point_radius_from_y(y_coordinates: Any) -> int:
    values = tuple(y_coordinates)
    if len(values) < 2:
        return 2
    pose_height = max(values) - min(values)
    return max(2, min(5, int(pose_height * 0.012 + 0.5)))


def _point(point: dict[str, Any], prefix: str):
    if point.get("provenance") not in {"OBSERVED", "INTERPOLATED"}:
        return None
    x, y = point.get(f"{prefix}_x_px"), point.get(f"{prefix}_y_px")
    if x is None or y is None:
        return None
    return round(x), round(y)


def _angle_panel_values(
    raw_sample: Any, joints: dict[str, Any], facts: dict[str, Any]
) -> dict[str, float | None]:
    stable_points = {
        joint_name: stable
        for joint_name, point in joints.items()
        if joint_name in _BODY_OVERLAY_JOINTS
        and isinstance(point, dict)
        and (stable := _point(point, "stabilized")) is not None
    }
    points = stable_points or _raw_pose_points(raw_sample)
    shoulder = _axis_angle(points, Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
    hip = _axis_angle(points, Joint.LEFT_HIP, Joint.RIGHT_HIP)
    left_knee = _finite_fact(facts.get("left_knee_angle_2d_deg"))
    right_knee = _finite_fact(facts.get("right_knee_angle_2d_deg"))
    if left_knee is None:
        left_knee = _three_point_angle(points, Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
    if right_knee is None:
        right_knee = _three_point_angle(
            points, Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE
        )
    angulation = (
        _normalize_axis_angle(shoulder - hip) if shoulder is not None and hip is not None else None
    )
    return {
        "Shoulder": shoulder,
        "Hip": hip,
        "R Knee": right_knee,
        "L Knee": left_knee,
        "Angulation": angulation,
    }


def _finite_fact(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _axis_angle(points: dict[str, tuple[int, int]], first: Joint, second: Joint) -> float | None:
    start, end = points.get(first.value), points.get(second.value)
    if start is None or end is None or start == end:
        return None
    return _normalize_axis_angle(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))


def _normalize_axis_angle(value: float) -> float:
    while value > 90:
        value -= 180
    while value <= -90:
        value += 180
    return value


def _three_point_angle(
    points: dict[str, tuple[int, int]], first: Joint, vertex: Joint, third: Joint
) -> float | None:
    a, b, c = points.get(first.value), points.get(vertex.value), points.get(third.value)
    if a is None or b is None or c is None:
        return None
    first_vector = (a[0] - b[0], a[1] - b[1])
    second_vector = (c[0] - b[0], c[1] - b[1])
    first_length = math.hypot(*first_vector)
    second_length = math.hypot(*second_vector)
    if first_length == 0 or second_length == 0:
        return None
    cosine = sum(x * y for x, y in zip(first_vector, second_vector, strict=True)) / (
        first_length * second_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _draw_angle_panel(cv2: Any, canvas: Any, values: dict[str, float | None]) -> None:
    left, top, width, height = 12, 12, 260, 190
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (left, top),
        (left + width, top + height),
        (8, 8, 8),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(
        canvas,
        (left, top),
        (left + width, top + height),
        (130, 130, 130),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Angles (2D)",
        (left + 18, top + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for row, (label, value) in enumerate(values.items()):
        baseline = top + 62 + row * 27
        cv2.putText(
            canvas,
            label,
            (left + 18, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            _fmt_panel_angle(value),
            (left + 176, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _containing_turn(turns: list[dict[str, Any]], timestamp_us: int):
    for turn in turns:
        start = turn.get("start_timestamp_us")
        end = turn.get("end_timestamp_us")
        if start is not None and end is not None and start <= timestamp_us <= end:
            return turn
    return None


def _draw_hud(
    cv2: Any,
    canvas: Any,
    sample: dict[str, Any],
    raw_sample: Any,
    signal: dict[str, Any] | None,
    turn: dict[str, Any] | None,
    facts: dict[str, Any],
    *,
    selection_source: str,
    raw_available: bool,
    stabilized_available: bool,
    analysis_trusted: bool,
    identity_debug: dict[str, Any],
) -> None:
    confidence = getattr(raw_sample, "identity_confidence", None)
    match_score = identity_debug.get("latest_identity_match_score")
    observed_age_us = identity_debug.get("last_observed_age_us")
    identity = (
        f"identity={sample.get('identity_state')} confidence={_fmt(confidence)} "
        f"match_score={_fmt(match_score)} observed_age_ms={_fmt_us_ms(observed_age_us)}"
    )
    proxy = signal.get("value") if signal else None
    turn_label = f"turn={turn.get('turn_id')} status={turn.get('status')}" if turn else "turn=null"
    labels = (
        f"sampled debug t={sample['timestamp_us'] / 1e6:.3f}s frame={sample['frame_index']}",
        f"target_source={selection_source} target={sample.get('target_id')} "
        f"track={sample.get('active_track_id')}",
        identity,
        f"analysis={'TRUSTED' if analysis_trusted else 'GATED'} "
        f"raw_pose={'AVAILABLE' if raw_available else 'UNAVAILABLE'} "
        f"stabilized_pose={'AVAILABLE' if stabilized_available else 'UNAVAILABLE'}",
        "RAW POSE DEBUG ONLY" if raw_available and not analysis_trusted else "",
        f"turn proxy / lateral proxy={_fmt(proxy)} {turn_label}",
        "2D knee L/R="
        f"{_fmt(facts.get('left_knee_angle_2d_deg'))}/"
        f"{_fmt(facts.get('right_knee_angle_2d_deg'))} deg "
        f"delta={_fmt(facts.get('bilateral_knee_abs_difference_2d_deg'))}",
        f"2D signed lateral proxy={_fmt(facts.get('signed_lateral_body_proxy'))}",
    )
    for row, label in enumerate(item for item in labels if item):
        cv2.putText(
            canvas,
            label,
            (10, 224 + 20 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )


def _fmt(value):
    return "null" if value is None else f"{value:.2f}"


def _fmt_us_ms(value):
    return "null" if value is None else f"{value / 1000:.0f}"


def _fmt_panel_angle(value: float | None) -> str:
    return "--" if value is None else f"{value:.0f} deg"
