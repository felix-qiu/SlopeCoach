"""A5 debug artifacts rendered only from already-sampled benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slopecoach_ml.pose import Joint
from slopecoach_ml.pose.overlay import COCO17_EDGES

from .temporal_debug import write_temporal_debug_artifacts

_BODY_OVERLAY_JOINTS = frozenset(joint.value for edge in COCO17_EDGES for joint in edge) | {
    Joint.NOSE.value
}


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
    width = height = None
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
            _draw_temporal_skeleton(cv2, canvas, sample.get("joints", {}))
            signal = signal_by_timestamp.get(sample["timestamp_us"])
            turn = _containing_turn(turns, sample["timestamp_us"])
            facts = facts_by_timestamp.get(sample["timestamp_us"], {})
            _draw_hud(cv2, canvas, sample, raw, signal, turn, facts)
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
    }


def _draw_temporal_skeleton(cv2: Any, canvas: Any, joints: dict[str, Any]) -> None:
    for first, second in COCO17_EDGES:
        a, b = joints.get(first.value), joints.get(second.value)
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        stable_a = _point(a, "stabilized")
        stable_b = _point(b, "stabilized")
        if stable_a is None or stable_b is None:
            continue
        cv2.line(canvas, stable_a, stable_b, (0, 255, 0), 2, cv2.LINE_AA)
    for joint_name, point in joints.items():
        if joint_name not in _BODY_OVERLAY_JOINTS:
            continue
        stable = _point(point, "stabilized") if isinstance(point, dict) else None
        if stable is None:
            continue
        thickness = 2 if point.get("provenance") == "INTERPOLATED" else -1
        cv2.circle(canvas, stable, 5, (0, 255, 255), thickness, cv2.LINE_AA)


def _point(point: dict[str, Any], prefix: str):
    if point.get("provenance") not in {"OBSERVED", "INTERPOLATED"}:
        return None
    x, y = point.get(f"{prefix}_x_px"), point.get(f"{prefix}_y_px")
    if x is None or y is None:
        return None
    return round(x), round(y)


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
) -> None:
    confidence = getattr(raw_sample, "identity_confidence", None)
    identity = (
        f"identity={sample.get('identity_state')} confidence={_fmt(confidence)}"
        if confidence is not None
        else f"identity={sample.get('identity_state')} confidence=null"
    )
    proxy = signal.get("value") if signal else None
    turn_label = f"turn={turn.get('turn_id')} status={turn.get('status')}" if turn else "turn=null"
    labels = (
        f"sampled debug t={sample['timestamp_us'] / 1e6:.3f}s frame={sample['frame_index']}",
        f"target={sample.get('target_id')} track={sample.get('active_track_id')} {identity}",
        f"turn proxy / lateral proxy={_fmt(proxy)} {turn_label}",
        "2D knee L/R="
        f"{_fmt(facts.get('left_knee_angle_2d_deg'))}/"
        f"{_fmt(facts.get('right_knee_angle_2d_deg'))} deg "
        f"delta={_fmt(facts.get('bilateral_knee_abs_difference_2d_deg'))}",
        f"2D signed lateral proxy={_fmt(facts.get('signed_lateral_body_proxy'))}",
    )
    for row, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (10, 22 + 20 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
        )


def _fmt(value):
    return "null" if value is None else f"{value:.2f}"
