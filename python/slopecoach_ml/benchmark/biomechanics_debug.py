"""Small A5 JSON artifacts plus reuse of the A4 skeleton contact sheet."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.pose.overlay import COCO17_EDGES

from .temporal_debug import write_temporal_debug_artifacts


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


def _fmt(value):
    return "null" if value is None else f"{value:.2f}"
