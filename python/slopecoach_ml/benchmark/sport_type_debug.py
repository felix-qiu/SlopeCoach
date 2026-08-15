"""A6 strict JSON debug artifacts with reused A5/A4 visual output."""

from __future__ import annotations

import json
from pathlib import Path

from .biomechanics_debug import write_biomechanics_debug_artifacts


def write_sport_type_debug_artifacts(output_dir, report, collector, *, max_frames=12):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sport = report["sport_type"]
    payloads = {
        "sport_type_result": sport,
        "sport_evidence": sport["provider_results"],
        "sport_cues": sport["cue_measurements"],
        "equipment_provider": {
            "models": report["equipment_models"],
            "summary": report["equipment_evidence"],
            "provider_kind_summary": report["provider_validation"]["provider_kind_summaries"][0],
        },
        "equipment_frames": report["_equipment_debug_frames"],
    }
    paths = {}
    for name, payload in payloads.items():
        path = destination / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths[name] = str(path)
    paths.update(
        write_biomechanics_debug_artifacts(
            destination,
            report["_upstream_biomechanics_report"],
            collector,
            max_frames=max_frames,
        )
    )
    equipment_contact = _write_equipment_contact_sheet(destination, report, collector)
    if equipment_contact:
        paths["equipment_contact_sheet"] = equipment_contact
    return paths


def _write_equipment_contact_sheet(destination, report, collector):
    frames = report["_equipment_debug_frames"]
    if not frames:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python/numpy") from error
    tiles = []
    sport = report["sport_type"]["effective_sport_type"]
    for item in frames:
        encoded = collector.images.get(item["frame_index"])
        if encoded is None:
            continue
        canvas = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        for key, color in (
            ("target_bbox", (0, 255, 0)),
            ("crop_bbox", (255, 180, 0)),
            ("association_zone", (0, 220, 255)),
        ):
            _rectangle(cv2, canvas, item[key], color, 2)
        for detection in item["associated_detections"]:
            color = (255, 0, 255) if detection["class_name"] == "skis" else (255, 0, 0)
            _rectangle(cv2, canvas, detection["bbox"], color, 3)
        labels = (
            "EQUIPMENT EVIDENCE ONLY",
            f"SPORT TYPE = {sport}",
            f"ski={item['max_ski_support']:.2f} snowboard={item['max_snowboard_support']:.2f}",
        )
        for row, label in enumerate(labels):
            cv2.putText(
                canvas,
                label,
                (10, 24 + row * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
        width = 360
        height = max(1, round(canvas.shape[0] * width / canvas.shape[1]))
        tiles.append(cv2.resize(canvas, (width, height)))
    if not tiles:
        return None
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
    path = destination / "equipment_contact_sheet.jpg"
    if not cv2.imwrite(str(path), cv2.vconcat(rows)):
        raise RuntimeError("EQUIPMENT_CONTACT_SHEET_WRITE_FAILED")
    return str(path)


def _rectangle(cv2, image, bbox, color, thickness):
    left, top = round(bbox["x_px"]), round(bbox["y_px"])
    right = round(bbox["x_px"] + bbox["width_px"])
    bottom = round(bbox["y_px"] + bbox["height_px"])
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
