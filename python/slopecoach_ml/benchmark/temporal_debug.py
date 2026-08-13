"""Lazy-OpenCV A4 temporal trace and representative overlay artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from slopecoach_ml.pose.overlay import COCO17_EDGES


def write_temporal_debug_artifacts(
    output_dir: str | Path, report: dict[str, object], collector, *, max_frames: int = 12
) -> dict[str, object]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("DEBUG_DEPENDENCY_MISSING: opencv-python/numpy") from error
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / "temporal_trace.json"
    signal_path = destination / "turn_signal.json"
    events_path = destination / "turn_events.json"
    trace_path.write_text(
        json.dumps(report["temporal_trace"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    signal_path.write_text(
        json.dumps(
            {"samples": report["turn_signal_samples"], "signal_runs": report["signal_runs"]},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    events_path.write_text(
        json.dumps(
            {
                "peaks": report["turn_events"],
                "zero_crossings": report["zero_crossings"],
                "segments": report["turn_segments"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    trace = report["temporal_trace"]
    signal = report["turn_signal_samples"]
    candidates = []
    for index, sample in enumerate(trace):
        score = 0
        if sample["temporal_segment_id"] is not None and (
            index == 0 or trace[index - 1]["temporal_segment_id"] != sample["temporal_segment_id"]
        ):
            score += 20
        score += 15 * sample["interpolated_joint_count"]
        if sample["temporal_segment_id"] is None:
            score += 5
        displacement = sum(
            abs(point["raw_x_px"] - point["stabilized_x_px"])
            + abs(point["raw_y_px"] - point["stabilized_y_px"])
            for point in sample["joints"].values()
            if None
            not in (
                point["raw_x_px"],
                point["raw_y_px"],
                point["stabilized_x_px"],
                point["stabilized_y_px"],
            )
        )
        score += min(displacement, 10)
        candidates.append((score, index))
    selected = sorted({index for _, index in sorted(candidates, reverse=True)[:max_frames]})
    overlays, tiles = [], []
    for index in selected:
        sample = trace[index]
        frame_index = sample["frame_index"]
        encoded = collector.images.get(frame_index)
        if encoded is None:
            continue
        canvas = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        joints = sample["joints"]
        for first, second in COCO17_EDGES:
            a, b = joints[first.value], joints[second.value]
            if None not in (a["raw_x_px"], a["raw_y_px"], b["raw_x_px"], b["raw_y_px"]):
                cv2.line(
                    canvas,
                    (round(a["raw_x_px"]), round(a["raw_y_px"])),
                    (round(b["raw_x_px"]), round(b["raw_y_px"])),
                    (160, 160, 160),
                    1,
                )
            if None not in (
                a["stabilized_x_px"],
                a["stabilized_y_px"],
                b["stabilized_x_px"],
                b["stabilized_y_px"],
            ):
                color = (
                    (255, 0, 255)
                    if "INTERPOLATED" in (a["provenance"], b["provenance"])
                    else (0, 220, 255)
                )
                cv2.line(
                    canvas,
                    (round(a["stabilized_x_px"]), round(a["stabilized_y_px"])),
                    (round(b["stabilized_x_px"]), round(b["stabilized_y_px"])),
                    color,
                    2,
                )
        proxy = signal[index]["value"]
        signal_run_id = next(
            (
                run["signal_run_id"]
                for run in report["signal_runs"]
                if run["start_timestamp_us"] <= sample["timestamp_us"] <= run["end_timestamp_us"]
            ),
            None,
        )
        label = (
            f"t={sample['timestamp_us'] / 1e6:.2f}s seg={sample['temporal_segment_id']} "
            f"run={signal_run_id} state={sample['identity_state']} "
            f"proxy={proxy if proxy is not None else 'null'}"
        )
        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        path = destination / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError("TEMPORAL_DEBUG_OVERLAY_WRITE_FAILED")
        overlays.append(str(path))
        width = 360
        height = max(1, round(canvas.shape[0] * width / canvas.shape[1]))
        tiles.append(cv2.resize(canvas, (width, height)))
    contact = None
    if tiles:
        height = max(tile.shape[0] for tile in tiles)
        rows = []
        for offset in range(0, len(tiles), 3):
            row = [
                cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                for tile in tiles[offset : offset + 3]
            ]
            while len(row) < 3:
                row.append(np.zeros_like(row[0]))
            rows.append(cv2.hconcat(row))
        contact = destination / "contact_sheet.jpg"
        if not cv2.imwrite(str(contact), cv2.vconcat(rows)):
            raise RuntimeError("TEMPORAL_DEBUG_CONTACT_SHEET_WRITE_FAILED")
    return {
        "selected_frame_indices": [trace[index]["frame_index"] for index in selected],
        "overlay_paths": overlays,
        "contact_sheet": str(contact) if contact else None,
        "temporal_trace": str(trace_path),
        "turn_signal": str(signal_path),
        "turn_events": str(events_path),
    }
