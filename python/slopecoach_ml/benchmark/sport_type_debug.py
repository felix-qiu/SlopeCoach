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
    return paths
