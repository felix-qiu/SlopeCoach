"""MVP request artifact assembly without automatic SportType inference."""

from __future__ import annotations

import json
from typing import Any

from .sport_type import MvpSportTypeProvenance

MVP_ANALYZE_VIDEO_CONTRACT_VERSION = "mvp-analyze-video-v1"


def build_mvp_analysis_payload(
    *,
    video: str,
    sport_type: MvpSportTypeProvenance,
    biomechanics_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach explicit product truth to the existing research biomechanics artifact.

    This is a Python MVP/reference artifact, not the future Rust production contract.
    """

    payload = {
        "contract_version": MVP_ANALYZE_VIDEO_CONTRACT_VERSION,
        "input_video": video,
        "sport_type": sport_type.to_dict(),
        "automatic_sport_type_research": {
            "status": "DEFERRED_RESEARCH_ONLY",
            "executed": False,
        },
        "analysis": biomechanics_report,
        "limitations": [
            "PYTHON_MVP_REFERENCE_PATH_NOT_PRODUCTION_DOMAIN_KERNEL",
            "AUTOMATIC_SPORT_TYPE_DEFERRED_RESEARCH_ONLY",
        ],
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload
