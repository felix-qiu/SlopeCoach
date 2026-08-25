"""MVP product artifact assembly without automatic SportType inference."""

from __future__ import annotations

import json

from slopecoach_ml.analysis_result import AnalysisResult, ProductReport

from .sport_type import MvpSportTypeProvenance

MVP_ANALYZE_VIDEO_CONTRACT_VERSION = "mvp-analyze-video-v1"


def build_mvp_analysis_payload(
    *,
    video: str,
    sport_type: MvpSportTypeProvenance,
    analysis_result: AnalysisResult,
    product_report: ProductReport,
    pipeline_provenance: dict[str, object],
) -> dict[str, object]:
    """Serialize one B3 artifact containing the canonical A9 result and projection.

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
        "quality_gate": {
            "status": analysis_result.quality_gate_status.value,
            "reason_codes": list(analysis_result.blockers),
            "primary_reason_code": analysis_result.primary_reason_code,
        },
        "analysis_result": analysis_result.to_dict(),
        "product_report": product_report.to_dict(),
        "pipeline_provenance": pipeline_provenance,
        "limitations": [
            "PYTHON_MVP_REFERENCE_PATH_NOT_PRODUCTION_DOMAIN_KERNEL",
            "AUTOMATIC_SPORT_TYPE_DEFERRED_RESEARCH_ONLY",
        ],
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload
