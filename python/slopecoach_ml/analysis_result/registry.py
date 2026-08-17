"""Fixed, ordered A9 analysis section registry."""

from __future__ import annotations

from .fingerprint import semantic_sha256

ANALYSIS_SECTION_REGISTRY_VERSION = "analysis-section-registry-v1"
ANALYSIS_SECTION_NAMES = (
    "SOURCE",
    "TARGET_IDENTITY",
    "SPORT_TYPE",
    "TURNS",
    "BIOMECHANICS",
    "DIAGNOSIS",
    "SCORECARD",
    "COACH",
)

_REGISTRY_PAYLOAD = {
    "version": ANALYSIS_SECTION_REGISTRY_VERSION,
    "ordered_sections": list(ANALYSIS_SECTION_NAMES),
    "availability_policy": {
        "AVAILABLE": "PAYLOAD_REQUIRED",
        "PARTIAL": "PAYLOAD_REQUIRED_WITH_LIMITATIONS_OR_REASONS",
        "UNAVAILABLE": "PAYLOAD_FORBIDDEN",
    },
}
ANALYSIS_SECTION_REGISTRY_SHA256 = semantic_sha256(_REGISTRY_PAYLOAD)


def analysis_section_registry_payload() -> dict[str, object]:
    return {
        "version": _REGISTRY_PAYLOAD["version"],
        "ordered_sections": list(ANALYSIS_SECTION_NAMES),
        "availability_policy": dict(_REGISTRY_PAYLOAD["availability_policy"]),
    }
