"""A9 research/reference end-to-end result boundary."""

from .builder import GROUND_TRUTH_DEFAULTS, build_analysis_result
from .contracts import (
    ANALYSIS_RESULT_CONTRACT_VERSION,
    ANALYSIS_SECTION_CONTRACT_VERSION,
    PRODUCT_REPORT_CONTRACT_VERSION,
    PRODUCT_REPORT_PROJECTION_POLICY_VERSION,
    AnalysisQualityGateStatus,
    AnalysisResult,
    AnalysisSection,
    AnalysisSectionStatus,
    ProductReport,
)
from .golden import build_golden_case, run_analysis_result_golden
from .projection import build_product_report
from .registry import (
    ANALYSIS_SECTION_NAMES,
    ANALYSIS_SECTION_REGISTRY_SHA256,
    ANALYSIS_SECTION_REGISTRY_VERSION,
    analysis_section_registry_payload,
)

__all__ = [
    "ANALYSIS_RESULT_CONTRACT_VERSION",
    "ANALYSIS_SECTION_CONTRACT_VERSION",
    "ANALYSIS_SECTION_NAMES",
    "ANALYSIS_SECTION_REGISTRY_SHA256",
    "ANALYSIS_SECTION_REGISTRY_VERSION",
    "GROUND_TRUTH_DEFAULTS",
    "PRODUCT_REPORT_CONTRACT_VERSION",
    "PRODUCT_REPORT_PROJECTION_POLICY_VERSION",
    "AnalysisQualityGateStatus",
    "AnalysisResult",
    "AnalysisSection",
    "AnalysisSectionStatus",
    "ProductReport",
    "analysis_section_registry_payload",
    "build_analysis_result",
    "build_golden_case",
    "build_product_report",
    "run_analysis_result_golden",
]
