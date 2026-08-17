"""A8 controlled template-coach research API."""

from .contracts import (
    COACH_CONTEXT_VERSION,
    COACH_REPORT_VERSION,
    COACH_TEMPLATE_VERSION,
    DRILL_LIBRARY_VERSION,
    CoachContext,
    CoachContextStatus,
    CoachReport,
    ControlledDrill,
    ProvisionalIssueSummary,
)
from .drills import (
    DRILL_LIBRARY,
    DRILL_LIBRARY_SHA256,
    canonical_drill_library_json,
    drill_for_diagnosis,
)
from .golden import run_coach_golden
from .issues import build_issue_summaries, prioritize_issues
from .pipeline import build_coach_context, build_coach_report
from .templates import (
    COACH_TEMPLATE_REGISTRY_SHA256,
    TEMPLATE_REGISTRY,
    canonical_template_registry_json,
)

__all__ = [
    "COACH_CONTEXT_VERSION",
    "COACH_REPORT_VERSION",
    "COACH_TEMPLATE_REGISTRY_SHA256",
    "COACH_TEMPLATE_VERSION",
    "DRILL_LIBRARY",
    "DRILL_LIBRARY_SHA256",
    "DRILL_LIBRARY_VERSION",
    "TEMPLATE_REGISTRY",
    "CoachContext",
    "CoachContextStatus",
    "CoachReport",
    "ControlledDrill",
    "ProvisionalIssueSummary",
    "build_coach_context",
    "build_coach_report",
    "build_issue_summaries",
    "canonical_drill_library_json",
    "canonical_template_registry_json",
    "drill_for_diagnosis",
    "prioritize_issues",
    "run_coach_golden",
]
