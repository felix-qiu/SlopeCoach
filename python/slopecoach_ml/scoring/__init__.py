"""A8 structure-only scorecard research API."""

from .contracts import (
    DIAGNOSIS_DIMENSION_REGISTRY_VERSION,
    ISSUE_PRIORITY_POLICY_VERSION,
    SCORECARD_CONTRACT_VERSION,
    SCORING_POLICY_VERSION,
    DimensionAssessment,
    DimensionAssessmentStatus,
    IssuePriorityPolicy,
    ScoreCard,
    ScoreDimension,
    ScoringPolicy,
    canonical_issue_priority_policy_json,
    issue_priority_policy_sha256,
)
from .golden import run_scorecard_golden
from .provenance_golden import run_a8_provenance_golden
from .registry import (
    DIAGNOSIS_DIMENSION_REGISTRY,
    DIAGNOSIS_DIMENSION_REGISTRY_SHA256,
    DiagnosisDimensionMapping,
    canonical_dimension_registry_json,
)
from .scorecard import SCORECARD_LIMITATIONS, build_scorecard

__all__ = [
    "DIAGNOSIS_DIMENSION_REGISTRY",
    "DIAGNOSIS_DIMENSION_REGISTRY_SHA256",
    "DIAGNOSIS_DIMENSION_REGISTRY_VERSION",
    "ISSUE_PRIORITY_POLICY_VERSION",
    "SCORECARD_CONTRACT_VERSION",
    "SCORECARD_LIMITATIONS",
    "SCORING_POLICY_VERSION",
    "DiagnosisDimensionMapping",
    "DimensionAssessment",
    "DimensionAssessmentStatus",
    "IssuePriorityPolicy",
    "ScoreCard",
    "ScoreDimension",
    "ScoringPolicy",
    "build_scorecard",
    "canonical_issue_priority_policy_json",
    "canonical_dimension_registry_json",
    "run_scorecard_golden",
    "run_a8_provenance_golden",
    "issue_priority_policy_sha256",
]
