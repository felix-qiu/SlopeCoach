"""Single-pass B3 product orchestration over a typed internal analysis context."""

from __future__ import annotations

from slopecoach_ml.analysis_result import build_analysis_result, build_product_report
from slopecoach_ml.coach import build_coach_report
from slopecoach_ml.diagnosis import (
    build_diagnosis_semantics_provenance,
    diagnose_biomechanics,
)
from slopecoach_ml.scoring import build_scorecard

from .analysis import build_mvp_analysis_payload
from .context import AnalysisContext
from .sport_type import MvpSportTypeProvenance


def assemble_analyze_video_product(
    *,
    video: str,
    sport_type: MvpSportTypeProvenance,
    biomechanics_report: dict[str, object],
) -> dict[str, object]:
    """Convert one completed A5 pass into the typed product boundary exactly once."""

    context = AnalysisContext.from_biomechanics_report(
        video=video,
        sport_type=sport_type,
        biomechanics_report=biomechanics_report,
    )
    return assemble_analysis_context(context)


def assemble_analysis_context(context: AnalysisContext) -> dict[str, object]:
    """Assemble A7-A9 outputs using only validated AnalysisContext fields."""

    sport = {
        **context.sport_type.to_dict(),
        "CALIBRATED_FUSION_CONTROLS_ROUTING": False,
        "limitations": ["SPORT_TYPE_USER_SELECTED_ROUTING_NOT_GT"],
    }

    diagnosis = provenance = scorecard = coach = None
    unavailable = {}
    if not context.target_safe_for_analysis:
        unavailable = _downstream_unavailable("TARGET_IDENTITY_UNCERTAIN")
    elif context.qualified_turn_count == 0:
        unavailable = _downstream_unavailable("NO_QUALIFIED_TURNS")
    elif not context.biomechanics_evidence_available:
        unavailable = _downstream_unavailable("INSUFFICIENT_DIAGNOSIS_EVIDENCE")
    else:
        diagnosis_result = diagnose_biomechanics(
            sport_type_result=sport,
            biomechanics_result=context.biomechanics_result,
            turn_segments=context.turn_segments,
        )
        diagnosis = diagnosis_result.to_dict()
        provenance = build_diagnosis_semantics_provenance(
            diagnosis_result.config,
            diagnosis_contract_version=diagnosis_result.contract_version,
        ).to_dict()
        if diagnosis["status"] == "NOT_ANALYZABLE_INSUFFICIENT_DIAGNOSIS_EVIDENCE":
            unavailable = _downstream_unavailable("INSUFFICIENT_DIAGNOSIS_EVIDENCE")
            unavailable.pop("DIAGNOSIS")
        else:
            # ScoreCard and Coach are sibling consumers of Diagnosis. Product orchestration
            # never obtains the canonical ScoreCard from CoachContext.
            scorecard = build_scorecard(diagnosis_result).to_dict()
            coach = build_coach_report(diagnosis_result).to_dict()

    analysis_result = build_analysis_result(
        source=context.source,
        target_identity=context.target_identity,
        sport_type=sport,
        turns=context.turns,
        biomechanics=context.biomechanics,
        diagnosis=diagnosis,
        diagnosis_semantics_provenance=provenance,
        scorecard=scorecard,
        coach=coach,
        unavailable_reasons=unavailable,
    )
    product_report = build_product_report(analysis_result)
    return build_mvp_analysis_payload(
        video=context.video,
        sport_type=context.sport_type,
        analysis_result=analysis_result,
        product_report=product_report,
        pipeline_provenance={
            **context.pipeline_provenance,
            "analysis_context_contract_version": context.contract_version,
            "analysis_context_sha256": context.analysis_context_sha256,
        },
    )


def _downstream_unavailable(reason: str) -> dict[str, str]:
    return {name: reason for name in ("DIAGNOSIS", "SCORECARD", "COACH")}
