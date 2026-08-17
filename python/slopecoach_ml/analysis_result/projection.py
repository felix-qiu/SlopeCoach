"""Pure app-facing projection; imports no upstream truth engines."""

from __future__ import annotations

from .contracts import AnalysisResult, ProductReport


def build_product_report(result: AnalysisResult) -> ProductReport:
    sections = {section.name: section for section in result.sections}

    def payload(name):
        section = sections[name]
        return section.payload if section.status.value != "UNAVAILABLE" else None

    sport = payload("SPORT_TYPE")
    scorecard = payload("SCORECARD")
    coach = payload("COACH")
    if result.quality_gate_status.value == "NOT_ANALYZABLE":
        scorecard = None
        coach = None
    return ProductReport(
        source_analysis_result_sha256=str(result.analysis_result_sha256),
        status=result.quality_gate_status,
        primary_reason_code=result.primary_reason_code,
        availability={name: sections[name].status.value for name in sections},
        sport=sport,
        scorecard=scorecard,
        headline=coach.get("headline") if coach else None,
        top_issues=tuple(coach.get("top_issues", ())) if coach else (),
        practice_plan=tuple(coach.get("practice_plan", ())) if coach else (),
        evidence_summary=coach.get("evidence_summary") if coach else None,
        warnings=result.warnings,
        limitations=result.limitations,
    )
