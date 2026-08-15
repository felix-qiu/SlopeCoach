"""Independent, explicitly-authored A6 SportType Golden runner."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import (
    SportEvidenceKind,
    SportEvidenceObservation,
    SportEvidenceScope,
    SportType,
)
from .pipeline import resolve_sport_type
from .providers import MockSportEvidenceProvider


def run_sport_type_golden(path: str | Path) -> dict[str, object]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    all_passed = True
    for case in fixture["cases"]:
        grouped = {}
        for item in case["observations"]:
            observation = SportEvidenceObservation(
                evidence_id=item["evidence_id"],
                kind=SportEvidenceKind(item["kind"]),
                provider_name=item["provider_name"],
                timestamp_us=item.get("timestamp_us"),
                temporal_segment_id=item.get("temporal_segment_id"),
                ski_support=item["ski_support"],
                snowboard_support=item["snowboard_support"],
                quality=item["quality"],
                scope=SportEvidenceScope(item["scope"]),
                reason=item.get("reason"),
                limitations=tuple(item.get("limitations", ())),
            )
            grouped.setdefault((observation.provider_name, observation.kind), []).append(
                observation
            )
        providers = tuple(
            MockSportEvidenceProvider(name, kind, tuple(observations)).infer()
            for (name, kind), observations in sorted(
                grouped.items(), key=lambda pair: (pair[0][1].value, pair[0][0])
            )
        )
        user = SportType(case["user_selection"]) if case.get("user_selection") else None
        result = resolve_sport_type(providers, user_selection=user)
        actual = {
            "effective_sport_type": result.effective_sport_type.value,
            "effective_source": result.effective_source.value,
            "resolution_status": result.resolution_status.value,
            "auto_sport_type": result.auto_decision.sport_type.value,
            "auto_status": result.auto_decision.status.value,
            "auto_user_disagreement": result.auto_user_disagreement,
            "ask_user_recommended": result.ask_user_recommended,
        }
        passed = actual == case["expected"]
        all_passed &= passed
        cases.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "expected": case["expected"],
                "actual": actual,
            }
        )
    payload = {
        "golden_passed": all_passed,
        "fixture_contract_version": fixture["contract_version"],
        "cases": cases,
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload
