"""Auto-first sport resolution with observable, authoritative user override."""

from __future__ import annotations

from .contracts import (
    SportType,
    SportTypeConfig,
    SportTypeResolutionStatus,
    SportTypeResult,
    SportTypeSource,
)
from .cues import extract_uncalibrated_sport_cues
from .fusion import ReferenceSportTypeFusion


def resolve_sport_type(
    provider_results,
    biomechanics_result=None,
    *,
    user_selection: SportType | None = None,
    config: SportTypeConfig | None = None,
    cue_measurements=None,
) -> SportTypeResult:
    results = tuple(provider_results)
    observations = tuple(observation for result in results for observation in result.observations)
    resolved_config = config or SportTypeConfig()
    auto = ReferenceSportTypeFusion(resolved_config).decide(observations)
    cues = (
        tuple(cue_measurements)
        if cue_measurements is not None
        else (
            extract_uncalibrated_sport_cues(biomechanics_result)
            if biomechanics_result is not None
            else ()
        )
    )
    if user_selection is not None:
        if user_selection is SportType.UNKNOWN:
            raise ValueError("user_selection must be SKI, SNOWBOARD, or null")
        return SportTypeResult(
            effective_sport_type=user_selection,
            effective_source=SportTypeSource.USER,
            resolution_status=SportTypeResolutionStatus.RESOLVED_USER,
            auto_decision=auto,
            user_selection=user_selection,
            auto_user_disagreement=(
                auto.sport_type is not SportType.UNKNOWN and auto.sport_type is not user_selection
            ),
            provider_results=results,
            cue_measurements=cues,
            ask_user_recommended=False,
            config=resolved_config,
        )
    return SportTypeResult(
        effective_sport_type=auto.sport_type,
        effective_source=SportTypeSource.AUTO,
        resolution_status=auto.status,
        auto_decision=auto,
        user_selection=None,
        auto_user_disagreement=False,
        provider_results=results,
        cue_measurements=cues,
        ask_user_recommended=auto.ask_user_recommended,
        config=resolved_config,
    )


def sport_specific_analysis_allowed(result: SportTypeResult) -> bool:
    return result.effective_sport_type in {SportType.SKI, SportType.SNOWBOARD}
