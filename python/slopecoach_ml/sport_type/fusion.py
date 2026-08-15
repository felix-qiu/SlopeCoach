"""Deterministic, dependency-free A6 sport evidence fusion."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    AutoSportTypeDecision,
    SportEvidenceKind,
    SportEvidenceObservation,
    SportType,
    SportTypeConfig,
    SportTypeResolutionStatus,
)


@dataclass(frozen=True)
class _KindSupport:
    kind: SportEvidenceKind
    ski: float
    snowboard: float


class ReferenceSportTypeFusion:
    """Engineering support fusion; returned supports are not probabilities."""

    def __init__(self, config: SportTypeConfig | None = None) -> None:
        self.config = config or SportTypeConfig()

    def decide(self, observations) -> AutoSportTypeDecision:
        ordered = tuple(sorted(observations, key=_evidence_sort_key))
        ids = [item.evidence_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence_id in auto decision run")
        supports = tuple(
            support
            for kind in SportEvidenceKind
            if (support := self._aggregate_kind(kind, ordered)) is not None
        )
        active = tuple(item.kind for item in supports)
        primary = tuple(item.kind for item in supports if item.kind.is_primary)
        combined = self._combine(supports)
        ski, snowboard = combined if combined else (None, None)
        margin = abs(ski - snowboard) if ski is not None and snowboard is not None else None
        common = {
            "ski_support": ski,
            "snowboard_support": snowboard,
            "margin": margin,
            "active_evidence_kinds": active,
            "primary_evidence_kinds": primary,
            "evidence_observation_count": len(ordered),
            "limitations": ("SUPPORT_VALUES_ARE_NOT_CALIBRATED_PROBABILITIES",),
        }
        if not primary:
            return self._unknown(
                SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE,
                "NO_ACTIVE_PRIMARY_EVIDENCE",
                **common,
            )
        primary_supports = tuple(item for item in supports if item.kind.is_primary)
        if (
            max(item.ski for item in primary_supports) >= self.config.primary_conflict_support
            and max(item.snowboard for item in primary_supports)
            >= self.config.primary_conflict_support
        ):
            return self._unknown(
                SportTypeResolutionStatus.CONFLICTING_PRIMARY_EVIDENCE,
                "PRIMARY_EVIDENCE_SUPPORTS_BOTH_SPORTS",
                **common,
            )
        if ski == snowboard:
            return self._unknown(SportTypeResolutionStatus.AMBIGUOUS, "EXACT_SUPPORT_TIE", **common)
        winner = SportType.SKI if ski > snowboard else SportType.SNOWBOARD
        primary_combined = self._combine(primary_supports)
        winner_primary = primary_combined[0 if winner is SportType.SKI else 1]
        if winner_primary < self.config.minimum_primary_support:
            return self._unknown(
                SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE,
                "WINNER_PRIMARY_SUPPORT_BELOW_MINIMUM",
                **common,
            )
        winner_support = ski if winner is SportType.SKI else snowboard
        if winner_support < self.config.minimum_auto_support:
            return self._unknown(
                SportTypeResolutionStatus.INSUFFICIENT_TOTAL_EVIDENCE,
                "WINNER_COMBINED_SUPPORT_BELOW_MINIMUM",
                **common,
            )
        if margin < self.config.minimum_auto_margin:
            return self._unknown(
                SportTypeResolutionStatus.AMBIGUOUS,
                "AUTO_SUPPORT_MARGIN_BELOW_MINIMUM",
                **common,
            )
        return AutoSportTypeDecision(
            sport_type=winner,
            status=SportTypeResolutionStatus.RESOLVED_AUTO,
            ask_user_recommended=False,
            reason_codes=("AUTO_THRESHOLDS_SATISFIED",),
            **common,
        )

    def _aggregate_kind(self, kind, observations):
        items = tuple(item for item in observations if item.kind is kind)
        clip_active = any(item.scope.value == "CLIP" for item in items)
        timestamps = {item.timestamp_us for item in items if item.scope.value == "FRAME"}
        if not clip_active and len(timestamps) < self.config.minimum_frame_observations_per_kind:
            return None
        denominator = sum(item.quality for item in items)
        if denominator <= 0:
            return None
        return _KindSupport(
            kind,
            sum(item.ski_support * item.quality for item in items) / denominator,
            sum(item.snowboard_support * item.quality for item in items) / denominator,
        )

    def _combine(self, supports):
        denominator = sum(self.config.weight_for(item.kind) for item in supports)
        if denominator <= 0:
            return None
        return (
            sum(self.config.weight_for(item.kind) * item.ski for item in supports) / denominator,
            sum(self.config.weight_for(item.kind) * item.snowboard for item in supports)
            / denominator,
        )

    @staticmethod
    def _unknown(status, reason, **kwargs):
        return AutoSportTypeDecision(
            sport_type=SportType.UNKNOWN,
            status=status,
            ask_user_recommended=True,
            reason_codes=(reason,),
            **kwargs,
        )


def _evidence_sort_key(item: SportEvidenceObservation):
    return (
        item.kind.value,
        item.scope.value,
        -1 if item.timestamp_us is None else item.timestamp_us,
        item.evidence_id,
        item.provider_name,
    )
