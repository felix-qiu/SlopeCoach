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
    SportTypeRoutingBasis,
)


@dataclass(frozen=True)
class _KindSupport:
    kind: SportEvidenceKind
    ski: float
    snowboard: float


@dataclass(frozen=True)
class _SupportEvaluation:
    winner: SportType | None
    status: SportTypeResolutionStatus
    reason: str


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
        common = {
            "active_evidence_kinds": active,
            "primary_evidence_kinds": primary,
            "evidence_observation_count": len(ordered),
            "limitations": ("SUPPORT_VALUES_ARE_NOT_CALIBRATED_PROBABILITIES",),
        }
        if not primary:
            return self._unknown(
                SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE,
                ("NO_ACTIVE_PRIMARY_EVIDENCE",),
                None,
                **common,
            )

        by_kind = {item.kind: item for item in supports}
        equipment = by_kind.get(SportEvidenceKind.EQUIPMENT)
        visual = by_kind.get(SportEvidenceKind.VISUAL_CLASSIFIER)
        equipment_evaluation = self._evaluate(equipment) if equipment else None
        visual_evaluation = self._evaluate(visual) if visual else None

        if equipment_evaluation and equipment_evaluation.winner is not None:
            reasons = ["EQUIPMENT_PRIMARY_THRESHOLDS_SATISFIED"]
            visual_direction = self._direction(visual)
            if visual_direction is equipment_evaluation.winner:
                reasons.append("EQUIPMENT_PRIMARY_VISUAL_AGREES")
            elif visual_direction is not None:
                reasons.append("EQUIPMENT_PRIMARY_VISUAL_DISAGREES")
            return self._resolved(
                equipment_evaluation.winner,
                SportTypeRoutingBasis.EQUIPMENT_PRIMARY,
                equipment,
                tuple(reasons),
                **common,
            )

        if visual_evaluation and visual_evaluation.winner is not None:
            equipment_observed = any(item.kind is SportEvidenceKind.EQUIPMENT for item in ordered)
            fallback_reason = (
                "VISUAL_FALLBACK_EQUIPMENT_INSUFFICIENT"
                if equipment is not None or equipment_observed
                else "VISUAL_FALLBACK_EQUIPMENT_UNAVAILABLE"
            )
            reasons = [fallback_reason]
            equipment_direction = self._direction(equipment)
            if (
                equipment_direction is not None
                and equipment_direction is not visual_evaluation.winner
            ):
                reasons.append("VISUAL_FALLBACK_WEAK_EQUIPMENT_DISAGREEMENT")
            return self._resolved(
                visual_evaluation.winner,
                SportTypeRoutingBasis.VISUAL_FALLBACK,
                visual,
                tuple(reasons),
                **common,
            )

        if equipment is not None and visual is not None:
            equipment_direction = self._direction(equipment)
            visual_direction = self._direction(visual)
            combined = self._combine((equipment, visual))
            if (
                equipment_direction is not None
                and visual_direction is not None
                and equipment_direction is not visual_direction
            ):
                return self._unknown(
                    SportTypeResolutionStatus.CONFLICTING_PRIMARY_EVIDENCE,
                    ("PRIMARY_KINDS_CONFLICT_UNRESOLVED",),
                    combined,
                    **common,
                )
            if equipment_direction is not None and equipment_direction is visual_direction:
                agreement_evaluation = self._evaluate_pair(combined)
                if agreement_evaluation.winner is not None:
                    return self._resolved(
                        agreement_evaluation.winner,
                        SportTypeRoutingBasis.PRIMARY_AGREEMENT,
                        combined,
                        ("PRIMARY_KINDS_AGREE",),
                        **common,
                    )
                return self._unknown(
                    agreement_evaluation.status,
                    ("PRIMARY_KINDS_AGREE", agreement_evaluation.reason),
                    combined,
                    **common,
                )

        available = equipment or visual
        evaluation = equipment_evaluation or visual_evaluation
        prefix = (
            "EQUIPMENT_PRIMARY_INSUFFICIENT"
            if equipment is not None
            else "VISUAL_FALLBACK_THRESHOLDS_NOT_SATISFIED"
        )
        return self._unknown(
            evaluation.status,
            (prefix, evaluation.reason),
            available,
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

    def _evaluate(self, support: _KindSupport) -> _SupportEvaluation:
        return self._evaluate_pair((support.ski, support.snowboard))

    def _evaluate_pair(self, support: tuple[float, float]) -> _SupportEvaluation:
        ski, snowboard = support
        if (
            ski >= self.config.primary_conflict_support
            and snowboard >= self.config.primary_conflict_support
        ):
            return _SupportEvaluation(
                None,
                SportTypeResolutionStatus.CONFLICTING_PRIMARY_EVIDENCE,
                "PRIMARY_KIND_SUPPORTS_BOTH_SPORTS",
            )
        if ski == snowboard:
            return _SupportEvaluation(
                None, SportTypeResolutionStatus.AMBIGUOUS, "EXACT_SUPPORT_TIE"
            )
        winner = SportType.SKI if ski > snowboard else SportType.SNOWBOARD
        winner_support = ski if winner is SportType.SKI else snowboard
        if winner_support < self.config.minimum_primary_support:
            return _SupportEvaluation(
                None,
                SportTypeResolutionStatus.INSUFFICIENT_PRIMARY_EVIDENCE,
                "WINNER_PRIMARY_SUPPORT_BELOW_MINIMUM",
            )
        if winner_support < self.config.minimum_auto_support:
            return _SupportEvaluation(
                None,
                SportTypeResolutionStatus.INSUFFICIENT_TOTAL_EVIDENCE,
                "WINNER_SUPPORT_BELOW_AUTO_MINIMUM",
            )
        if abs(ski - snowboard) < self.config.minimum_auto_margin:
            return _SupportEvaluation(
                None,
                SportTypeResolutionStatus.AMBIGUOUS,
                "AUTO_SUPPORT_MARGIN_BELOW_MINIMUM",
            )
        return _SupportEvaluation(
            winner,
            SportTypeResolutionStatus.RESOLVED_AUTO,
            "AUTO_THRESHOLDS_SATISFIED",
        )

    @staticmethod
    def _direction(support: _KindSupport | None) -> SportType | None:
        if support is None or support.ski == support.snowboard:
            return None
        return SportType.SKI if support.ski > support.snowboard else SportType.SNOWBOARD

    @staticmethod
    def _support_values(support: _KindSupport | tuple[float, float] | None):
        if support is None:
            return None, None, None
        ski, snowboard = (
            (support.ski, support.snowboard) if isinstance(support, _KindSupport) else support
        )
        return ski, snowboard, abs(ski - snowboard)

    @classmethod
    def _resolved(cls, sport_type, routing_basis, support, reason_codes, **kwargs):
        ski, snowboard, margin = cls._support_values(support)
        return AutoSportTypeDecision(
            sport_type=sport_type,
            status=SportTypeResolutionStatus.RESOLVED_AUTO,
            routing_basis=routing_basis,
            ski_support=ski,
            snowboard_support=snowboard,
            margin=margin,
            ask_user_recommended=False,
            reason_codes=reason_codes,
            **kwargs,
        )

    @classmethod
    def _unknown(cls, status, reason_codes, support, **kwargs):
        ski, snowboard, margin = cls._support_values(support)
        return AutoSportTypeDecision(
            sport_type=SportType.UNKNOWN,
            status=status,
            routing_basis=SportTypeRoutingBasis.NONE,
            ski_support=ski,
            snowboard_support=snowboard,
            margin=margin,
            ask_user_recommended=True,
            reason_codes=reason_codes,
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
