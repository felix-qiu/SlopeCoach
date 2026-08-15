"""Provider boundaries for future dedicated sport evidence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from slopecoach_ml.identity import TargetIdentityState
from slopecoach_ml.pose import BoundingBox2D, FrameGeometry

from .contracts import (
    SportEvidenceKind,
    SportEvidenceObservation,
    SportEvidenceProviderResult,
    SportEvidenceProviderStatus,
)


def execute_sport_evidence_providers(
    providers, frame_contexts: tuple[TargetSportFrameContext, ...]
) -> tuple[SportEvidenceProviderResult, ...]:
    """Execute providers independently so one failure cannot erase other evidence."""

    results = []
    for provider in providers:
        try:
            context = (
                frame_contexts if getattr(provider, "execution_scope", "CLIP") == "FRAME" else None
            )
            results.append(provider.infer(context))
        except Exception as error:
            results.append(
                SportEvidenceProviderResult(
                    provider_name=provider.name,
                    evidence_kind=provider.kind,
                    status=SportEvidenceProviderStatus.FAILED,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return tuple(results)


def summarize_provider_kind(results, kind: SportEvidenceKind) -> dict[str, object]:
    matching = tuple(item for item in results if item.evidence_kind is kind)
    statuses = [item.status for item in matching]
    with_evidence = statuses.count(SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE)
    successfully_executed = with_evidence + statuses.count(
        SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE
    )
    configured = sum(item is not SportEvidenceProviderStatus.NOT_CONFIGURED for item in statuses)
    failed = statuses.count(SportEvidenceProviderStatus.FAILED)
    if with_evidence:
        overall = SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE
    elif successfully_executed:
        overall = SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE
    elif configured and failed == configured:
        overall = SportEvidenceProviderStatus.FAILED
    elif configured:
        overall = SportEvidenceProviderStatus.CONFIGURED
    else:
        overall = SportEvidenceProviderStatus.NOT_CONFIGURED
    return {
        "evidence_kind": kind.value,
        "overall_status": overall.value,
        "provider_count": len(matching),
        "configured_provider_count": configured,
        "executed_provider_count": successfully_executed,
        "provider_with_evidence_count": with_evidence,
        "failed_provider_count": failed,
        "observation_count": sum(len(item.observations) for item in matching),
    }


class SportEvidenceProvider(Protocol):
    name: str
    kind: SportEvidenceKind

    def infer(self, context: Any = None) -> SportEvidenceProviderResult: ...


@dataclass(frozen=True)
class TargetSportFrameContext:
    """Provider-only target crop context; pixels never enter fusion contracts."""

    timestamp_us: int
    frame_index: int
    geometry: FrameGeometry
    target_id: str
    active_track_id: int
    target_bbox: BoundingBox2D
    identity_state: TargetIdentityState
    frame_reference: Any

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_us, bool)
            or not isinstance(self.timestamp_us, int)
            or self.timestamp_us < 0
        ):
            raise ValueError("timestamp_us must be a non-negative integer")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_id must be a non-empty string")
        if (
            isinstance(self.active_track_id, bool)
            or not isinstance(self.active_track_id, int)
            or self.active_track_id < 1
        ):
            raise ValueError("active_track_id must be a positive integer")
        if self.identity_state is not TargetIdentityState.LOCKED:
            raise ValueError("sport target context requires LOCKED target identity")
        self.geometry.validate()
        self.target_bbox.validate(self.geometry)


@dataclass(frozen=True)
class NotConfiguredSportEvidenceProvider:
    name: str
    kind: SportEvidenceKind

    def infer(self, context: Any = None) -> SportEvidenceProviderResult:
        return SportEvidenceProviderResult(
            provider_name=self.name,
            evidence_kind=self.kind,
            status=SportEvidenceProviderStatus.NOT_CONFIGURED,
            limitations=("DEDICATED_SPORT_EVIDENCE_PROVIDER_NOT_CONFIGURED",),
        )


class NotConfiguredEquipmentSportEvidenceProvider(NotConfiguredSportEvidenceProvider):
    def __init__(self) -> None:
        super().__init__("equipment-sport-provider", SportEvidenceKind.EQUIPMENT)


class NotConfiguredVisualSportEvidenceProvider(NotConfiguredSportEvidenceProvider):
    def __init__(self) -> None:
        super().__init__("visual-sport-provider", SportEvidenceKind.VISUAL_CLASSIFIER)


@dataclass(frozen=True)
class MockSportEvidenceProvider:
    name: str
    kind: SportEvidenceKind
    observations: tuple[SportEvidenceObservation, ...] = ()

    def infer(self, context: Any = None) -> SportEvidenceProviderResult:
        return SportEvidenceProviderResult(
            provider_name=self.name,
            evidence_kind=self.kind,
            status=(
                SportEvidenceProviderStatus.EXECUTED_WITH_EVIDENCE
                if self.observations
                else SportEvidenceProviderStatus.EXECUTED_NO_EVIDENCE
            ),
            observations=self.observations,
        )
