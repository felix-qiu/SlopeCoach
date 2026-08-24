"""Explicit user SportType boundary for the MVP product request path.

Automatic A6 evidence providers and fusion deliberately do not participate here. They remain
research-only and are reachable through the dedicated research benchmark commands.
"""

from __future__ import annotations

from dataclasses import dataclass

from slopecoach_ml.sport_type.contracts import (
    SportType,
    SportTypeResolutionStatus,
    SportTypeSource,
)

MVP_SPORT_TYPE_CONTRACT_VERSION = "mvp-user-sport-type-v1"
_PRODUCT_SPORT_TYPES = frozenset((SportType.SKI, SportType.SNOWBOARD))


@dataclass(frozen=True)
class MvpSportTypeProvenance:
    """Truth-preserving product provenance with no inferred SportType fields."""

    effective_sport_type: SportType
    effective_source: SportTypeSource = SportTypeSource.USER
    resolution_status: SportTypeResolutionStatus = SportTypeResolutionStatus.RESOLVED_USER
    contract_version: str = MVP_SPORT_TYPE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.effective_sport_type not in _PRODUCT_SPORT_TYPES:
            raise ValueError("MVP_SPORT_TYPE_REQUIRED: expected SKI or SNOWBOARD")
        if self.effective_source is not SportTypeSource.USER:
            raise ValueError("MVP SportType source must be USER")
        if self.resolution_status is not SportTypeResolutionStatus.RESOLVED_USER:
            raise ValueError("MVP SportType resolution must be RESOLVED_USER")
        if self.contract_version != MVP_SPORT_TYPE_CONTRACT_VERSION:
            raise ValueError(f"contract_version must be {MVP_SPORT_TYPE_CONTRACT_VERSION}")

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_version": self.contract_version,
            "effective_sport_type": self.effective_sport_type.value,
            "effective_source": self.effective_source.value,
            "resolution_status": self.resolution_status.value,
        }


def select_user_sport_type(value: SportType | str | None) -> MvpSportTypeProvenance:
    """Resolve only explicit product values; never delegates to A6 automatic fusion."""

    if value is None:
        raise ValueError("MVP_SPORT_TYPE_REQUIRED: expected SKI or SNOWBOARD")
    if isinstance(value, str):
        try:
            value = SportType(value.upper())
        except ValueError as error:
            raise ValueError("MVP_SPORT_TYPE_REQUIRED: expected SKI or SNOWBOARD") from error
    if not isinstance(value, SportType) or value not in _PRODUCT_SPORT_TYPES:
        raise ValueError("MVP_SPORT_TYPE_REQUIRED: expected SKI or SNOWBOARD")
    return MvpSportTypeProvenance(effective_sport_type=value)
