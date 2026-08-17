"""Provider/clip/source aggregation without treating frames as independent samples."""

from __future__ import annotations

import statistics
from collections import defaultdict

from ..contracts import SportEvidenceKind
from .contracts import RawProviderSportEvidenceSummary


def summarize_observations(
    *,
    provider_name: str,
    evidence_kind: SportEvidenceKind,
    source_video_id: str,
    video_sha256: str,
    observations: list[dict[str, object]],
    provenance: dict[str, object] | None = None,
) -> RawProviderSportEvidenceSummary:
    usable = []
    for item in observations:
        quality = _ratio(item.get("quality"), "quality")
        ski = _ratio(item.get("ski_support"), "ski_support")
        snowboard = _ratio(item.get("snowboard_support"), "snowboard_support")
        usable.append((ski, snowboard, quality, item.get("timestamp_us")))
    quality_sum = sum(item[2] for item in usable)
    if not usable or quality_sum <= 0:
        values = (None, None, None, None)
        status = "NO_USABLE_OBSERVATIONS"
    else:
        ski = sum(item[0] * item[2] for item in usable) / quality_sum
        snowboard = sum(item[1] * item[2] for item in usable) / quality_sum
        direction = snowboard - ski
        values = (ski, snowboard, direction, abs(direction))
        status = "AVAILABLE"
    return RawProviderSportEvidenceSummary(
        calibration_channel_id=f"{evidence_kind.value}::{provider_name}",
        provider_name=provider_name,
        evidence_kind=evidence_kind,
        source_video_id=source_video_id,
        video_sha256=video_sha256,
        observation_count=len(observations),
        distinct_timestamp_count=len({item[3] for item in usable if item[3] is not None}),
        raw_ski_support=values[0],
        raw_snowboard_support=values[1],
        raw_direction=values[2],
        raw_margin=values[3],
        source_status=status,
        limitations=("FRAME_OBSERVATIONS_ARE_CORRELATED",),
        provenance=provenance or {},
    )


def aggregate_subclips_by_source(
    summaries: list[RawProviderSportEvidenceSummary],
) -> list[RawProviderSportEvidenceSummary]:
    groups: dict[tuple[str, str], list[RawProviderSportEvidenceSummary]] = defaultdict(list)
    for item in summaries:
        groups[(item.source_video_id, item.calibration_channel_id)].append(item)
    result = []
    for key in sorted(groups):
        items = groups[key]
        first = items[0]
        sha_values = {item.video_sha256 for item in items}
        if len(sha_values) != 1:
            raise ValueError("subclips for one source must share video_sha256")
        available = [item for item in items if item.raw_direction is not None]
        if not available:
            result.append(first)
            continue
        directions = [item.raw_direction for item in available]
        ski = [item.raw_ski_support for item in available]
        snowboard = [item.raw_snowboard_support for item in available]
        direction = statistics.median(directions)
        result.append(
            RawProviderSportEvidenceSummary(
                calibration_channel_id=first.calibration_channel_id,
                provider_name=first.provider_name,
                evidence_kind=first.evidence_kind,
                source_video_id=first.source_video_id,
                video_sha256=first.video_sha256,
                observation_count=sum(item.observation_count for item in items),
                distinct_timestamp_count=sum(item.distinct_timestamp_count for item in items),
                raw_ski_support=statistics.median(ski),
                raw_snowboard_support=statistics.median(snowboard),
                raw_direction=direction,
                raw_margin=abs(direction),
                clip_count_per_source=len(items),
                limitations=(
                    "SUBCLIPS_AGGREGATED_TO_ONE_INDEPENDENT_SOURCE_SAMPLE",
                    "FRAME_OBSERVATIONS_ARE_CORRELATED",
                ),
                provenance=first.provenance,
            )
        )
    return result


def _ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a ratio")
    return float(value)
