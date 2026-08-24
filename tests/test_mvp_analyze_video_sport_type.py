from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slopecoach_ml.cli import __main__ as cli
from slopecoach_ml.pose import FrameGeometry
from slopecoach_ml.product import build_mvp_analysis_payload, select_user_sport_type
from slopecoach_ml.sport_type import SportType


@pytest.mark.parametrize(
    ("value", "expected"),
    [("SKI", "SKI"), ("snowboard", "SNOWBOARD"), (SportType.SKI, "SKI")],
)
def test_product_selection_produces_user_provenance(value, expected):
    result = select_user_sport_type(value).to_dict()
    assert result["effective_sport_type"] == expected
    assert result["effective_source"] == "USER"
    assert result["resolution_status"] == "RESOLVED_USER"


@pytest.mark.parametrize("value", [None, "AUTO", "UNKNOWN", SportType.UNKNOWN])
def test_product_selection_rejects_missing_auto_and_unknown(value):
    with pytest.raises(ValueError, match="MVP_SPORT_TYPE_REQUIRED"):
        select_user_sport_type(value)


def test_analyze_video_parser_requires_explicit_product_sport_type():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["analyze-video", "clip.mp4"])


def test_analyze_video_parser_rejects_auto():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["analyze-video", "clip.mp4", "--sport-type", "AUTO"]
        )


def test_a6_research_benchmark_still_exposes_auto_and_provider_controls():
    args = cli.build_parser().parse_args(
        [
            "benchmark-sport-type",
            "clip.mp4",
            "--sport-type",
            "auto",
            "--equipment-provider",
            "rtmdet-coco",
            "--visual-provider",
            "openai-clip",
        ]
    )
    assert args.sport_type == "auto"
    assert args.equipment_provider == "rtmdet-coco"
    assert args.visual_provider == "openai-clip"


def test_product_payload_has_no_inferred_or_calibrated_sport_fields():
    payload = build_mvp_analysis_payload(
        video="clip.mp4",
        sport_type=select_user_sport_type("SNOWBOARD"),
        biomechanics_report={"benchmark_contract_version": "test-biomechanics"},
    )
    assert payload["sport_type"] == {
        "contract_version": "mvp-user-sport-type-v1",
        "effective_sport_type": "SNOWBOARD",
        "effective_source": "USER",
        "resolution_status": "RESOLVED_USER",
    }
    assert payload["automatic_sport_type_research"] == {
        "status": "DEFERRED_RESEARCH_ONLY",
        "executed": False,
    }
    assert "auto_decision" not in payload["sport_type"]
    assert "provider_results" not in payload["sport_type"]
    json.dumps(payload, allow_nan=False)


class _Detector:
    name = "test-detector"

    def detect(self, image, geometry):
        return ()


class _PoseProvider:
    name = "test-pose"

    def estimate_detections(self, image, detections, geometry, **kwargs):
        return None


def _forbidden(*args, **kwargs):
    raise AssertionError("automatic SportType research code executed in product path")


def test_analyze_video_never_constructs_or_calls_automatic_sport_type(
    monkeypatch, tmp_path
):
    frame = SimpleNamespace(
        image=object(),
        geometry=FrameGeometry(640, 480),
        timestamp_us=0,
        frame_index=0,
    )
    monkeypatch.setattr(
        cli,
        "_real_providers",
        lambda: (_Detector(), _PoseProvider(), "cpu", {"detector_seconds": 0.0}),
    )
    monkeypatch.setattr(cli, "OpenCVVideoSampler", lambda *args, **kwargs: [frame])
    monkeypatch.setattr(
        cli,
        "benchmark_biomechanics_frames",
        lambda **kwargs: {"benchmark_contract_version": "test-biomechanics"},
    )
    for name in (
        "OpenMMLabEquipmentBackend",
        "MMDetEquipmentSportEvidenceProvider",
        "OpenAIClipVisualSportBackend",
        "ClipVisualSportEvidenceProvider",
        "apply_calibrated_fusion",
    ):
        monkeypatch.setattr(cli, name, _forbidden)
    monkeypatch.setattr(
        "slopecoach_ml.sport_type.pipeline.ReferenceSportTypeFusion", _forbidden
    )

    output = tmp_path / "analysis.json"
    exit_code = cli.main(
        [
            "analyze-video",
            "clip.mp4",
            "--sport-type",
            "SKI",
            "--input-non-mirrored",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sport_type"]["effective_sport_type"] == "SKI"
    assert payload["sport_type"]["effective_source"] == "USER"
    assert payload["sport_type"]["resolution_status"] == "RESOLVED_USER"
    assert payload["automatic_sport_type_research"]["executed"] is False
