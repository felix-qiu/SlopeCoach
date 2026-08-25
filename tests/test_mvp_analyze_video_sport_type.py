from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slopecoach_ml.cli import __main__ as cli
from slopecoach_ml.pose import FrameGeometry
from slopecoach_ml.product import select_user_sport_type
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


@pytest.mark.parametrize("sport_type", ["SKI", "SNOWBOARD"])
def test_analyze_video_parser_accepts_supported_product_sport_types(sport_type):
    args = cli.build_parser().parse_args(
        ["analyze-video", "clip.mp4", "--sport-type", sport_type]
    )
    assert args.sport_type == sport_type


@pytest.mark.parametrize("sport_type", ["AUTO", "UNKNOWN"])
def test_analyze_video_parser_rejects_non_product_sport_types(sport_type):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["analyze-video", "clip.mp4", "--sport-type", sport_type]
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


def test_product_sport_provenance_has_no_inferred_or_calibrated_fields():
    sport_type = select_user_sport_type("SNOWBOARD").to_dict()
    assert sport_type == {
        "contract_version": "mvp-user-sport-type-v1",
        "effective_sport_type": "SNOWBOARD",
        "effective_source": "USER",
        "resolution_status": "RESOLVED_USER",
    }
    assert "auto_decision" not in sport_type
    assert "provider_results" not in sport_type
    json.dumps(sport_type, allow_nan=False)


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
    counts = {
        "sampler": 0,
        "video_iteration": 0,
        "detector": 0,
        "pose": 0,
        "biomechanics_pass": 0,
    }
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

    def sampler(*args, **kwargs):
        counts["sampler"] += 1

        def frames():
            counts["video_iteration"] += 1
            yield frame

        return frames()

    def biomechanics_runner(**kwargs):
        counts["biomechanics_pass"] += 1
        for sampled in kwargs["frames"]:
            counts["detector"] += 1
            detections = kwargs["detector"].detect(sampled.image, sampled.geometry)
            counts["pose"] += 1
            kwargs["pose_provider"].estimate_detections(
                sampled.image,
                detections,
                sampled.geometry,
                timestamp_us=sampled.timestamp_us,
                frame_index=sampled.frame_index,
            )
        assert kwargs["warmup_frames"] == 0
        return {
            "benchmark_contract_version": "ski-bench-biomechanics-v2",
            "feature_schema_version": "biomechanics-feature-schema-v1",
            "feature_registry_sha256": (
                "2777c3fbf7513e7537122f897f1901e61baf7eeddcee927937decb7476953048"
            ),
            "video": {
                "duration_seconds": 1.0,
                "width_px": 640,
                "height_px": 480,
            },
            "models": {"detector": {"model_id": "det"}, "pose": {"model_id": "pose"}},
            "runtime": {"device": "cpu", "warmup_frames": 0},
            "performance": {"total_seconds": 0.1},
            "identity_input": {
                "identity_locked_frame_count": 1,
                "identity_unsafe_frame_count": 0,
            },
            "frame_biomechanics": {"trusted_frame_count": 1},
            "turn_segments": [],
            "biomechanics_result": {
                "contract_version": "temporal-biomechanics-v2",
                "frame_facts": [],
                "temporal_segment_features": [],
                "turn_features": [],
                "feature_coverage": {},
                "limitations": ["IMAGE_SPACE_2D_ONLY_NOT_PHYSICAL_3D"],
            },
        }

    monkeypatch.setattr(cli, "OpenCVVideoSampler", sampler)
    monkeypatch.setattr(
        cli,
        "benchmark_biomechanics_frames",
        biomechanics_runner,
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
    assert counts == {
        "sampler": 1,
        "video_iteration": 1,
        "detector": 1,
        "pose": 1,
        "biomechanics_pass": 1,
    }
