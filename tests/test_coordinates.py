from __future__ import annotations

import pytest

from slopecoach_ml.pose import FrameGeometry, Keypoint2D, PreprocessTransform2D


SOURCE = FrameGeometry(1920, 1080)


def transform(**overrides: float) -> PreprocessTransform2D:
    values = {
        "crop_x_px": 0.0,
        "crop_y_px": 0.0,
        "crop_width_px": 1920.0,
        "crop_height_px": 1080.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "pad_left_px": 0.0,
        "pad_top_px": 0.0,
        "model_width_px": 1920.0,
        "model_height_px": 1080.0,
    }
    values.update(overrides)
    return PreprocessTransform2D(**values)


@pytest.mark.parametrize(
    "adapter,point",
    [
        (
            transform(
                scale_x=0.5, scale_y=0.5, model_width_px=960, model_height_px=540
            ),
            Keypoint2D(0, 0, 1),
        ),
        (
            transform(
                crop_width_px=1920,
                crop_height_px=1080,
                scale_x=1 / 3,
                scale_y=1 / 3,
                pad_top_px=140,
                model_width_px=640,
                model_height_px=640,
            ),
            Keypoint2D(1919, 1079, 0.8),
        ),
        (
            transform(
                crop_x_px=200,
                crop_y_px=100,
                crop_width_px=1000,
                crop_height_px=800,
                scale_x=0.5,
                scale_y=0.5,
                pad_left_px=20,
                pad_top_px=30,
                model_width_px=540,
                model_height_px=460,
            ),
            Keypoint2D(777.25, 456.75, 0.91),
        ),
    ],
)
def test_synthetic_forward_inverse_round_trip(adapter, point) -> None:
    recovered = adapter.inverse(adapter.forward(point, SOURCE), SOURCE)
    assert recovered.x_px == pytest.approx(point.x_px, abs=1e-9)
    assert recovered.y_px == pytest.approx(point.y_px, abs=1e-9)
    assert recovered.confidence == point.confidence
    assert recovered.coordinate_space.value == "SourcePixel2D"


@pytest.mark.parametrize(
    "adapter",
    [
        transform(scale_x=0),
        transform(crop_x_px=-1),
        transform(crop_width_px=2000),
        transform(pad_left_px=-1),
        transform(pad_left_px=100, model_width_px=1920),
    ],
)
def test_invalid_or_impossible_transform_rejected(adapter) -> None:
    with pytest.raises(ValueError):
        adapter.validate(SOURCE)
