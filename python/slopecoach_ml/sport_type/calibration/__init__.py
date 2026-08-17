"""A6.3 SportType calibration research/reference API."""

from .aggregation import aggregate_subclips_by_source, summarize_observations
from .artifact import artifact_fingerprint, fit_calibration_artifact
from .contracts import *  # noqa: F403
from .dataset import build_calibration_dataset, prepare_sport_type_gt
from .fusion import apply_calibrated_fusion, run_calibration_golden, unavailable_fusion
from .platt import (
    CalibrationSample,
    deterministic_fold_assignment,
    evaluate_channel,
    fit_scalar_logistic,
    logit,
    stable_sigmoid,
)

__all__ = [
    "CalibrationSample",
    "aggregate_subclips_by_source",
    "apply_calibrated_fusion",
    "artifact_fingerprint",
    "build_calibration_dataset",
    "deterministic_fold_assignment",
    "evaluate_channel",
    "fit_calibration_artifact",
    "fit_scalar_logistic",
    "logit",
    "prepare_sport_type_gt",
    "run_calibration_golden",
    "stable_sigmoid",
    "summarize_observations",
    "unavailable_fusion",
]
