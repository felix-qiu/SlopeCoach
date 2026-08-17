"""Dependency-free deterministic scalar logistic calibration and grouped OOF evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .contracts import CalibrationChannelStatus, SportCalibrationFitConfig

SPLIT_SALT = "slopecoach-a6.3-grouped-stratified-v1"


@dataclass(frozen=True)
class CalibrationSample:
    source_video_id: str
    raw_direction: float
    snowboard_label: int


@dataclass(frozen=True)
class LogisticFit:
    slope_a: float | None
    intercept_b: float | None
    converged: bool
    iterations: int
    status: str


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1 / (1 + exp)
    exp = math.exp(value)
    return exp / (1 + exp)


def logit(probability: float, epsilon: float = 1e-6) -> float:
    value = min(1 - epsilon, max(epsilon, probability))
    return math.log(value / (1 - value))


def fit_scalar_logistic(
    samples: list[CalibrationSample], config: SportCalibrationFitConfig
) -> LogisticFit:
    if not samples or len({item.snowboard_label for item in samples}) < 2:
        return LogisticFit(None, None, False, 0, "INSUFFICIENT_CLASSES")
    prior = sum(item.snowboard_label for item in samples) / len(samples)
    a = 0.0
    b = logit(prior, config.probability_epsilon)
    for iteration in range(1, config.maximum_newton_iterations + 1):
        ga = config.l2_regularization * a
        gb = 0.0
        haa = config.l2_regularization
        hab = 0.0
        hbb = 0.0
        for item in samples:
            p = stable_sigmoid(a * item.raw_direction + b)
            residual = p - item.snowboard_label
            weight = p * (1 - p)
            ga += residual * item.raw_direction
            gb += residual
            haa += weight * item.raw_direction * item.raw_direction
            hab += weight * item.raw_direction
            hbb += weight
        determinant = haa * hbb - hab * hab
        if not math.isfinite(determinant) or determinant <= 1e-18:
            return LogisticFit(None, None, False, iteration, "SINGULAR_HESSIAN")
        step_a = (hbb * ga - hab * gb) / determinant
        step_b = (-hab * ga + haa * gb) / determinant
        a -= step_a
        b -= step_b
        if not math.isfinite(a) or not math.isfinite(b):
            return LogisticFit(None, None, False, iteration, "NON_FINITE_COEFFICIENT")
        if max(abs(step_a), abs(step_b)) <= config.convergence_tolerance:
            return LogisticFit(a, b, True, iteration, "CONVERGED")
    return LogisticFit(a, b, False, config.maximum_newton_iterations, "MAX_ITERATIONS")


def deterministic_fold_assignment(
    samples: list[CalibrationSample], dataset_id: str, fold_count: int
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for label in (0, 1):
        source_ids = sorted(
            {item.source_video_id for item in samples if item.snowboard_label == label}
        )
        ordered = sorted(
            source_ids,
            key=lambda source_id: hashlib.sha256(
                f"{dataset_id}\0{source_id}\0{SPLIT_SALT}".encode()
            ).hexdigest(),
        )
        for index, source_id in enumerate(ordered):
            assignments[source_id] = index % fold_count
    return assignments


def evaluate_channel(
    samples: list[CalibrationSample], dataset_id: str, config: SportCalibrationFitConfig
) -> dict[str, object]:
    unique = _one_sample_per_source(samples)
    ski_count = sum(item.snowboard_label == 0 for item in unique)
    snowboard_count = sum(item.snowboard_label == 1 for item in unique)
    effective_folds = min(config.cross_validation_folds, ski_count, snowboard_count)
    base = {
        "sample_count": len(unique),
        "ski_count": ski_count,
        "snowboard_count": snowboard_count,
        "effective_cv_folds": effective_folds,
        "fold_assignment": {},
        "brier_score": None,
        "log_loss": None,
        "prior_only_brier_score": None,
        "prior_only_log_loss": None,
        "brier_skill_vs_prior": None,
        "log_loss_improvement_vs_prior": None,
        "classification_accuracy_at_0_5": None,
        "ece_5_bin": None,
    }
    if (
        ski_count < config.minimum_labeled_sources_per_class
        or snowboard_count < config.minimum_labeled_sources_per_class
        or effective_folds < 3
    ):
        return {**base, "status": CalibrationChannelStatus.REJECTED_INSUFFICIENT_DATA.value}
    assignments = deterministic_fold_assignment(unique, dataset_id, effective_folds)
    predictions: list[tuple[int, float, float]] = []
    for fold in range(effective_folds):
        train = [item for item in unique if assignments[item.source_video_id] != fold]
        test = [item for item in unique if assignments[item.source_video_id] == fold]
        fitted = fit_scalar_logistic(train, config)
        if not fitted.converged or fitted.slope_a is None or fitted.intercept_b is None:
            return {
                **base,
                "fold_assignment": assignments,
                "status": CalibrationChannelStatus.REJECTED_FIT_FAILURE.value,
            }
        prior = sum(item.snowboard_label for item in train) / len(train)
        predictions.extend(
            (
                item.snowboard_label,
                stable_sigmoid(fitted.slope_a * item.raw_direction + fitted.intercept_b),
                prior,
            )
            for item in test
        )
    metrics = _metrics(predictions, config.probability_epsilon)
    final = fit_scalar_logistic(unique, config)
    if not final.converged or final.slope_a is None:
        status = CalibrationChannelStatus.REJECTED_FIT_FAILURE
    elif final.slope_a <= 0:
        status = CalibrationChannelStatus.REJECTED_NON_MONOTONIC_CHANNEL
    elif metrics["brier_score"] >= metrics["prior_only_brier_score"]:
        status = CalibrationChannelStatus.REJECTED_NO_BRIER_IMPROVEMENT
    elif metrics["log_loss"] >= metrics["prior_only_log_loss"]:
        status = CalibrationChannelStatus.REJECTED_NO_LOG_LOSS_IMPROVEMENT
    else:
        status = CalibrationChannelStatus.ACCEPTED_RESEARCH_CALIBRATION
    return {
        **base,
        **metrics,
        "fold_assignment": assignments,
        "status": status.value,
        "final_fit": {
            "slope_a": final.slope_a,
            "intercept_b": final.intercept_b,
            "converged": final.converged,
            "iterations": final.iterations,
        },
    }


def _one_sample_per_source(samples: list[CalibrationSample]) -> list[CalibrationSample]:
    grouped: dict[str, list[CalibrationSample]] = {}
    for item in samples:
        grouped.setdefault(item.source_video_id, []).append(item)
    result = []
    for source_id in sorted(grouped):
        items = grouped[source_id]
        labels = {item.snowboard_label for item in items}
        if len(labels) != 1:
            raise ValueError("one source cannot have conflicting SportType labels")
        directions = sorted(item.raw_direction for item in items)
        midpoint = len(directions) // 2
        median = (
            directions[midpoint]
            if len(directions) % 2
            else (directions[midpoint - 1] + directions[midpoint]) / 2
        )
        result.append(CalibrationSample(source_id, median, items[0].snowboard_label))
    return result


def _metrics(predictions: list[tuple[int, float, float]], epsilon: float) -> dict[str, object]:
    count = len(predictions)
    brier = sum((prediction - label) ** 2 for label, prediction, _ in predictions) / count
    prior_brier = sum((prior - label) ** 2 for label, _, prior in predictions) / count

    def loss(label: int, probability: float) -> float:
        p = min(1 - epsilon, max(epsilon, probability))
        return -(label * math.log(p) + (1 - label) * math.log(1 - p))

    log_loss = sum(loss(label, prediction) for label, prediction, _ in predictions) / count
    prior_loss = sum(loss(label, prior) for label, _, prior in predictions) / count
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "prior_only_brier_score": prior_brier,
        "prior_only_log_loss": prior_loss,
        "brier_skill_vs_prior": 1 - brier / prior_brier if prior_brier else None,
        "log_loss_improvement_vs_prior": prior_loss - log_loss,
        "classification_accuracy_at_0_5": sum(
            (prediction >= 0.5) == bool(label) for label, prediction, _ in predictions
        )
        / count,
        "ece_5_bin": _ece(predictions) if count >= 50 else None,
    }


def _ece(predictions: list[tuple[int, float, float]]) -> float:
    total = len(predictions)
    value = 0.0
    for index in range(5):
        lower, upper = index / 5, (index + 1) / 5
        bucket = [
            item for item in predictions if lower <= item[1] < upper or index == 4 and item[1] == 1
        ]
        if bucket:
            value += (
                len(bucket)
                / total
                * abs(
                    sum(item[1] for item in bucket) / len(bucket)
                    - sum(item[0] for item in bucket) / len(bucket)
                )
            )
    return value
