"""Deterministic zh-CN coaching templates over controlled issue facts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .contracts import COACH_TEMPLATE_VERSION, ProvisionalIssueSummary


@dataclass(frozen=True)
class CoachTemplate:
    template_id: str
    diagnosis_code: str
    title: str
    explanation: str
    limitation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "diagnosis_code": self.diagnosis_code,
            "title": self.title,
            "explanation": self.explanation,
            "limitation": self.limitation,
        }


TEMPLATE_REGISTRY = (
    CoachTemplate(
        "LIMITED_MODULATION_ZH_CN_V1",
        "LIMITED_KNEE_FLEXION_MODULATION_2D",
        "可以先关注屈伸变化",
        "在当前可评估的转弯中，双膝平均角度的变化范围在部分转弯里触发了研究性 2D 阈值。",
        "这是基于相机画面的 2D 信号，不是物理刚度或受力测量。",
    ),
    CoachTemplate(
        "ASYMMETRY_ZH_CN_V1",
        "BILATERAL_KNEE_ASYMMETRY_2D",
        "可以先关注左右动作一致性",
        "在当前可评估的转弯中，左右膝角度差的研究性 2D 指标在部分转弯里触发。",
        "这不代表左右雪板受力或物理压力差。",
    ),
    CoachTemplate(
        "TIMING_ZH_CN_V1",
        "KNEE_FLEXION_TIMING_OFFSET_2D",
        "可以先关注屈伸变化的时机",
        (
            "在当前可评估的转弯中，最小双膝平均角度相对 provisional turn apex "
            "的位置出现了较大的研究性 2D 相位偏移。"
        ),
        "这不是刃角时机或压力时机测量。",
    ),
)


def canonical_template_registry_json(registry=TEMPLATE_REGISTRY) -> str:
    payload = {
        "template_version": COACH_TEMPLATE_VERSION,
        "templates": [item.to_dict() for item in registry],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


COACH_TEMPLATE_REGISTRY_SHA256 = hashlib.sha256(
    canonical_template_registry_json().encode()
).hexdigest()


def render_issue_template(issue: ProvisionalIssueSummary) -> dict[str, object]:
    template = next(
        item for item in TEMPLATE_REGISTRY if item.diagnosis_code == issue.diagnosis_code
    )
    return {
        "template_id": template.template_id,
        "title": template.title,
        "explanation": template.explanation,
        "evidence": (
            f"触发 {issue.triggered_turn_count} / {issue.evaluable_turn_count} 个可评估转弯。"
        ),
        "limitation": template.limitation,
    }
