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


@dataclass(frozen=True)
class CoachLanguagePolicy:
    issue_templates: tuple[CoachTemplate, ...]
    issues_headline_template: str
    no_qualified_turns_headline: str
    not_analyzable_headline: str
    no_trigger_headline: str
    evidence_template: str
    controlled_warnings: tuple[str, ...]
    language: str = "zh-CN"
    template_version: str = COACH_TEMPLATE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "template_version": self.template_version,
            "language": self.language,
            "issue_templates": [item.to_dict() for item in self.issue_templates],
            "headline_templates": {
                "ISSUES_HEADLINE_TEMPLATE": self.issues_headline_template,
                "NO_QUALIFIED_TURNS_HEADLINE": self.no_qualified_turns_headline,
                "NOT_ANALYZABLE_HEADLINE": self.not_analyzable_headline,
                "NO_TRIGGER_HEADLINE": self.no_trigger_headline,
            },
            "evidence_templates": {"ISSUE_EVIDENCE_TEMPLATE": self.evidence_template},
            "controlled_warnings": list(self.controlled_warnings),
        }


LANGUAGE_POLICY = CoachLanguagePolicy(
    issue_templates=TEMPLATE_REGISTRY,
    issues_headline_template="这段视频有 {count} 个可以优先关注的动作信号",
    no_qualified_turns_headline="当前没有足够的完整转弯证据生成动作建议",
    not_analyzable_headline="当前证据不足，暂时无法生成动作建议",
    no_trigger_headline=("当前已实现且可评估的研究规则没有触发；这不代表完整技术表现已被验证。"),
    evidence_template="触发 {triggered_turn_count} / {evaluable_turn_count} 个可评估转弯。",
    controlled_warnings=("COACHING_IS_PROVISIONAL_RESEARCH_PRACTICE_GUIDANCE",),
)


def canonical_template_registry_json(policy: CoachLanguagePolicy = LANGUAGE_POLICY) -> str:
    return json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


COACH_TEMPLATE_REGISTRY_SHA256 = hashlib.sha256(
    canonical_template_registry_json().encode()
).hexdigest()


def render_issue_template(
    issue: ProvisionalIssueSummary, policy: CoachLanguagePolicy = LANGUAGE_POLICY
) -> dict[str, object]:
    template = next(
        item for item in policy.issue_templates if item.diagnosis_code == issue.diagnosis_code
    )
    return {
        "template_id": template.template_id,
        "title": template.title,
        "explanation": template.explanation,
        "evidence": policy.evidence_template.format(
            triggered_turn_count=issue.triggered_turn_count,
            evaluable_turn_count=issue.evaluable_turn_count,
        ),
        "limitation": template.limitation,
    }


def render_headline(*, status: str, issue_count: int, upstream_status: str) -> str:
    if status == "NOT_ANALYZABLE_UPSTREAM":
        if upstream_status == "NOT_ANALYZABLE_NO_QUALIFIED_TURNS":
            return LANGUAGE_POLICY.no_qualified_turns_headline
        return LANGUAGE_POLICY.not_analyzable_headline
    if issue_count:
        return LANGUAGE_POLICY.issues_headline_template.format(count=issue_count)
    return LANGUAGE_POLICY.no_trigger_headline
