"""Small deterministic A8 research practice library."""

from __future__ import annotations

import hashlib
import json

from .contracts import DRILL_LIBRARY_VERSION, ControlledDrill

_SAFETY = (
    "PRACTICE_WITHIN_CURRENT_ABILITY",
    "KEEP_CLEAR_SPACE_AROUND",
    "STOP_IF_PAIN_OR_LOSS_OF_CONTROL",
)

DRILL_LIBRARY = (
    ControlledDrill(
        "FLEXION_RANGE_AWARENESS_TURNS",
        ("SKI", "SNOWBOARD"),
        ("LIMITED_KNEE_FLEXION_MODULATION_2D",),
        "屈伸幅度感知转弯",
        "在低速、可控的连续转弯中，关注双膝屈伸变化是否存在明显但平滑的幅度变化。",
        ("选择容易且可控的地形。", "保持轻松节奏完成连续转弯。"),
        ("留意双膝角度变化是否明显且平滑。",),
        _SAFETY,
        ("RESEARCH_PRACTICE_FOCUS_ONLY", "NOT_PHYSICAL_STIFFNESS_CORRECTION"),
    ),
    ControlledDrill(
        "LEFT_RIGHT_KNEE_SYMMETRY_AWARENESS",
        ("SKI", "SNOWBOARD"),
        ("BILATERAL_KNEE_ASYMMETRY_2D",),
        "左右动作一致性感知",
        "在轻松、可控的连续转弯中，观察左右方向的膝部动作是否表现出明显不同。",
        ("选择容易且可控的地形。", "以相近节奏完成左右方向的连续转弯。"),
        ("留意左右方向的膝部动作观感是否明显不同。",),
        _SAFETY,
        ("RESEARCH_PRACTICE_FOCUS_ONLY", "NOT_PRESSURE_OR_LOAD_SYMMETRY"),
    ),
    ControlledDrill(
        "TURN_RHYTHM_FLEXION_TIMING_AWARENESS",
        ("SKI", "SNOWBOARD"),
        ("KNEE_FLEXION_TIMING_OFFSET_2D",),
        "转弯节奏与屈伸时机感知",
        "保持均匀转弯节奏，关注膝部屈伸变化与转弯中段的相对时机。",
        ("选择容易且可控的地形。", "保持均匀节奏并观察每次转弯中段。"),
        ("留意膝部屈伸变化相对转弯中段的时机。",),
        _SAFETY,
        ("RESEARCH_PRACTICE_FOCUS_ONLY", "NOT_EDGE_OR_PRESSURE_TIMING"),
    ),
)


def canonical_drill_library_json(library=DRILL_LIBRARY) -> str:
    payload = {
        "library_version": DRILL_LIBRARY_VERSION,
        "drills": [item.to_dict() for item in library],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


DRILL_LIBRARY_SHA256 = hashlib.sha256(canonical_drill_library_json().encode()).hexdigest()


def drill_for_diagnosis(code: str, sport_type: str) -> ControlledDrill:
    drill = next((item for item in DRILL_LIBRARY if code in item.mapped_diagnosis_codes), None)
    if drill is None:
        raise ValueError(f"no controlled drill for diagnosis: {code}")
    if sport_type not in drill.applicable_sport_types:
        raise ValueError(f"drill {drill.drill_id} is not applicable to {sport_type}")
    return drill
