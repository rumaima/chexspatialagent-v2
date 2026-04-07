# utils/question_router.py — maps clinical question templates to tool chains
"""
Parses the spatial question templates defined in the paper and maps each
to a QuestionType + extracted entities (finding, reference, device, zone, border).
Used by the Planner to generate deterministic tool invocation plans.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Optional


class QuestionType(str, Enum):
    PRESENCE          = "presence"
    LOCATION          = "location"
    CONTAINMENT       = "containment"
    LATERALITY        = "laterality"
    DISTRIBUTION      = "distribution"
    EXTENT_LOCATION   = "extent_location"
    EXTENT_LATERALITY = "extent_laterality"
    ZONE_PRESENCE     = "zone_presence"
    ZONE_DEVICE       = "zone_device"
    RELATIVE_POSITION = "relative_position"
    DEVICE_TIP        = "device_tip"
    BORDER_SILHOUETTE = "border_silhouette"
    LYMPH_NODE        = "lymph_node"
    APPEARANCE        = "appearance"


@dataclass
class ParsedQuestion:
    q_type: QuestionType
    finding: Optional[str] = None
    reference: Optional[str] = None
    device: Optional[str] = None
    zone: Optional[str] = None
    border: Optional[str] = None
    raw: str = ""


_PATTERNS: list[tuple[re.Pattern, QuestionType]] = [
    (re.compile(r"^is (?P<finding>.+?) present in the chest", re.I),           QuestionType.PRESENCE),
    (re.compile(r"^where is the (?P<finding>.+?) located", re.I),               QuestionType.LOCATION),
    (re.compile(r"^is the (?P<finding>.+?) contained entirely within", re.I),   QuestionType.CONTAINMENT),
    (re.compile(r"^is the (?P<finding>.+?) more extensive in the right lung or in the left lung", re.I), QuestionType.LATERALITY),
    (re.compile(r"^what is the distribution pattern of the (?P<finding>.+?):", re.I), QuestionType.DISTRIBUTION),
    (re.compile(r"^what is the laterality of the (?P<finding>.+?):", re.I),     QuestionType.LATERALITY),
    (re.compile(r"^what is the extent and location of the (?P<finding>.+?):", re.I), QuestionType.EXTENT_LOCATION),
    (re.compile(r"^what is the extent and laterality of the (?P<finding>.+?):", re.I), QuestionType.EXTENT_LATERALITY),
    (re.compile(r"^is the (?P<device>.+?) present in the (?P<zone>(?:right |left )?(?:upper|middle|lower) zone) of the lung", re.I), QuestionType.ZONE_DEVICE),
    (re.compile(r"^is the (?P<finding>.+?) located predominantly in the (?P<zone>(?:right |left )?(?:upper|middle|lower) zone) of the lung", re.I), QuestionType.ZONE_PRESENCE),
    (re.compile(r"^what is the relative position of (?P<finding>.+?) with respect to (?:the )?(?P<reference>.+?)\??$", re.I), QuestionType.RELATIVE_POSITION),
    (re.compile(r"^where is the tip of the (?P<device>.+?) relative to (?:the )?(?P<reference>.+?)\??$", re.I), QuestionType.DEVICE_TIP),
    (re.compile(r"^does the (?P<finding>.+?) involve or obscure (?:the )?(?P<border>.+?)\??$", re.I), QuestionType.BORDER_SILHOUETTE),
    (re.compile(r"^where are the lymph nodes located predominantly", re.I),     QuestionType.LYMPH_NODE),
    (re.compile(r"^where is the (?P<finding>.+?) of the (?P<reference>.+?) located", re.I), QuestionType.LOCATION),
    (re.compile(r"^what is the appearance of the (?P<finding>.+?)\??$", re.I),  QuestionType.APPEARANCE),
]


def parse_question(question: str) -> ParsedQuestion:
    """Parse a clinical question string into a ParsedQuestion."""
    q = question.strip()
    for pattern, q_type in _PATTERNS:
        m = pattern.match(q)
        if m:
            gd = m.groupdict()
            return ParsedQuestion(
                q_type=q_type,
                finding=gd.get("finding"),
                reference=gd.get("reference"),
                device=gd.get("device"),
                zone=gd.get("zone"),
                border=gd.get("border"),
                raw=q,
            )
    return ParsedQuestion(q_type=QuestionType.PRESENCE, finding=q, raw=q)


def generate_questions(
    findings_all: set[str] | None = None,
    devices: set[str] | None = None,
    f1: set[str] | None = None,
    f2: set[str] | None = None,
    f3: set[str] | None = None,
    f4: set[str] | None = None,
    f5: set[str] | None = None,
    refs: set[str] | None = None,
    ref: set[str] | None = None,
    borders: set[str] | None = None,
    lzones: set[str] | None = None,
    loc_rel: bool = True,
    zones: bool = True,
) -> set[str]:
    """
    Generate the full set of spatial questions from entity sets.
    Implements the question-generation logic from the paper exactly.
    """
    # Defaults drawn from the CXR ontology
    findings_all = findings_all or {
        "consolidation", "pleural effusion", "atelectasis", "pneumothorax",
        "cardiomegaly", "pulmonary edema", "mass", "nodule",
    }
    devices  = devices  or {"endotracheal tube", "nasogastric tube", "central venous catheter"}
    f1 = f1 or {"consolidation", "opacity", "pulmonary edema", "atelectasis"}
    f2 = f2 or {"pleural effusion", "pneumothorax"}
    f3 = f3 or {"mass", "nodule", "fracture", "calcification"}
    f4 = f4 or {"cardiomegaly", "mediastinal widening"}
    f5 = f5 or {"fracture", "deformity", "scarring", "calcification"}
    refs    = refs   or {"carina", "hilum", "diaphragm", "aortic arch"}
    ref     = ref    or refs
    borders = borders or {"left heart border", "right heart border",
                          "left hemidiaphragm", "right hemidiaphragm"}
    lzones  = lzones or {"upper zone", "middle zone", "lower zone"}

    questions: set[str] = set()

    for a in findings_all | devices:
        questions.add(f"Is {a} present in the chest X-ray?")
    for a in findings_all:
        questions.add(f"Where is the {a} located?")
        questions.add(f"Is the {a} contained entirely within the lung field without extending beyond the pleural surface?")
        questions.add(f"Is the {a} more extensive in the right lung or in the left lung?")
    for a in f1:
        questions.add(f"What is the distribution pattern of the {a}: localized, focal, diffuse, unilateral, bilateral, central, peripheral, or multilobular?")
    for a in f2:
        questions.add(f"What is the laterality of the {a}: unilateral or bilateral?")
    for a in f3:
        questions.add(f"What is the extent and location of the {a}: localized, focal, diffuse, central, or peripheral?")
    for a in f4:
        questions.add(f"What is the extent and laterality of the {a}: localized, focal, diffuse, unilateral, or bilateral?")
    if loc_rel:
        for a in devices:
            for lz in lzones:
                questions.add(f"Is the {a} present in the {lz} of the lung?")
    if zones:
        for a in findings_all:
            for lz in lzones:
                questions.add(f"Is the {a} located predominantly in the {lz} of the lung?")
    if refs:
        for r1, r2 in combinations(refs, 2):
            questions.add(f"What is the relative position of {r1} with respect to the {r2}?")
        for a in findings_all | devices:
            for r in refs:
                questions.add(f"What is the relative position of {a} with respect to the {r}?")
    if devices:
        for dev in devices:
            for b in list(findings_all | refs | borders):
                questions.add(f"Where is the tip of the {dev} relative to the {b}?")
    if "lymph node" in ref:
        questions.add("Where are the lymph nodes located predominantly?")
    if borders:
        for a in findings_all:
            for b in borders:
                questions.add(f"Does the {a} involve or obscure the {b}?")

    STRUCTURAL = {"fracture", "deformity", "shadow", "scarring", "calcification"}
    structural_findings = (f3 | f5) & STRUCTURAL
    if structural_findings and ref:
        for a in structural_findings:
            for r in ref:
                questions.add(f"Where is the {a} of the {r} located?")

    if not findings_all and not devices and ref:
        for r in ref:
            questions.add(f"What is the appearance of the {r}?")

    return questions
