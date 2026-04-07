# agents/planner.py — Φplan: reasoning planner that deconstructs any question
"""
The planner receives the clinical question, the CXR image, and a full
description of every available tool.  It reasons from first principles:

  1. DECONSTRUCT  — what spatial/clinical concepts does the question require?
  2. MAP          — which tools can directly address each concept?
  3. SEQUENCE     — what order minimises wasted work and maximises accuracy?
  4. JUSTIFY      — give an explicit reason for every tool selected (and every
                    tool deliberately NOT selected).

This replaces the old rule-based routing table entirely.  The planner now
handles any question — spatial taxonomy questions, free-form clinical questions,
comparative questions, composite multi-part questions — without needing a
pre-written chain for each type.
"""

import json
import logging
from dataclasses import dataclass, field

import config
from tools.registry import toolbox_prompt_block
from utils.json_utils import safe_parse_json
from utils.model_loader import chexagent_ask

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step: int
    tool_id: str
    tool_name: str
    args: dict
    purpose: str          # what this step produces
    reasoning: str        # WHY this tool was chosen for this step


@dataclass
class Plan:
    question: str
    deconstruction: str   # planner's breakdown of what the question requires
    reasoning: str        # overall strategy
    steps: list[PlanStep]
    tools_excluded: list[dict] = field(default_factory=list)  # tools considered but skipped

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def print_plan(self):
        print(f"\n  Question    : {self.question}")
        print(f"  Breakdown   : {self.deconstruction}")
        print(f"  Strategy    : {self.reasoning}")
        print(f"\n  Tool plan ({len(self.steps)} steps):")
        for s in self.steps:
            print(f"    [{s.step}] {s.tool_name}")
            print(f"         args    : {json.dumps(s.args)}")
            print(f"         purpose : {s.purpose}")
            print(f"         why     : {s.reasoning}")
        if self.tools_excluded:
            print(f"\n  Tools NOT used:")
            for ex in self.tools_excluded:
                print(f"    - {ex.get('tool_name','?')}: {ex.get('reason','')}")


# ── Planner system prompt ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Planner Agent (Φplan) in a medical chest X-ray spatial reasoning system.

Your job is to receive a clinical question and produce a precise, reasoned tool \
invocation plan.  You must reason from FIRST PRINCIPLES — not from templates.

You have access to the following toolbox:
{toolbox}

━━━ YOUR REASONING PROCESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — DECONSTRUCT the question:
  • What anatomical structures or pathologies are involved?
  • What spatial concept is being asked about?
    (presence / location / laterality / distribution / extent / containment /
     relative position / device positioning / silhouette / appearance /
     comparison / quantification / or something else entirely)
  • What sub-questions must be answered to address the main question?
  • What information is needed as a prerequisite for each sub-question?

STEP 2 — MAP tools to sub-questions:
  • For each sub-question, identify every tool that can contribute
  • Consider what each tool outputs and whether that output is needed
  • Consider tool dependencies (e.g. segmentation needs detection first)

STEP 3 — SEQUENCE the plan:
  • Order tools so each step's prerequisites are met by prior steps
  • Eliminate redundant steps
  • Always start with image_quality_assessor
  • Always end with terminate
  • Aim for 3–8 steps; be lean but complete

STEP 4 — JUSTIFY every decision:
  • For each tool selected: explain exactly why it is needed
  • For each tool NOT selected: explain why it was considered and skipped

━━━ STRICT RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Never add a tool just because it sounds relevant — only add it if its output
  is directly required to answer the question
- args must be specific: name the exact finding, device, reference structure,
  zone, or border mentioned in the question — never use generic placeholders
- If the question involves a device (ETT, NGT, CVC, pacemaker, chest drain):
  tube_line_localizer is mandatory
- If the question involves cardiac size or heart failure signs:
  cardiomegaly_quantifier is mandatory
- If the question asks about pneumothorax size or pleural line:
  pleural_line_detector is mandatory
- If the question involves costophrenic angle blunting or CP angle effusion:
  costophrenic_angle_analyzer is mandatory
- If the question involves tracheal deviation or mediastinal shift:
  trachea_mediastinum_analyzer is mandatory
- If the question is comparative (e.g. "has this worsened") or entirely outside
  the toolbox capabilities: still produce the best possible plan and note the
  limitation in reasoning

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY valid JSON — no markdown, no preamble:
{
  "deconstruction": "breakdown of what the question requires (2-4 sentences)",
  "reasoning": "overall strategy — why these tools in this order (2-3 sentences)",
  "plan": [
    {
      "step": 1,
      "tool_id": "image_quality_assessor",
      "tool_name": "Image Quality Assessor",
      "args": {},
      "purpose": "what this step produces for downstream steps",
      "reasoning": "why this specific tool is required at this position"
    }
  ],
  "tools_excluded": [
    {
      "tool_id": "...",
      "tool_name": "...",
      "reason": "why this tool was considered but not included"
    }
  ]
}"""

_USER_TMPL = "Clinical question: {question}"


# ── Planner ────────────────────────────────────────────────────────────────────

class Planner:
    """
    Φplan — fully reasoning planner.

    Deconstructs any clinical question from first principles and composes
    a bespoke tool chain with explicit justification for every decision.
    Uses CheXagent-2-3b as the reasoning engine (local, no API key).
    """

    def plan(self, question: str, image_path: str | None = None) -> Plan:
        logger.info("[Planner] Deconstructing question: %r", question)

        system = _SYSTEM_PROMPT.format(toolbox=toolbox_prompt_block())
        user   = _USER_TMPL.format(question=question)

        # Combine system + user into a single prompt for CheXagent
        full_prompt = f"{system}\n\n{user}"

        try:
            raw    = chexagent_ask(full_prompt, image_path=image_path,
                                   max_new_tokens=900)
            parsed = safe_parse_json(raw)

            if not parsed or "plan" not in parsed:
                raise ValueError(f"Planner returned unparseable response: {raw[:300]}")

            steps = [
                PlanStep(
                    step      = s["step"],
                    tool_id   = s["tool_id"],
                    tool_name = s["tool_name"],
                    args      = s.get("args", {}),
                    purpose   = s.get("purpose", ""),
                    reasoning = s.get("reasoning", ""),
                )
                for s in parsed["plan"]
            ]

            excluded = parsed.get("tools_excluded", [])

            plan = Plan(
                question        = question,
                deconstruction  = parsed.get("deconstruction", ""),
                reasoning       = parsed.get("reasoning", ""),
                steps           = steps,
                tools_excluded  = excluded,
            )

            logger.info("[Planner] Plan generated: %d steps — %s",
                        len(steps), [s.tool_id for s in steps])
            logger.info("[Planner] Deconstruction: %s", plan.deconstruction)
            if excluded:
                logger.info("[Planner] Excluded tools: %s",
                            [e.get("tool_name") for e in excluded])
            return plan

        except Exception as e:
            logger.error("[Planner] CheXagent planning failed: %s", e)
            return self._emergency_fallback(question)

    def _emergency_fallback(self, question: str) -> Plan:
        """
        Last-resort plan when CheXagent is unavailable or fails.
        Runs the full general-purpose stack — catches most questions
        at the cost of running unnecessary tools.
        """
        logger.warning("[Planner] Using emergency fallback plan")
        return Plan(
            question       = question,
            deconstruction = "Could not deconstruct — using general-purpose fallback.",
            reasoning      = "Emergency fallback: runs full stack to maximise coverage.",
            steps=[
                PlanStep(1, "image_quality_assessor", "Image Quality Assessor",
                         {}, "Assess image quality",
                         "Always required — poor quality invalidates all downstream results"),
                PlanStep(2, "lung_region_detector", "Lung Region Detector",
                         {"finding": question}, "Segment lung fields",
                         "Provides spatial reference frame for all spatial tools"),
                PlanStep(3, "opacity_segmenter", "Opacity Segmenter",
                         {"finding": question}, "Detect and segment primary pathology",
                         "Covers the most common question types"),
                PlanStep(4, "airspace_density_mapper", "Airspace Density Mapper",
                         {}, "Map opacity across zones",
                         "Covers distribution and zone questions"),
                PlanStep(5, "differential_ranker", "Differential Ranker",
                         {}, "Rank differential diagnoses",
                         "Provides diagnostic context for the summarizer"),
                PlanStep(6, "terminate", "Terminate",
                         {}, "Signal completion", "Required to end plan"),
            ],
            tools_excluded=[],
        )
