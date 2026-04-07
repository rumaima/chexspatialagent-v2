# agents/summarizer.py — Φsum: synthesises tool outputs into a clinical report
# Uses Qwen2.5-VL-7B-Instruct locally — no API key required.

import logging

import config
from tools.base import ToolResult
from utils.model_loader import qwen_ask

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a senior radiologist providing a structured CXR interpretation.
Given automated tool outputs and the original clinical question, produce a clear, \
concise report.

Structure your response as:

## Technical quality
One sentence on image quality and whether findings can be relied upon.

## Key findings
Numbered list of significant findings with spatial locations.

## Assessment
Most likely diagnosis with confidence (high / moderate / low).

## Answer to clinical question
Direct, specific answer to exactly what was asked.

Use standard radiological terminology. Be concise and clinically actionable."""

_USER_TMPL = """\
Clinical question: {question}

How the planner broke down this question:
{deconstruction}

Planner strategy:
{reasoning}

Tool outputs:
{findings}

Synthesise these results into a final clinical report."""


class Summarizer:
    """
    Φsum — synthesises Y + (q, v, plan context) → final clinical response.
    Uses Qwen2.5-VL-7B (local, ~5 GB at 4-bit). Falls back to CheXagent-2.
    """

    def summarize(
        self,
        results: list[ToolResult],
        question: str,
        image_path: str | None = None,
        plan=None,
    ) -> str:
        findings = "\n\n---\n\n".join(r.to_findings_block() for r in results)

        deconstruction = getattr(plan, "deconstruction", "") if plan else ""
        reasoning      = getattr(plan, "reasoning", "")      if plan else ""

        user_text = _USER_TMPL.format(
            question       = question,
            deconstruction = deconstruction or "Not available.",
            reasoning      = reasoning      or "Not available.",
            findings       = findings,
        )

        logger.info("[Summarizer] Generating report from %d tool results", len(results))

        # Primary: Qwen2.5-VL-7B
        try:
            return qwen_ask(
                system_prompt  = _SYSTEM,
                user_prompt    = user_text,
                image_path     = image_path,
                max_new_tokens = 800,
                model_name     = config.QWEN_MODEL,
                load_in_4bit   = config.QWEN_LOAD_IN_4BIT,
            )
        except Exception as e:
            logger.warning("[Summarizer] Qwen failed (%s); falling back to CheXagent", e)

        # Fallback: CheXagent-2
        try:
            from utils.model_loader import chexagent_ask
            combined = f"{_SYSTEM}\n\n{user_text}"
            return chexagent_ask(combined, image_path=image_path, max_new_tokens=600)
        except Exception as e2:
            logger.error("[Summarizer] CheXagent fallback also failed: %s", e2)
            parts = [r.output.get("raw_output", str(r.output))
                     for r in results if r.success]
            return " ".join(parts) or "Summarizer unavailable — see tool outputs above."
