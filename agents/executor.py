# agents/executor.py — Φexec: executes a plan step-by-step, collecting Y
"""
The executor carries out exactly the plan the planner produced — no
interpretation, no re-routing.  It passes each step's output forward
as context so every subsequent tool call can see what prior steps found.

Context passing:
  - accumulated_findings: growing dict of {tool_id: output} from all prior steps
  - This is injected into the args of every subsequent step so tools can
    condition on prior results (e.g. the segmenter knows what the classifier found)
"""

import json
import logging
from typing import Callable

import config
from agents.planner import Plan, PlanStep
from tools.base import ToolResult
from tools.registry import get_tool
from utils.json_utils import safe_parse_json
from utils.model_loader import chexagent_ask

logger = logging.getLogger(__name__)

StepCallback = Callable[[int, PlanStep, ToolResult | None], None]


_SIM_PROMPT = """\
You are executing the tool "{tool_name}" (id: {tool_id}) as part of a \
CXR spatial reasoning pipeline.

Step purpose : {purpose}
Reason this tool was chosen: {reasoning}
Args         : {args_json}
Expected output format: {output_format}
Original clinical question: {question}

Results from prior steps (use these to inform your output):
{prior_findings}

The chest X-ray image is attached.
Return ONLY valid JSON matching the output format exactly.
Be clinically specific and realistic. No markdown, no explanation."""


class Executor:
    """
    Φexec — executes the planner's tool chain sequentially.

    Two modes:
      "real"      — each tool calls its actual DL model
      "simulated" — CheXagent-2 acts as every tool (no weights needed)

    In both modes, prior tool outputs are accumulated and passed forward
    so each tool can condition on what earlier steps found.
    """

    def __init__(self, mode: str = "real"):
        assert mode in ("simulated", "real"), "mode must be 'simulated' or 'real'"
        self.mode = mode

    def execute(
        self,
        plan: Plan,
        question: str,
        image_path: str | None = None,
        on_step: StepCallback | None = None,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        accumulated_findings: dict = {}   # grows after each step

        for i, step in enumerate(plan):
            logger.info("[Executor:%s] Step %d/%d — %s  |  why: %s",
                        self.mode, i + 1, len(plan), step.tool_id, step.reasoning[:80])

            if on_step:
                on_step(i, step, None)   # signal step start

            # Inject accumulated prior results into step args
            enriched_args = {
                **step.args,
                "_prior_findings": accumulated_findings,
            }

            if self.mode == "real":
                result = self._execute_real(step, enriched_args, image_path)
            else:
                result = self._execute_simulated(
                    step, enriched_args, question,
                    accumulated_findings, image_path
                )

            results.append(result)

            # Accumulate output for next steps
            if result.success and result.output:
                accumulated_findings[step.tool_id] = result.output

            if on_step:
                on_step(i, step, result)  # signal step complete

            if config.LOG_TOOL_OUTPUTS:
                logger.debug("[Executor] Step %d output: %s",
                             step.step, json.dumps(result.output, indent=2)[:400])

        return results

    # ── Real execution ─────────────────────────────────────────────────────────

    def _execute_real(self, step: PlanStep,
                      enriched_args: dict,
                      image_path: str | None) -> ToolResult:
        try:
            tool = get_tool(step.tool_id)
        except KeyError as e:
            logger.error("[Executor] Tool not found: %s", step.tool_id)
            return ToolResult(tool_id=step.tool_id, tool_name=step.tool_name,
                              step=step.step, success=False, error=str(e))
        return tool.run(image_path, enriched_args, step.step)

    # ── Simulated execution ────────────────────────────────────────────────────

    def _execute_simulated(
        self,
        step: PlanStep,
        enriched_args: dict,
        question: str,
        accumulated_findings: dict,
        image_path: str | None,
    ) -> ToolResult:
        try:
            tool = get_tool(step.tool_id)
            output_format = tool.spec.output_format
        except KeyError:
            output_format = "JSON object with relevant clinical findings"

        # Summarise prior findings compactly for the prompt
        prior_text = (
            json.dumps(accumulated_findings, indent=2)[:800]
            if accumulated_findings
            else "None — this is the first tool call."
        )

        # Strip internal key before showing args
        display_args = {k: v for k, v in enriched_args.items()
                        if k != "_prior_findings"}

        prompt = _SIM_PROMPT.format(
            tool_name      = step.tool_name,
            tool_id        = step.tool_id,
            purpose        = step.purpose,
            reasoning      = step.reasoning,
            args_json      = json.dumps(display_args),
            output_format  = output_format,
            question       = question,
            prior_findings = prior_text,
        )

        try:
            raw    = chexagent_ask(prompt, image_path=image_path, max_new_tokens=500)
            output = safe_parse_json(raw) or {"raw_output": raw}
            return ToolResult(tool_id=step.tool_id, tool_name=step.tool_name,
                              step=step.step, success=True, output=output)
        except Exception as e:
            logger.warning("[Executor] CheXagent simulation failed for %s: %s",
                           step.tool_id, e)
            return ToolResult(tool_id=step.tool_id, tool_name=step.tool_name,
                              step=step.step, success=False, error=str(e))
