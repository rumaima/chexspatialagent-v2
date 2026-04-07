# pipeline.py — CXR SpatialAgent: Plan-Execute-Summarize (fully open-source)
"""
Usage:
    python pipeline.py --image cxr.jpg --question "Is there pleural effusion?"
    python pipeline.py --image cxr.jpg --question "..." --mode simulated
    python pipeline.py --image cxr.jpg --question "..." --plan-only
    python pipeline.py --image cxr.jpg --question "..." --output result.json
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass

import config
from agents.executor import Executor
from agents.planner import Plan, Planner
from agents.summarizer import Summarizer
from tools.base import ToolResult

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


@dataclass
class AgentResult:
    question: str
    image_path: str | None
    plan: Plan
    tool_results: list[ToolResult]
    report: str


class SpatialAgent:
    """
    APE = {Φplan, Φexec, Φsum}

    Φplan  — deconstructs any question and builds a justified tool chain
    Φexec  — executes the plan exactly, passing prior outputs forward
    Φsum   — synthesises Y into a structured clinical report
    """

    def __init__(self, mode: str = "real"):
        self.planner    = Planner()
        self.executor   = Executor(mode=mode)
        self.summarizer = Summarizer()

    def run(
        self,
        question: str,
        image_path: str | None = None,
        verbose: bool = True,
    ) -> AgentResult:
        sep = "═" * 62

        if verbose:
            print(f"\n{sep}")
            print("  CXR SpatialAgent — Plan · Execute · Summarize")
            print(sep)
            print(f"  Question : {question}")
            print(f"  Image    : {image_path or '(none)'}")
            print(f"  Mode     : {self.executor.mode}")
            print(f"{sep}\n")

        # ── Phase 1: Plan ──────────────────────────────────────────────────────
        if verbose:
            print("▶ PHASE 1: Planner (Φplan) — deconstructing question…")

        plan = self.planner.plan(question, image_path)

        if verbose:
            print(f"\n  Breakdown : {plan.deconstruction}")
            print(f"  Strategy  : {plan.reasoning}")
            print(f"\n  Tool plan ({len(plan)} steps):")
            for s in plan:
                print(f"    [{s.step}] {s.tool_name}")
                print(f"          args    : {json.dumps(s.args)}")
                print(f"          purpose : {s.purpose}")
                print(f"          why     : {s.reasoning}")
            if plan.tools_excluded:
                print(f"\n  Tools NOT used:")
                for ex in plan.tools_excluded:
                    print(f"    ✗ {ex.get('tool_name','?')}: {ex.get('reason','')}")
            print()

        # ── Phase 2: Execute ───────────────────────────────────────────────────
        if verbose:
            print("▶ PHASE 2: Executor (Φexec) — running tool chain…")

        def on_step(i, step, result):
            if result is None and verbose:
                print(f"  [{step.step}/{len(plan)}] {step.tool_name}…",
                      end=" ", flush=True)
            elif result is not None and verbose:
                status = "✓" if result.success else "✗"
                print(status)

        results = self.executor.execute(plan, question, image_path, on_step=on_step)

        if verbose:
            print()

        # ── Phase 3: Summarize ─────────────────────────────────────────────────
        if verbose:
            print("▶ PHASE 3: Summarizer (Φsum) — synthesising report…\n")

        report = self.summarizer.summarize(results, question, image_path, plan=plan)

        if verbose:
            print(report)
            print(f"\n{sep}\n")

        return AgentResult(question=question, image_path=image_path,
                           plan=plan, tool_results=results, report=report)


def main():
    parser = argparse.ArgumentParser(description="CXR SpatialAgent")
    parser.add_argument("--image",    type=str, default=None)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--mode",     choices=["real", "simulated"], default="real",
                        help="real=DL models (default); simulated=CheXagent acts as tools")
    parser.add_argument("--plan-only", action="store_true",
                        help="Deconstruct question and print plan without executing")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    agent = SpatialAgent(mode=args.mode)

    if args.plan_only:
        plan = agent.planner.plan(args.question, args.image)
        print(f"\n{'═'*62}")
        print("  Planner output")
        print(f"{'═'*62}")
        print(f"\n  Question    : {plan.question}")
        print(f"  Breakdown   : {plan.deconstruction}")
        print(f"  Strategy    : {plan.reasoning}")
        print(f"\n  Steps ({len(plan)}):")
        for s in plan:
            print(f"\n  [{s.step}] {s.tool_name}  (id: {s.tool_id})")
            print(f"       args    : {json.dumps(s.args)}")
            print(f"       purpose : {s.purpose}")
            print(f"       why     : {s.reasoning}")
        if plan.tools_excluded:
            print(f"\n  Excluded tools:")
            for ex in plan.tools_excluded:
                print(f"    ✗ {ex.get('tool_name','?')}: {ex.get('reason','')}")
        return

    result = agent.run(question=args.question, image_path=args.image)

    if args.output_json:
        out = {
            "question":   result.question,
            "image_path": result.image_path,
            "plan": {
                "deconstruction":  result.plan.deconstruction,
                "reasoning":       result.plan.reasoning,
                "steps": [
                    {
                        "step":      s.step,
                        "tool_id":   s.tool_id,
                        "tool_name": s.tool_name,
                        "args":      s.args,
                        "purpose":   s.purpose,
                        "reasoning": s.reasoning,
                    }
                    for s in result.plan
                ],
                "tools_excluded": result.plan.tools_excluded,
            },
            "tool_results": [
                {
                    "step":      r.step,
                    "tool_id":   r.tool_id,
                    "tool_name": r.tool_name,
                    "success":   r.success,
                    "output":    r.output,
                    "error":     r.error,
                }
                for r in result.tool_results
            ],
            "report": result.report,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
