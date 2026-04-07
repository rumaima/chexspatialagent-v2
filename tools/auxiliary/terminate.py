# tools/auxiliary/terminate.py

from tools.base import BaseTool


class Terminate(BaseTool):
    """
    Signals the end of the execution plan.

    Terminate consolidates all prior tool outputs into a single findings dict
    and signals the executor that no further steps should run. The summarizer
    then uses the consolidated output (alongside the original question and image)
    to produce the final report.

    This tool always executes synchronously — it never calls a model or
    external service, so it works identically in both 'real' and 'simulated'
    modes.
    """

    id = "terminate"
    name = "Terminate"
    category = "Auxiliary"
    description = (
        "Signals completion of the plan. "
        "Consolidates all tool outputs and passes them to the summarizer."
    )
    input_format = "All collected tool outputs (passed automatically by executor)"
    output_format = '{ "status": "complete", "steps_executed": int }'
    example = "Always the final step in any plan"

    def execute(self, image_path: str | None, args: dict) -> dict:
        steps_executed = args.get("steps_executed", 0)
        return {
            "status": "complete",
            "steps_executed": steps_executed,
            "message": "All tool steps completed. Passing findings to summarizer.",
        }
