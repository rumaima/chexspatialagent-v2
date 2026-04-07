# tools/base.py — BaseTool interface for all CXR spatial analysis tools

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Serialisable description of a tool, sent to the planner."""
    id: str
    name: str
    category: str
    description: str
    input_format: str
    output_format: str
    example: str

    def to_prompt_block(self) -> str:
        return (
            f"Tool: {self.name} (id: {self.id})\n"
            f"Category: {self.category}\n"
            f"Description: {self.description}\n"
            f"Input: {self.input_format}\n"
            f"Output format: {self.output_format}\n"
            f"Example use: {self.example}"
        )


@dataclass
class ToolResult:
    tool_id: str
    tool_name: str
    step: int
    success: bool
    output: dict = field(default_factory=dict)
    error: str = ""

    def to_findings_block(self) -> str:
        import json
        return (
            f"Step {self.step} — {self.tool_name}:\n"
            f"{json.dumps(self.output, indent=2)}"
        )


class BaseTool(ABC):
    """
    All CXR tools subclass this. Implement `execute()` with your model or
    heuristic logic. The executor calls `run()` which wraps execute() with
    error handling and result packaging.
    """

    # Override these as class attributes in subclasses
    id: str = ""
    name: str = ""
    category: str = ""
    description: str = ""
    input_format: str = "CXR image"
    output_format: str = "JSON object"
    example: str = ""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            id=self.id,
            name=self.name,
            category=self.category,
            description=self.description,
            input_format=self.input_format,
            output_format=self.output_format,
            example=self.example,
        )

    @abstractmethod
    def execute(self, image_path: str | None, args: dict) -> dict:
        """
        Run the tool. Return a dict matching the tool's output_format.

        Args:
            image_path: Local path to the CXR image (JPEG/PNG). May be None
                        if the tool operates on previously collected findings.
            args: Free-form dict from the planner's args field for this step.

        Returns:
            Dict of findings. Must be JSON-serialisable.
        """
        ...

    def run(self, image_path: str | None, args: dict, step: int) -> ToolResult:
        """Called by the executor. Wraps execute() with error handling."""
        try:
            output = self.execute(image_path, args)
            return ToolResult(
                tool_id=self.id,
                tool_name=self.name,
                step=step,
                success=True,
                output=output,
            )
        except Exception as exc:
            return ToolResult(
                tool_id=self.id,
                tool_name=self.name,
                step=step,
                success=False,
                output={},
                error=str(exc),
            )
