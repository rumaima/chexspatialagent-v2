# prompts/executor.py — System prompt for the Executor agent (Φexec)

EXECUTOR_SYSTEM = """\
You are a medical CXR spatial analysis executor. You simulate executing a \
specialised imaging tool and return realistic, clinically plausible results.

Given a tool invocation, return ONLY a JSON object matching the tool's output \
format. Be specific and clinically realistic. Reflect any pathology suggested \
by the clinical context. No markdown, no explanation — just the JSON result.\
"""


EXECUTOR_USER_TEMPLATE = """\
Tool: {tool_name} (id: {tool_id})
Purpose: {purpose}
Args: {args_json}
Expected output format: {output_format}
Clinical question context: {question}

Return the simulated tool output JSON for this CXR analysis step.\
"""


def build_executor_user(
    tool_name: str,
    tool_id: str,
    purpose: str,
    args: dict,
    output_format: str,
    question: str,
) -> str:
    import json
    return EXECUTOR_USER_TEMPLATE.format(
        tool_name=tool_name,
        tool_id=tool_id,
        purpose=purpose,
        args_json=json.dumps(args),
        output_format=output_format,
        question=question,
    )
