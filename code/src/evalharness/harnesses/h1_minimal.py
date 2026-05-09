"""H1 minimal harness: open-loop baseline with raw context growth."""

from __future__ import annotations

import json
from typing import Optional

from ..core import AgentState, ErrorType, Harness, Observation, TaskRuntime, ToolCall


class H1Minimal(Harness):
    """Open-loop baseline: append-all context and no feedback mechanisms."""

    def build_context(self, state: AgentState, runtime: TaskRuntime) -> str:
        tools = runtime.list_tool_specs(
            schema_style=self.config["tools"]["schema_style"],
            exposure=self.config["tools"]["max_tools_exposed"],
        )
        tool_block = "\n".join(tool.render("verbose") for tool in tools)
        parts = [
            f"<pr_description>",
            state.goal,
            f"</pr_description>",
            "",
            "<instructions>",
            "# Task Instructions",
            "",
            "## Overview",
            "You're a software engineer working inside a benchmark workspace. "
            "Investigate the repository with the provided tools, reproduce the bug, "
            "apply the minimal fix, and verify it. You MUST actually call an edit "
            "tool (`write_file` or `replace_in_file`) to change code — describing "
            "the fix in prose is not enough.",
            "",
            "## Important Boundaries",
            f"- MODIFY source code in the repository (workspace: {runtime.working_directory}).",
            "- DO NOT MODIFY test files (tests/, conftest.py, *_test.py, test_*.py).",
            "- DO NOT MODIFY build/packaging config (pyproject.toml, setup.cfg, setup.py).",
            "",
            "## Recommended Workflow",
            "1. Explore: `list_files` + `read_file` to find the relevant source code.",
            "2. Reproduce: `run_command` with a short Python script that triggers the bug.",
            "3. Edit: `replace_in_file` (preferred) or `write_file` to fix the source code. "
            "You MUST make at least one edit to source files.",
            "4. Verify: `run_command` to confirm the fix. Also `git_diff` to check your changes.",
            "5. Submit: only after verifying that `git_diff` shows your changes.",
            "",
            "CRITICAL: Do NOT submit until you have made actual code changes. "
            "The harness will reject submissions with an empty git diff.",
            "",
            "## Command Execution Rules",
            "- Each response MUST be a single JSON tool call on its own line:",
            '  {"tool": "<name>", "args": {...}}',
            "- You can include brief THOUGHT reasoning before the JSON if you want, "
            "but the JSON must be valid and the last thing in the response.",
            "- Paths are relative to the workspace directory.",
            "",
            "## Example Response (abridged)",
            '  Thought: Need to locate the logging setup in config.py.',
            '  {"tool": "read_file", "args": {"path": "pydicom/config.py"}}',
            "  ... (later, after finding the culprit) ...",
            '  {"tool": "replace_in_file", "args": {"path": "pydicom/config.py", '
            '"old": "handler = logging.StreamHandler()", '
            '"new": "handler = logging.NullHandler()"}}',
            "",
            "## Submission",
            "Once your fix is applied and verified, emit EXACTLY this on its own line "
            "to terminate the trajectory:",
            f"  <final>COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT — short summary of fix</final>",
            "EvalHarness collects `git diff --binary` automatically — you do not need "
            "to print the patch yourself.",
            "",
            "## Available Tools",
            tool_block,
            "</instructions>",
            "",
        ]
        if "last_premature_final" in state.persistent_memory:
            parts.extend(
                [
                    "IMPORTANT: your previous response tried to terminate without "
                    "making any edits. Do NOT emit <final> yet. Use an edit tool.",
                    "",
                ]
            )

        if state.history:
            parts.append("## History")
            for step in state.history:
                parts.append(f"Step {step.step_id}:")
                parts.append(f"  action: {self._action_to_str(step.action)}")
                out = str(step.observation.output)
                if len(out) > 1500:
                    out = out[:800] + f"\n... [truncated {len(out)-1500} chars] ...\n" + out[-700:]
                parts.append(f"  output: {out}")
                if step.observation.error is not None:
                    parts.append(
                        f"  error: {step.observation.error.value}: "
                        f"{step.observation.error_message}"
                    )
            parts.append("")
        parts.append("Next action (respond with a single JSON tool call):")
        return "\n".join(parts)

    def validate_action(
        self,
        raw_output: str,
        runtime: TaskRuntime,
    ) -> tuple[Optional[ToolCall], Optional[ErrorType]]:
        # NOTE: do NOT check _is_terminal_output here — the main loop
        # handles terminal detection and prefers executing tool calls
        # when both a tool call and <final> appear in the same output.
        # Try each line for a valid JSON tool call (models may emit multiple)
        for line in raw_output.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return ToolCall(name=parsed["tool"], arguments=parsed.get("args", {})), None
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        # Fallback: find first { ... } block
        try:
            start = raw_output.find("{")
            if start == -1:
                return None, ErrorType.MALFORMED_OUTPUT
            # Find matching closing brace
            depth = 0
            for i in range(start, len(raw_output)):
                if raw_output[i] == "{":
                    depth += 1
                elif raw_output[i] == "}":
                    depth -= 1
                    if depth == 0:
                        parsed = json.loads(raw_output[start : i + 1])
                        if "tool" in parsed:
                            return ToolCall(name=parsed["tool"], arguments=parsed.get("args", {})), None
                        break
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return None, ErrorType.MALFORMED_OUTPUT

    def filter_observation(self, obs: Observation) -> Observation:
        return obs

    def should_retry(self, obs: Observation, attempt: int) -> bool:
        # H1 is minimal but at least tolerates 1 parse retry — otherwise
        # the model's first stray prose answer kills the whole trajectory.
        if obs.error is None:
            return False
        if obs.error.value in {"malformed_output"} and attempt == 0:
            return True
        return False

    @staticmethod
    def _action_to_str(action: ToolCall) -> str:
        return json.dumps({"tool": action.name, "args": action.arguments}, ensure_ascii=True)
