"""H2 improved harness: compressed context plus retry."""

from __future__ import annotations

import json
import re
from typing import Optional

from ..core import AgentState, ErrorType, Harness, Observation, Step, TaskRuntime, ToolCall
from ..utils import truncate_text


class H2Improved(Harness):
    """Closed-loop harness with compression, retrieval, and retry."""

    def build_context(self, state: AgentState, runtime: TaskRuntime) -> str:
        tools = runtime.list_tool_specs(
            schema_style=self.config["tools"]["schema_style"],
            exposure=self.config["tools"]["max_tools_exposed"],
        )
        window = int(self.config["context"]["compression_window"])
        top_k = int(self.config["context"]["retrieval"]["top_k"])

        recent = state.history[-window:] if state.history else []
        older = state.history[:-window] if len(state.history) > window else []

        tool_block = "\n".join(tool.render("minimal") for tool in tools)
        parts = [
            "<pr_description>",
            state.goal,
            "</pr_description>",
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
            "- Each non-final response MUST be exactly one JSON object and nothing else: "
            '{"tool": "<name>", "args": {...}}.',
            "- Do NOT write Thought, reasoning, explanation, markdown, code fences, or prose.",
            "- The JSON object must be the entire response. It must not be wrapped in text.",
            "- Paths are relative to the workspace directory.",
            "",
            "## Example Response",
            '  {"tool": "read_file", "args": {"path": "pydicom/config.py"}}',
            "",
            "## Submission",
            "After applying the fix, emit EXACTLY one final message of the form:",
            "  <final>COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT — summary</final>",
            "EvalHarness collects `git diff --binary` automatically.",
            "",
            "## Available Tools",
            tool_block,
            "</instructions>",
            "",
        ]
        if "drift_reset" in state.persistent_memory:
            parts.extend([
                "## Harness note",
                f"  {state.persistent_memory['drift_reset']['note']}",
                "",
            ])
        if "last_premature_final" in state.persistent_memory:
            parts.extend([
                "IMPORTANT: your previous response tried to terminate without "
                "making any edits. Do NOT emit <final> yet. Use an edit tool.",
                "",
            ])
        if state.history and state.history[-1].action.name == "_parse_error":
            parts.extend([
                "PARSER ERROR: your previous response was not a valid JSON tool call.",
                "Recover now by outputting exactly one JSON object and nothing else.",
                'Example: {"tool": "read_file", "args": {"path": "path/to/file.py"}}',
                "",
            ])

        if older:
            parts.extend([
                "## Summary of earlier steps",
                f"  {self._summarize_history(older)}",
                "",
            ])
            relevant = self._bm25_retrieve(state.goal, older, k=top_k)
            if relevant:
                parts.append("## Relevant earlier steps (retrieved)")
                for step in relevant:
                    parts.append(
                        f"Step {step.step_id}: {self._action_to_str(step.action)} -> "
                        f"{truncate_text(str(step.observation.output), 220)}"
                    )
                parts.append("")

        if recent:
            parts.append("## Recent steps")
            for step in recent:
                parts.append(f"Step {step.step_id}:")
                parts.append(f"  action: {self._action_to_str(step.action)}")
                # 8000 chars ≈ 2000 tokens — enough to hold most full file
                # reads or command outputs so the model doesn't infinite-loop
                # re-reading the same file because it never got to the
                # relevant line. max_tokens is 32000, so 8 recent steps still
                # fit with plenty of headroom.
                parts.append(f"  output: {truncate_text(str(step.observation.output), 8000)}")
                if step.observation.error is not None:
                    parts.append(f"  error: {step.observation.error.value}: "
                                 f"{step.observation.error_message or ''}")
            parts.append("")

        parts.append("Next action (respond with a single JSON tool call):")
        return "\n".join(parts)

    def validate_action(
        self,
        raw_output: str,
        runtime: TaskRuntime,
    ) -> tuple[Optional[ToolCall], Optional[ErrorType]]:
        # NOTE: terminal detection handled by main loop, not here.
        # Scan for complete JSON objects. This is more robust than hand-written
        # brace matching because braces can occur inside JSON strings.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw_output):
            try:
                parsed, _ = decoder.raw_decode(raw_output[match.start() :])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            tool_call = self._to_tool_call(parsed)
            if tool_call is not None:
                return tool_call, None
        return None, ErrorType.MALFORMED_OUTPUT

    @staticmethod
    def _to_tool_call(parsed: object) -> Optional[ToolCall]:
        if not isinstance(parsed, dict):
            return None
        tool = parsed.get("tool")
        if not isinstance(tool, str):
            return None
        args = parsed.get("args", {})
        if not isinstance(args, dict):
            return None
        return ToolCall(name=tool, arguments=args)

    def filter_observation(self, obs: Observation) -> Observation:
        if obs.error_message:
            sanitized = re.sub(r"/[^\s:]+", "<path>", obs.error_message)
            sanitized = re.sub(r"0x[0-9a-fA-F]+", "<addr>", sanitized)
            obs.error_message = truncate_text(sanitized, 300)
        if isinstance(obs.output, str):
            obs.output = truncate_text(obs.output, 6_000)
        return obs

    def system_prompt(self, runtime: TaskRuntime) -> str:
        return (
            "You are a SWE-bench coding agent. Until final submission, every "
            "assistant message must be exactly one JSON object tool call and "
            "nothing else. Do not include thoughts, analysis, markdown, or prose. "
            "When the fix is complete, submit only the exact final tag requested "
            "by the task instructions."
        )

    def should_retry(self, obs: Observation, attempt: int) -> bool:
        if obs.error is None:
            return False
        retry_cfg = self.config["orchestration"]["retry"]
        max_attempts = int(retry_cfg.get("max_attempts", 1))
        retryable = set(retry_cfg.get("retryable_errors", []))
        if attempt >= max_attempts - 1:
            return False
        return obs.error.value in retryable

    @staticmethod
    def _summarize_history(steps: list[Step]) -> str:
        if not steps:
            return "(no earlier steps)"
        tool_counts: dict[str, int] = {}
        error_count = 0
        for step in steps:
            tool_counts[step.action.name] = tool_counts.get(step.action.name, 0) + 1
            if step.observation.error is not None:
                error_count += 1
        usage = ", ".join(f"{tool}:{count}" for tool, count in sorted(tool_counts.items()))
        return (
            f"Executed {len(steps)} earlier steps. Tool usage: {usage}. "
            f"Observed errors: {error_count}."
        )

    @staticmethod
    def _bm25_retrieve(query: str, steps: list[Step], k: int) -> list[Step]:
        if not steps:
            return []
        query_terms = set(query.lower().split())
        scored: list[tuple[int, Step]] = []
        for step in steps:
            document = f"{step.action.name} {step.observation.output}".lower()
            overlap = sum(1 for term in query_terms if term in document)
            scored.append((overlap, step))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [step for score, step in scored[:k] if score > 0]

    @staticmethod
    def _action_to_str(action: ToolCall) -> str:
        return json.dumps({"tool": action.name, "args": action.arguments}, ensure_ascii=True)


class H2ImprovedV2(H2Improved):
    """Current H2 policy with anti-stall guidance."""

    def build_context(self, state: AgentState, runtime: TaskRuntime) -> str:
        base_context = super().build_context(state, runtime)
        v2_section = self._build_v2_policy_section(state)
        marker = "Next action (respond with a single JSON tool call):"
        if marker in base_context:
            return base_context.replace(marker, f"{v2_section}\n{marker}")
        return f"{base_context}\n{v2_section}"

    def system_prompt(self, runtime: TaskRuntime) -> str:
        return (
            super().system_prompt(runtime)
            + " This run uses H2 policy: avoid excessive repeated searching, "
            "local environment setup, or visible-test chasing, but keep enough "
            "evidence to make and validate a targeted source edit."
        )

    @staticmethod
    def _build_v2_policy_section(state: AgentState) -> str:
        history = state.history
        step_count = len(history)
        edit_steps = [
            step for step in history
            if step.action.name in {"replace_in_file", "write_file"}
            and step.observation.error is None
        ]
        run_errors = [
            step for step in history
            if step.action.name == "run_command" and step.observation.error is not None
        ]
        env_failures = [
            step for step in history
            if step.action.name == "run_command"
            and H2ImprovedV2._looks_like_environment_failure(str(step.observation.output))
        ]
        hidden_test_chases = [
            step for step in history
            if H2ImprovedV2._looks_like_missing_test_chase(step)
        ]
        visible_test_passes = [
            step for step in history
            if step.action.name == "run_command"
            and H2ImprovedV2._looks_like_visible_tests_passed(str(step.observation.output))
        ]

        notes = [
            "## H2 policy",
            "- Hidden/eval test names are often not present in the workspace. "
            "If repeated searches or AttributeError results suggest a named test "
            "is unavailable, shift from looking for that exact test to reasoning "
            "from the problem statement and related source code.",
            "- If imports, builds, pytest installation, or package setup fail twice, "
            "avoid spending many more steps on environment repair. Prefer "
            "source-level reasoning and a targeted code edit.",
            "- Passing visible/local tests does NOT mean the issue is fixed. If the "
            "problem statement describes a behavior gap, still patch the source.",
            "- Do not end with an empty diff. If you have identified a likely file "
            "and function, consider a minimal `replace_in_file` edit instead of "
            "continuing broad exploration.",
            "",
            "## H2 run status",
            f"- Steps used: {step_count}/50.",
            f"- Successful source edit tool calls so far: {len(edit_steps)}.",
            f"- Run-command errors so far: {len(run_errors)}.",
        ]
        if env_failures:
            notes.append(
                f"- Environment/setup failures observed: {len(env_failures)}. "
                "Avoid repeated install/build/test attempts unless the next one is "
                "likely to change the diagnosis."
            )
        if hidden_test_chases:
            notes.append(
                f"- Missing-test chasing signals observed: {len(hidden_test_chases)}. "
                "Use the available source context instead of repeatedly searching "
                "for hidden tests."
            )
        if visible_test_passes and not edit_steps:
            notes.append(
                "- Local visible tests have passed but no source edit exists. "
                "Treat this as insufficient coverage and still patch the issue."
            )
        if step_count >= 25 and not edit_steps:
            notes.append(
                "- Late-stage guidance: no source edit exists yet. Prefer moving "
                "toward a concrete `replace_in_file` or `write_file` edit; read "
                "again only if needed to identify the precise edit location."
            )
        if step_count >= 35 and not edit_steps:
            notes.append(
                "- If exploration is no longer producing new evidence, choose the "
                "best supported hypothesis from the files already read and make a "
                "targeted source edit."
            )
        if step_count >= 40 and edit_steps:
            notes.append(
                "- You already edited source. Inspect `git_diff` and, if feasible, "
                "run a focused validation before submitting. Avoid broad or "
                "repetitive tests that are unlikely to add signal."
            )
        notes.append("")
        return "\n".join(notes)

    @staticmethod
    def _looks_like_environment_failure(output: str) -> bool:
        lowered = output.lower()
        patterns = (
            "no module named pytest",
            "no module named django",
            "no module named matplotlib",
            "no module named sklearn",
            "module not found",
            "modulenotfounderror",
            "build_ext",
            "pip install",
            "setup.py egg_info did not run successfully",
            "multiple top-level packages discovered",
            "check_build",
        )
        return any(pattern in lowered for pattern in patterns)

    @staticmethod
    def _looks_like_missing_test_chase(step: Step) -> bool:
        action = step.action
        args_text = json.dumps(action.arguments, ensure_ascii=True).lower()
        output = str(step.observation.output).lower()
        if "test_" not in args_text:
            return False
        missing_patterns = (
            "no matches",
            "has no attribute",
            "unittest.loader._failedtest",
            "collected 0 items",
            "not found",
        )
        return any(pattern in output for pattern in missing_patterns) or output.strip() == ""

    @staticmethod
    def _looks_like_visible_tests_passed(output: str) -> bool:
        lowered = output.lower()
        if "failed" in lowered or "error" in lowered:
            return False
        tail_lines = [line.strip() for line in lowered.splitlines()[-3:]]
        return " passed" in lowered or any(line == "ok" for line in tail_lines)
