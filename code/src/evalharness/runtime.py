"""Prepared-workspace task runtime for coding benchmark experiments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import AgentState, ErrorType, GradeResult, Observation, TaskRuntime, ToolCall, ToolSpec
from .utils import load_mapping_file, slugify, truncate_text


@dataclass
class PreparedWorkspaceTask:
    """A task backed by a local workspace template and a grader."""

    task_id: str
    goal: str
    workspace_template: Path
    grader: dict[str, Any]
    allowed_tools: list[str] | None = None
    setup_commands: list[list[str] | str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def make_runtime(self, scratch_root: str | Path, run_label: str) -> TaskRuntime:
        return PreparedWorkspaceRuntime(self, scratch_root=scratch_root, run_label=run_label)


def load_tasks(path: str | Path) -> list[PreparedWorkspaceTask]:
    """Load JSONL or JSON task definitions."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        rows = load_mapping_file(file_path)
        if isinstance(rows, dict) and "tasks" in rows:
            rows = rows["tasks"]
    else:
        rows = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    tasks: list[PreparedWorkspaceTask] = []
    for row in rows:
        workspace_template = (file_path.parent / row["workspace_template"]).resolve()
        tasks.append(
            PreparedWorkspaceTask(
                task_id=row["task_id"],
                goal=row["goal"],
                workspace_template=workspace_template,
                grader=row.get("grader", {"type": "always_pass"}),
                allowed_tools=row.get("allowed_tools"),
                setup_commands=row.get("setup_commands", []),
                metadata=row.get("metadata", {}),
            )
        )
    return tasks


class PreparedWorkspaceRuntime(TaskRuntime):
    """Concrete runtime that copies a prepared workspace for each run."""

    def __init__(self, task: PreparedWorkspaceTask, scratch_root: str | Path, run_label: str):
        self.task = task
        self.task_id = task.task_id
        self.goal = task.goal
        self.metadata = dict(task.metadata)
        scratch_root = Path(scratch_root)
        scratch_root.mkdir(parents=True, exist_ok=True)

        self.workspace_root = (scratch_root / slugify(task.task_id) / slugify(run_label)).resolve()
        if self.workspace_root.exists():
            shutil.rmtree(self.workspace_root)
        shutil.copytree(task.workspace_template, self.workspace_root)
        self.working_directory = self.workspace_root
        self._run_setup_commands()

    def list_tool_specs(self, schema_style: str, exposure: str) -> list[ToolSpec]:
        tool_specs = self._all_tool_specs()
        if exposure == "task_relevant" and self.task.allowed_tools:
            allowed = set(self.task.allowed_tools)
            tool_specs = [tool for tool in tool_specs if tool.name in allowed]
        return tool_specs

    def execute_tool(self, action: ToolCall) -> Observation:
        dispatch = {
            "list_files": self._tool_list_files,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "replace_in_file": self._tool_replace_in_file,
            "search_text": self._tool_search_text,
            "run_command": self._tool_run_command,
            "git_status": self._tool_git_status,
            "git_diff": self._tool_git_diff,
        }
        handler = dispatch.get(action.name)
        if handler is None:
            return Observation(
                output="",
                error=ErrorType.PERMISSION_DENIED,
                error_message=f"unknown tool: {action.name}",
                tool_name=action.name,
            )

        if self.task.allowed_tools and action.name not in set(self.task.allowed_tools):
            return Observation(
                output="",
                error=ErrorType.PERMISSION_DENIED,
                error_message=f"tool not allowed for task: {action.name}",
                tool_name=action.name,
            )

        try:
            return handler(action.arguments)
        except TimeoutError as exc:
            return Observation(
                output="",
                error=ErrorType.TIMEOUT,
                error_message=str(exc),
                tool_name=action.name,
            )
        except PermissionError as exc:
            return Observation(
                output="",
                error=ErrorType.PERMISSION_DENIED,
                error_message=str(exc),
                tool_name=action.name,
            )
        except Exception as exc:
            return Observation(
                output="",
                error=ErrorType.UNKNOWN,
                error_message=str(exc),
                tool_name=action.name,
            )

    def grade(self, final_output: Any, state: AgentState) -> GradeResult:
        grader_type = self.task.grader.get("type", "always_pass")
        if grader_type == "always_pass":
            return GradeResult(success=True, score=1.0)

        if grader_type == "command":
            command = self.task.grader["command"]
            success_exit_codes = set(self.task.grader.get("success_exit_codes", [0]))
            proc = subprocess.run(
                command,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=int(self.task.grader.get("timeout_sec", 120)),
                env=self._command_env(final_output),
            )
            success = proc.returncode in success_exit_codes
            return GradeResult(
                success=success,
                score=1.0 if success else 0.0,
                stdout=truncate_text(proc.stdout, 8_000),
                stderr=truncate_text(proc.stderr, 8_000),
                details={"returncode": proc.returncode},
            )

        if grader_type == "file_exists":
            target = self._resolve_path(self.task.grader["path"])
            success = target.exists()
            return GradeResult(success=success, score=1.0 if success else 0.0)

        if grader_type == "file_contains":
            target = self._resolve_path(self.task.grader["path"])
            expected = self.task.grader["substring"]
            content = target.read_text(encoding="utf-8") if target.exists() else ""
            success = expected in content
            return GradeResult(
                success=success,
                score=1.0 if success else 0.0,
                details={"path": str(target)},
            )

        if grader_type == "final_output_contains":
            expected = self.task.grader["substring"]
            text = "" if final_output is None else str(final_output)
            success = expected in text
            return GradeResult(success=success, score=1.0 if success else 0.0)

        raise ValueError(f"Unsupported grader type: {grader_type}")

    def close(self) -> None:
        return None

    def _run_setup_commands(self) -> None:
        for command in self.task.setup_commands:
            subprocess.run(
                command,
                cwd=self.workspace_root,
                check=True,
                capture_output=True,
                text=True,
            )

    def _command_env(self, final_output: Any) -> dict[str, str]:
        env = dict(os.environ)
        env["EVALHARNESS_WORKSPACE"] = str(self.workspace_root)
        env["EVALHARNESS_FINAL_OUTPUT"] = "" if final_output is None else str(final_output)
        return env

    def _resolve_path(self, path_str: str) -> Path:
        candidate = (self.workspace_root / path_str).resolve()
        if self.workspace_root.resolve() not in candidate.parents and candidate != self.workspace_root.resolve():
            raise PermissionError(f"path escapes workspace: {path_str}")
        return candidate

    def _tool_list_files(self, arguments: dict[str, Any]) -> Observation:
        raw_path = arguments.get("path", ".")
        start = self._resolve_path(raw_path)
        if not start.exists():
            return Observation(
                output="",
                error=ErrorType.UNKNOWN,
                error_message=f"path does not exist: {raw_path}",
                tool_name="list_files",
            )
        if not start.is_dir():
            return Observation(
                output="",
                error=ErrorType.UNKNOWN,
                error_message=f"path is not a directory: {raw_path}",
                tool_name="list_files",
            )
        max_entries = int(arguments.get("max_entries", 200))
        lines: list[str] = []
        for root, dirs, files in os.walk(start):
            rel_root = Path(root).relative_to(self.workspace_root)
            for name in sorted(dirs):
                lines.append(f"{rel_root / name}/")
            for name in sorted(files):
                lines.append(str(rel_root / name))
            if len(lines) >= max_entries:
                break
        return Observation(output="\n".join(lines[:max_entries]), tool_name="list_files")

    def _tool_read_file(self, arguments: dict[str, Any]) -> Observation:
        target = self._resolve_path(arguments["path"])
        text = target.read_text(encoding="utf-8")
        start_line = int(arguments.get("start_line", 1))
        end_line = arguments.get("end_line")
        lines = text.splitlines()
        end_index = len(lines) if end_line is None else int(end_line)
        excerpt = "\n".join(lines[max(start_line - 1, 0) : end_index])
        payload = (
            f"path: {target.relative_to(self.workspace_root)}\n"
            "----- BEGIN FILE -----\n"
            f"{excerpt}\n"
            "----- END FILE -----"
        )
        return Observation(output=payload, tool_name="read_file")

    def _tool_write_file(self, arguments: dict[str, Any]) -> Observation:
        target = self._resolve_path(arguments["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(arguments.get("content", ""))
        append = bool(arguments.get("append", False))
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        action = "appended" if append else "wrote"
        return Observation(
            output=f"{action} {len(content)} chars to {target.relative_to(self.workspace_root)}",
            tool_name="write_file",
        )

    def _tool_replace_in_file(self, arguments: dict[str, Any]) -> Observation:
        target = self._resolve_path(arguments["path"])
        old = str(arguments["old"])
        new = str(arguments["new"])
        count = arguments.get("count")
        text = target.read_text(encoding="utf-8")
        if old not in text:
            return Observation(
                output="",
                error=ErrorType.UNKNOWN,
                error_message=f"text not found in {target.relative_to(self.workspace_root)}",
                tool_name="replace_in_file",
            )
        replaced = text.replace(old, new, int(count)) if count is not None else text.replace(old, new)
        target.write_text(replaced, encoding="utf-8")
        return Observation(
            output=f"updated {target.relative_to(self.workspace_root)}",
            tool_name="replace_in_file",
        )

    def _tool_search_text(self, arguments: dict[str, Any]) -> Observation:
        pattern = str(arguments["pattern"])
        search_root = self._resolve_path(arguments.get("path", "."))
        rg_cmd = ["rg", "-n", "--no-heading", pattern, str(search_root)]
        try:
            proc = subprocess.run(
                rg_cmd,
                capture_output=True,
                text=True,
                timeout=int(arguments.get("timeout_sec", 30)),
            )
        except FileNotFoundError:
            proc = subprocess.run(
                ["grep", "-R", "-n", pattern, str(search_root)],
                capture_output=True,
                text=True,
                timeout=int(arguments.get("timeout_sec", 30)),
            )
        output = truncate_text(proc.stdout.strip(), 8_000)
        if proc.returncode not in (0, 1):
            return Observation(
                output=output,
                error=ErrorType.UNKNOWN,
                error_message=truncate_text(proc.stderr.strip(), 1_000),
                tool_name="search_text",
            )
        return Observation(output=output, tool_name="search_text")

    def _tool_run_command(self, arguments: dict[str, Any]) -> Observation:
        command = arguments["command"]
        timeout_sec = int(arguments.get("timeout_sec", 120))
        cwd = self._resolve_path(arguments.get("cwd", "."))
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=isinstance(command, str),
        )
        output = (
            "STDOUT:\n"
            f"{truncate_text(proc.stdout, 4_000)}\n"
            "STDERR:\n"
            f"{truncate_text(proc.stderr, 4_000)}\n"
            f"RETURN CODE: {proc.returncode}"
        )
        if proc.returncode != 0:
            return Observation(
                output=output,
                error=ErrorType.UNKNOWN,
                error_message=f"command exited with return code {proc.returncode}",
                tool_name="run_command",
            )
        return Observation(output=output, tool_name="run_command")

    def _tool_git_status(self, arguments: dict[str, Any]) -> Observation:
        del arguments
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = proc.stdout.strip() or "(clean working tree)"
        if proc.returncode != 0:
            return Observation(
                output=output,
                error=ErrorType.UNKNOWN,
                error_message=truncate_text(proc.stderr.strip(), 1_000),
                tool_name="git_status",
            )
        return Observation(output=truncate_text(output, 8_000), tool_name="git_status")

    def _tool_git_diff(self, arguments: dict[str, Any]) -> Observation:
        staged = bool(arguments.get("staged", False))
        command = ["git", "diff", "--binary"]
        if staged:
            command.append("--staged")
        proc = subprocess.run(
            command,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=int(arguments.get("timeout_sec", 30)),
        )
        output = proc.stdout.strip() or "(no diff)"
        if proc.returncode != 0:
            return Observation(
                output=truncate_text(output, 8_000),
                error=ErrorType.UNKNOWN,
                error_message=truncate_text(proc.stderr.strip(), 1_000),
                tool_name="git_diff",
            )
        return Observation(output=truncate_text(output, 8_000), tool_name="git_diff")

    def _all_tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="list_files",
                description="List files and directories under a workspace path.",
                minimal_description="List files under a directory.",
                arguments={
                    "path": "relative directory path, default '.'",
                    "max_entries": "optional integer cap on returned paths",
                },
            ),
            ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file from the workspace.",
                minimal_description="Read a text file.",
                arguments={
                    "path": "relative file path",
                    "start_line": "optional 1-based starting line",
                    "end_line": "optional 1-based inclusive ending line",
                },
            ),
            ToolSpec(
                name="write_file",
                description="Write or append UTF-8 text into a workspace file.",
                minimal_description="Write a text file.",
                arguments={
                    "path": "relative file path",
                    "content": "text content to write",
                    "append": "optional boolean; append instead of overwrite",
                },
            ),
            ToolSpec(
                name="replace_in_file",
                description="Replace a substring inside a UTF-8 text file.",
                minimal_description="Replace text inside a file.",
                arguments={
                    "path": "relative file path",
                    "old": "substring to replace",
                    "new": "replacement text",
                    "count": "optional maximum number of replacements",
                },
            ),
            ToolSpec(
                name="search_text",
                description="Search text recursively using ripgrep or grep.",
                minimal_description="Search for text in the workspace.",
                arguments={
                    "pattern": "search pattern",
                    "path": "optional relative path to search from",
                    "timeout_sec": "optional timeout in seconds",
                },
            ),
            ToolSpec(
                name="run_command",
                description="Run a shell command inside the workspace and capture stdout/stderr.",
                minimal_description="Run a command in the workspace.",
                arguments={
                    "command": "shell string or argv list",
                    "cwd": "optional relative working directory",
                    "timeout_sec": "optional timeout in seconds",
                },
            ),
            ToolSpec(
                name="git_status",
                description="Show concise git working tree status for the current workspace.",
                minimal_description="Show git status.",
                arguments={},
            ),
            ToolSpec(
                name="git_diff",
                description="Show the current git diff for the workspace.",
                minimal_description="Show git diff.",
                arguments={
                    "staged": "optional boolean; show staged diff only",
                    "timeout_sec": "optional timeout in seconds",
                },
            ),
        ]
