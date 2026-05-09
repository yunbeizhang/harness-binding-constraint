"""Docker-based task runtime for SWE-bench instances.

Executes all tool calls inside the official SWE-bench Docker container
so the agent has access to the full development environment (dependencies,
pytest, importable project modules, etc.).
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

from .core import ErrorType, GradeResult, Observation, ToolCall, ToolSpec

logger = logging.getLogger("evalharness.docker_runtime")


def get_swebench_docker_image(instance_id: str) -> str:
    """Compute the Docker image name for a SWE-bench instance."""
    id_safe = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_safe}:latest".lower()


class DockerSweBenchRuntime:
    """Runtime that executes tools inside a SWE-bench Docker container."""

    def __init__(
        self,
        *,
        instance_id: str,
        goal: str,
        metadata: dict[str, Any] | None = None,
        container_cwd: str = "/testbed",
        pull_timeout: int = 300,
        command_timeout: int = 120,
        docker_executable: str = "docker",
    ):
        self.task_id = instance_id
        self.goal = goal
        self.metadata = metadata or {}
        self.container_cwd = container_cwd
        self.command_timeout = command_timeout
        self.docker_executable = docker_executable
        self.working_directory = Path(container_cwd)

        image = get_swebench_docker_image(instance_id)
        docker_executable = os.environ.get("EVALHARNESS_DOCKER", docker_executable)
        self.docker_executable = docker_executable
        container_name = f"evalharness-{uuid.uuid4().hex[:8]}"

        logger.info("Starting container %s from %s", container_name, image)
        result = subprocess.run(
            [
                docker_executable, "run", "-d",
                "--name", container_name,
                "-w", container_cwd,
                "--rm",
                image,
                "sleep", "2h",
            ],
            capture_output=True, text=True, timeout=pull_timeout, check=True,
        )
        self.container_id = result.stdout.strip()
        self._container_name = container_name
        logger.info("Container %s started (id=%s)", container_name, self.container_id[:12])

    # ------------------------------------------------------------------
    # TaskRuntime interface
    # ------------------------------------------------------------------

    def list_tool_specs(self, schema_style: str = "verbose", exposure: str = "all") -> list[ToolSpec]:
        """Return tool definitions for prompt construction."""
        all_tools = [
            ToolSpec(
                name="list_files",
                description="List files in a directory.",
                arguments={"path": ("str", "Relative path (default: '.')"), "max_entries": ("int", "Max entries (default: 200)")},
            ),
            ToolSpec(
                name="read_file",
                description="Read a file's contents.",
                arguments={"path": ("str", "Relative path"), "start_line": ("int", "1-based start (optional)"), "end_line": ("int", "1-based end (optional)")},
            ),
            ToolSpec(
                name="write_file",
                description="Write content to a file (creates parent dirs).",
                arguments={"path": ("str", "Relative path"), "content": ("str", "File content"), "append": ("bool", "Append instead of overwrite (default: false)")},
            ),
            ToolSpec(
                name="replace_in_file",
                description="Replace exact text in a file.",
                arguments={"path": ("str", "Relative path"), "old": ("str", "Text to find"), "new": ("str", "Replacement text"), "count": ("int", "Max replacements (default: all)")},
            ),
            ToolSpec(
                name="search_text",
                description="Search for a pattern in files (grep -Rn).",
                arguments={"pattern": ("str", "Search pattern"), "path": ("str", "Directory to search (default: '.')"), "include": ("str", "File glob filter (optional)")},
            ),
            ToolSpec(
                name="run_command",
                description="Run a shell command.",
                arguments={"command": ("str", "Shell command to execute"), "cwd": ("str", "Working directory (optional)")},
            ),
            ToolSpec(
                name="git_status",
                description="Show git status --short.",
                arguments={},
            ),
            ToolSpec(
                name="git_diff",
                description="Show git diff.",
                arguments={"staged": ("bool", "Show staged changes (default: false)")},
            ),
        ]
        return all_tools

    def execute_tool(self, action: ToolCall) -> Observation:
        """Dispatch a tool call to the Docker container."""
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
                error=ErrorType.UNKNOWN,
                error_message=f"Unknown tool: {action.name}",
                tool_name=action.name,
            )
        try:
            return handler(action.arguments)
        except subprocess.TimeoutExpired:
            return Observation(
                output="", error=ErrorType.TIMEOUT,
                error_message=f"Command timed out after {self.command_timeout}s",
                tool_name=action.name,
            )
        except Exception as exc:
            return Observation(
                output="", error=ErrorType.UNKNOWN,
                error_message=str(exc),
                tool_name=action.name,
            )

    def grade(self, final_output: Any, state: Any) -> GradeResult:
        """Grade by extracting git diff from the container."""
        patch = self._extract_git_patch()
        changed_files = self._list_changed_files()
        return GradeResult(
            success=bool(patch.strip()),
            score=1.0 if patch.strip() else 0.0,
            stdout="", stderr="",
            details={
                "instance_id": self.task_id,
                "model_patch": patch,
                "changed_files": changed_files,
                "final_output": "" if final_output is None else str(final_output),
            },
        )

    def close(self) -> None:
        """Stop and remove the container."""
        if hasattr(self, "container_id") and self.container_id:
            logger.info("Stopping container %s", self._container_name)
            subprocess.Popen(
                f"(timeout 60 {self.docker_executable} stop {self.container_id} || "
                f"{self.docker_executable} rm -f {self.container_id}) >/dev/null 2>&1 &",
                shell=True,
            )

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Docker exec helper
    # ------------------------------------------------------------------

    def _docker_exec(
        self, command: str, cwd: str | None = None, timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command inside the container via docker exec."""
        cmd = [
            self.docker_executable, "exec",
            "-w", cwd or self.container_cwd,
            self.container_id,
            "bash", "-c", command,
        ]
        return subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout or self.command_timeout,
            encoding="utf-8", errors="replace",
        )

    def _docker_exec_stdin(
        self,
        command: str,
        *,
        input_text: str,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command inside the container and stream data through stdin."""
        cmd = [
            self.docker_executable, "exec", "-i",
            "-w", cwd or self.container_cwd,
            self.container_id,
            "bash", "-c", command,
        ]
        return subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout or self.command_timeout,
            encoding="utf-8", errors="replace",
        )

    def _validate_path(self, path_str: str) -> str:
        """Ensure path is under container_cwd. Returns absolute path."""
        if not path_str:
            return self.container_cwd
        # Normalize: if relative, prepend cwd; if absolute, check prefix
        if path_str.startswith("/"):
            if not path_str.startswith(self.container_cwd):
                raise ValueError(f"Path {path_str} is outside {self.container_cwd}")
            return path_str
        return f"{self.container_cwd}/{path_str}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_list_files(self, args: dict) -> Observation:
        path = self._validate_path(args.get("path", "."))
        max_entries = int(args.get("max_entries", 200))
        proc = self._docker_exec(f"find {path} -maxdepth 2 \\( -type f -o -type d \\) | head -{max_entries}")
        # Make paths relative to container_cwd
        lines = proc.stdout.strip().splitlines()
        prefix = self.container_cwd.rstrip("/") + "/"
        rel_lines = [l.replace(prefix, "", 1) if l.startswith(prefix) else l for l in lines]
        return Observation(output="\n".join(rel_lines), tool_name="list_files")

    def _tool_read_file(self, args: dict) -> Observation:
        path = self._validate_path(args.get("path", ""))
        proc = self._docker_exec(f"cat {path}")
        if proc.returncode != 0:
            return Observation(
                output=proc.stderr or proc.stdout,
                error=ErrorType.UNKNOWN,
                error_message=f"read_file failed (rc={proc.returncode})",
                tool_name="read_file",
            )
        content = proc.stdout
        # Apply line range if specified
        start = args.get("start_line")
        end = args.get("end_line")
        if start is not None or end is not None:
            lines = content.splitlines(keepends=True)
            s = (int(start) - 1) if start else 0
            e = int(end) if end else len(lines)
            content = "".join(lines[s:e])
        return Observation(output=content, tool_name="read_file")

    def _tool_write_file(self, args: dict) -> Observation:
        path = self._validate_path(args.get("path", ""))
        content = args.get("content", "")
        append = args.get("append", False)
        # Ensure parent directory exists
        parent = "/".join(path.split("/")[:-1])
        if parent:
            self._docker_exec(f"mkdir -p {shlex.quote(parent)}")
        op = ">>" if append else ">"
        proc = self._docker_exec_stdin(
            f"cat {op} {shlex.quote(path)}",
            input_text=content,
        )
        if proc.returncode != 0:
            return Observation(
                output=proc.stderr,
                error=ErrorType.UNKNOWN,
                error_message=f"write_file failed (rc={proc.returncode})",
                tool_name="write_file",
            )
        return Observation(output=f"Wrote {len(content)} chars to {args.get('path', path)}", tool_name="write_file")

    def _tool_replace_in_file(self, args: dict) -> Observation:
        path = self._validate_path(args.get("path", ""))
        old = args.get("old", "")
        new = args.get("new", "")
        count = args.get("count")
        if not old:
            return Observation(
                output="", error=ErrorType.MALFORMED_OUTPUT,
                error_message="'old' argument is required for replace_in_file",
                tool_name="replace_in_file",
            )
        # Read file content
        proc = self._docker_exec(f"cat {path}")
        if proc.returncode != 0:
            return Observation(
                output=proc.stderr,
                error=ErrorType.UNKNOWN,
                error_message=f"Cannot read file (rc={proc.returncode})",
                tool_name="replace_in_file",
            )
        original = proc.stdout
        if old not in original:
            return Observation(
                output=f"Pattern not found in {args.get('path', path)}",
                error=ErrorType.UNKNOWN,
                error_message="old text not found in file",
                tool_name="replace_in_file",
            )
        replaced = original.replace(old, new, int(count)) if count else original.replace(old, new)
        # Write back via stdin so large files do not overflow argv/env limits.
        proc = self._docker_exec_stdin(
            f"cat > {shlex.quote(path)}",
            input_text=replaced,
        )
        if proc.returncode != 0:
            return Observation(
                output=proc.stderr,
                error=ErrorType.UNKNOWN,
                error_message="write-back failed",
                tool_name="replace_in_file",
            )
        n = original.count(old) if count is None else min(int(count), original.count(old))
        return Observation(output=f"Replaced {n} occurrence(s) in {args.get('path', path)}", tool_name="replace_in_file")

    def _tool_search_text(self, args: dict) -> Observation:
        pattern = args.get("pattern", "")
        path = self._validate_path(args.get("path", "."))
        include = args.get("include", "")
        cmd = f"grep -Rn '{pattern}' {path}"
        if include:
            cmd = f"grep -Rn --include='{include}' '{pattern}' {path}"
        cmd += " | head -100"
        proc = self._docker_exec(cmd)
        # grep returns 1 if no match, which is not an error
        output = proc.stdout.strip()
        if not output and proc.returncode == 1:
            output = "(no matches)"
        return Observation(output=output, tool_name="search_text")

    def _tool_run_command(self, args: dict) -> Observation:
        command = args.get("command", "")
        cwd = args.get("cwd")
        if cwd:
            cwd = self._validate_path(cwd)
        proc = self._docker_exec(command, cwd=cwd)
        output = proc.stdout
        if proc.stderr:
            output = output + "\n[stderr]\n" + proc.stderr if output else proc.stderr
        error = None
        error_msg = None
        if proc.returncode != 0:
            error = ErrorType.UNKNOWN
            error_msg = f"Command exited with code {proc.returncode}"
        return Observation(output=output, error=error, error_message=error_msg, tool_name="run_command")

    def _tool_git_status(self, args: dict) -> Observation:
        proc = self._docker_exec("git status --short")
        return Observation(output=proc.stdout.strip(), tool_name="git_status")

    def _tool_git_diff(self, args: dict) -> Observation:
        staged = args.get("staged", False)
        cmd = "git diff --binary" + (" --staged" if staged else "")
        proc = self._docker_exec(cmd)
        return Observation(output=proc.stdout, tool_name="git_diff")

    # ------------------------------------------------------------------
    # Grading helpers
    # ------------------------------------------------------------------

    def _extract_git_patch(self) -> str:
        proc = self._docker_exec("git diff --binary")
        return proc.stdout if proc.returncode == 0 else ""

    def _list_changed_files(self) -> list[str]:
        proc = self._docker_exec("git status --short")
        if proc.returncode != 0:
            return []
        return [line[3:].strip() for line in proc.stdout.splitlines() if line.strip()]
