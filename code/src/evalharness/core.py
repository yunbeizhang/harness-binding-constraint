"""Core harness abstractions for EvalHarness."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

from .utils import approx_token_count, deep_merge, load_mapping_file, truncate_text


class ErrorType(Enum):
    """Typed error categories used by retry and stability logic."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TRANSIENT_NETWORK = "transient_network"
    MALFORMED_OUTPUT = "malformed_output"
    EMPTY_TOOL_OUTPUT = "empty_tool_output"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN = "unknown"


@dataclass
class UsageStats:
    """Provider-reported token usage when available."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cost: Optional[float] = None


@dataclass
class ModelResponse:
    """A normalized text response from a model provider."""

    text: str
    raw: Any = None
    usage: Optional[UsageStats] = None
    stop_reason: Optional[str] = None


class Model(Protocol):
    """Model interface used by the harness runtime."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Generate a completion for the provided prompt."""
        ...


@dataclass(frozen=True)
class ToolSpec:
    """Tool metadata shown to the model in different harness styles."""

    name: str
    description: str
    arguments: dict[str, str]
    minimal_description: Optional[str] = None

    def render(self, style: str) -> str:
        """Render a prompt-friendly tool description."""
        description = self.minimal_description or self.description
        if style == "verbose":
            arg_lines = "\n".join(
                f"      - {name}: {desc}" for name, desc in self.arguments.items()
            )
            return (
                f"  - name: {self.name}\n"
                f"    description: {self.description}\n"
                f"    arguments:\n{arg_lines}"
            )
        args_inline = ", ".join(f"{name}={desc}" for name, desc in self.arguments.items())
        return f"  - {self.name}: {description}. Args: {args_inline}"


@dataclass
class ToolCall:
    """A single tool invocation emitted by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class Observation:
    """Output returned by the environment after a tool call."""

    output: Any
    error: Optional[ErrorType] = None
    error_message: Optional[str] = None
    tool_name: Optional[str] = None
    wall_time_sec: float = 0.0


@dataclass
class Step:
    """One model action plus its resulting observation."""

    step_id: int
    action: ToolCall
    observation: Observation
    raw_model_output: str
    prompt_tokens_estimate: int
    verification: Optional[str] = None
    model_usage: Optional[UsageStats] = None
    verification_usage: Optional[UsageStats] = None
    wall_time_sec: float = 0.0


@dataclass
class GradeResult:
    """Post-run grading result for a task trajectory."""

    success: bool
    score: float = 0.0
    stdout: str = ""
    stderr: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class TaskRuntime(ABC):
    """Abstract task runtime used by the harness execution loop."""

    task_id: str
    goal: str
    working_directory: Path
    metadata: dict[str, Any]

    @abstractmethod
    def list_tool_specs(self, schema_style: str, exposure: str) -> list[ToolSpec]:
        """Return the tool list currently available to the task."""

    @abstractmethod
    def execute_tool(self, action: ToolCall) -> Observation:
        """Execute a tool call inside the task environment."""

    @abstractmethod
    def grade(self, final_output: Any, state: "AgentState") -> GradeResult:
        """Grade the completed trajectory."""

    def close(self) -> None:
        """Optional cleanup hook."""


class TaskDefinition(Protocol):
    """Task specification interface expected by the experiment driver."""

    task_id: str
    goal: str
    metadata: dict[str, Any]

    def make_runtime(self, scratch_root: str | Path, run_label: str) -> TaskRuntime:
        """Create an isolated runtime for one trajectory."""
        ...


@dataclass
class AgentState:
    """Internal harness state across a trajectory."""

    task_id: str
    goal: str
    history: list[Step] = field(default_factory=list)
    persistent_memory: dict[str, Any] = field(default_factory=dict)
    prompt_token_estimates: list[int] = field(default_factory=list)
    checkpoints: list["AgentState"] = field(default_factory=list)
    step_count: int = 0

    def snapshot(self) -> "AgentState":
        """Create a shallow checkpoint snapshot for rollback."""
        return AgentState(
            task_id=self.task_id,
            goal=self.goal,
            history=list(self.history),
            persistent_memory=dict(self.persistent_memory),
            prompt_token_estimates=list(self.prompt_token_estimates),
            checkpoints=[],
            step_count=self.step_count,
        )


@dataclass
class TrajectoryResult:
    """Final run artifact persisted by the experiment driver."""

    task_id: str
    success: bool
    score: float
    final_output: Any
    step_count: int
    total_wall_time_sec: float
    termination_reason: str
    model_name: str
    harness_name: str
    grade_result: GradeResult
    steps: list[Step] = field(default_factory=list)
    harness_metrics: dict[str, Any] = field(default_factory=dict)
    # Cost aggregates
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost: float = 0.0
    api_calls: int = 0


class Harness(ABC):
    """Base class shared by H1, H2, and H3."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.name = self.config["name"]
        self.version = self.config["version"]
        self._last_verify_usage: Optional[UsageStats] = None

    def _load_config(self, path: Path) -> dict[str, Any]:
        cfg = load_mapping_file(path)
        parent_name = cfg.get("inherits")
        if parent_name:
            parent_path = path.parent / f"{parent_name}{path.suffix}"
            parent_cfg = self._load_config(parent_path)
            cfg = deep_merge(parent_cfg, {k: v for k, v in cfg.items() if k != "inherits"})
        return cfg

    def run(self, model: Model, runtime: TaskRuntime) -> TrajectoryResult:
        """Execute one trajectory with one model and one harness."""
        state = AgentState(
            task_id=runtime.task_id,
            goal=runtime.goal,
            persistent_memory={
                "working_directory": str(runtime.working_directory),
                "retry_count": 0,
                "parse_retry_count": 0,
            },
        )
        started_at = time.time()
        termination_reason = "max_steps_reached"
        final_output: Any = None
        grade_result = GradeResult(success=False, score=0.0)
        max_steps = int(self.config["orchestration"]["max_steps"])
        system_prompt = self.system_prompt(runtime)

        try:
            for step_idx in range(max_steps):
                state.step_count = step_idx
                self._maybe_checkpoint(state)
                state = self._maybe_handle_drift(state)

                context = self._clip_context_to_limit(
                    self.build_context(state, runtime),
                    int(self.config["context"]["max_tokens"]),
                )
                prompt_tokens = approx_token_count(context)
                state.prompt_token_estimates.append(prompt_tokens)

                try:
                    response = model.generate(
                        context,
                        system_prompt=system_prompt,
                        max_output_tokens=self.config.get("model", {}).get("max_output_tokens"),
                    )
                except Exception as exc:  # pragma: no cover - integration path
                    termination_reason = f"model_error_{type(exc).__name__}"
                    state.persistent_memory["last_model_error"] = str(exc)
                    break

                raw_output = response.text
                action, parse_error = self.validate_action(raw_output, runtime)

                # If model output has BOTH a valid tool call AND a terminal
                # marker, prefer executing the tool call first. Only treat as
                # terminal if there is no valid tool call.
                is_terminal = self._is_terminal_output(raw_output)

                if is_terminal and (action is None or parse_error is not None):
                    # Terminal output with no valid tool call → submission attempt.
                    # Only reject empty patches if output_validation is enabled
                    # (H2/H3). H1 (output_validation=false) allows empty submissions.
                    output_validation = self.config.get("verification", {}).get("output_validation", False)
                    has_changes = self._check_has_changes(runtime) if output_validation else True
                    if not has_changes:
                        empty_reject_count = state.persistent_memory.get("empty_reject_count", 0) + 1
                        state.persistent_memory["empty_reject_count"] = empty_reject_count
                        state.persistent_memory["last_premature_final"] = truncate_text(raw_output, 1_200)
                        if empty_reject_count >= 5:
                            termination_reason = "refused_without_progress"
                            break
                        no_edit_obs = Observation(
                            output=(
                                "SUBMISSION REJECTED: No code changes detected (git diff is empty). "
                                "You MUST edit source files to fix the bug before submitting. "
                                "Use replace_in_file or write_file to make changes, then try again."
                            ),
                            error=ErrorType.UNKNOWN,
                            error_message="empty_patch_rejected",
                            tool_name="_harness_check",
                        )
                        step = Step(
                            step_id=state.step_count,
                            action=ToolCall(name="_submit_rejected", arguments={}),
                            observation=no_edit_obs,
                            raw_model_output=raw_output,
                            prompt_tokens_estimate=prompt_tokens,
                            model_usage=response.usage,
                        )
                        state.history.append(step)
                        continue
                    final_output = self._extract_final(raw_output)
                    termination_reason = "task_complete"
                    break

                if is_terminal and action is not None:
                    # Has both tool call and <final> — execute tool first,
                    # will check terminal again on the next iteration
                    pass

                if parse_error is not None or action is None:
                    parse_retries = state.persistent_memory.get("parse_retry_count", 0) + 1
                    state.persistent_memory["parse_retry_count"] = parse_retries
                    synthetic = Observation(
                        output=truncate_text(raw_output, 1_200),
                        error=parse_error or ErrorType.MALFORMED_OUTPUT,
                        error_message="model output could not be parsed into a tool call",
                        tool_name="_parser",
                    )
                    step = Step(
                        step_id=state.step_count,
                        action=ToolCall(name="_parse_error", arguments={}),
                        observation=synthetic,
                        raw_model_output=raw_output,
                        prompt_tokens_estimate=prompt_tokens,
                        model_usage=response.usage,
                        wall_time_sec=time.time() - started_at,
                    )
                    state = self.update_state(state, step)
                    if self.should_retry(synthetic, attempt=0) and parse_retries < 10:
                        continue
                    termination_reason = f"parse_error_{synthetic.error.value}"
                    break

                if not self._tool_is_available(runtime, action.name):
                    obs = Observation(
                        output="",
                        error=ErrorType.PERMISSION_DENIED,
                        error_message=f"tool not available in this task: {action.name}",
                        tool_name=action.name,
                    )
                else:
                    obs = self._execute_with_retry(action, runtime, state)

                anomaly = self.check_anomaly(state, action, obs)
                if anomaly is not None:
                    state, obs = self._handle_anomaly(state, obs, anomaly)

                filtered = self.filter_observation(obs)
                verification = self.verify_step(state, action, filtered, model, runtime)

                # Capture verification usage if verify_step stored one
                v_usage = self._last_verify_usage
                self._last_verify_usage = None

                step = Step(
                    step_id=step_idx,
                    action=action,
                    observation=filtered,
                    raw_model_output=raw_output,
                    prompt_tokens_estimate=prompt_tokens,
                    verification=verification,
                    model_usage=response.usage,
                    verification_usage=v_usage,
                    wall_time_sec=time.time() - started_at,
                )
                state = self.update_state(state, step)
                # Reset rejection counter on successful tool execution
                state.persistent_memory.pop("empty_reject_count", None)

                if (
                    filtered.error is not None
                    and self.config["orchestration"]["escalation_policy"] == "terminate"
                ):
                    termination_reason = f"unrecoverable_error_{filtered.error.value}"
                    break

            if final_output is None and state.history:
                final_output = state.history[-1].observation.output

            try:
                grade_result = runtime.grade(final_output, state)
            except Exception as exc:  # pragma: no cover - integration path
                grade_result = GradeResult(
                    success=False,
                    score=0.0,
                    stderr=str(exc),
                    details={"grade_error": str(exc)},
                )
        finally:
            runtime.close()

        # Aggregate cost/token stats from all steps (action + verification)
        _input = _output = _cached = _calls = 0
        _cost = 0.0
        for s in state.history:
            if s.model_usage:
                _input += s.model_usage.input_tokens or 0
                _output += s.model_usage.output_tokens or 0
                _cached += s.model_usage.cached_tokens or 0
                _cost += s.model_usage.cost or 0.0
                _calls += 1
            if s.verification_usage:
                _input += s.verification_usage.input_tokens or 0
                _output += s.verification_usage.output_tokens or 0
                _cached += s.verification_usage.cached_tokens or 0
                _cost += s.verification_usage.cost or 0.0
                _calls += 1

        return TrajectoryResult(
            task_id=runtime.task_id,
            success=grade_result.success,
            score=grade_result.score,
            final_output=final_output,
            step_count=len(state.history),
            total_wall_time_sec=time.time() - started_at,
            termination_reason=termination_reason,
            model_name=model.name,
            harness_name=self.name,
            grade_result=grade_result,
            steps=list(state.history),
            harness_metrics=self._collect_metrics(state),
            total_input_tokens=_input,
            total_output_tokens=_output,
            total_cached_tokens=_cached,
            total_cost=_cost,
            api_calls=_calls,
        )

    def _maybe_checkpoint(self, state: AgentState) -> None:
        if not self.config["observability"].get("checkpoint", False):
            return
        state.checkpoints.append(state.snapshot())
        retention = int(self.config["observability"].get("checkpoint_retention", 10))
        if len(state.checkpoints) > retention:
            state.checkpoints.pop(0)

    def _maybe_handle_drift(self, state: AgentState) -> AgentState:
        drift_cfg = self.config["context"].get("drift_check", {})
        if not drift_cfg.get("enabled", False):
            return state
        interval = int(drift_cfg.get("check_interval", 0))
        if state.step_count == 0 or interval <= 0 or state.step_count % interval != 0:
            return state
        if self._detect_drift(state, float(drift_cfg.get("threshold_kl", 0.0))):
            return self._handle_drift(state)
        return state

    def _clip_context_to_limit(self, context: str, max_tokens: int) -> str:
        char_budget = max_tokens * 4
        if len(context) <= char_budget:
            return context
        clipped = context[-char_budget:]
        return "[context truncated by harness]\n" + clipped

    def _tool_is_available(self, runtime: TaskRuntime, tool_name: str) -> bool:
        tools = runtime.list_tool_specs(schema_style="minimal", exposure="all")
        return any(tool.name == tool_name for tool in tools)

    def _execute_with_retry(
        self,
        action: ToolCall,
        runtime: TaskRuntime,
        state: AgentState,
    ) -> Observation:
        retry_cfg = self.config["orchestration"].get("retry", {})
        if not self.config["orchestration"].get("retry_on_failure", False):
            return runtime.execute_tool(action)

        max_attempts = int(retry_cfg.get("max_attempts", 1))
        base_delay_sec = float(retry_cfg.get("base_delay_sec", 0.0))
        retryable = set(retry_cfg.get("retryable_errors", []))

        last_obs: Optional[Observation] = None
        for attempt in range(max_attempts):
            started_at = time.time()
            obs = runtime.execute_tool(action)
            obs.tool_name = action.name
            obs.wall_time_sec = time.time() - started_at
            last_obs = obs
            if obs.error is None:
                return obs
            if obs.error.value not in retryable:
                return obs
            if not self.should_retry(obs, attempt):
                return obs
            state.persistent_memory["retry_count"] += 1
            if base_delay_sec > 0:
                time.sleep(base_delay_sec * (2**attempt))
        return last_obs or Observation(output="", error=ErrorType.UNKNOWN)

    @abstractmethod
    def build_context(self, state: AgentState, runtime: TaskRuntime) -> str:
        """Construct the current model-visible context."""

    @abstractmethod
    def validate_action(
        self,
        raw_output: str,
        runtime: TaskRuntime,
    ) -> tuple[Optional[ToolCall], Optional[ErrorType]]:
        """Parse a model output into a structured tool call."""

    @abstractmethod
    def filter_observation(self, obs: Observation) -> Observation:
        """Filter or sanitize an observation before adding it to state."""

    def system_prompt(self, runtime: TaskRuntime) -> str:
        """Default system prompt shared across harnesses (mini-SWE-agent style)."""
        return (
            "You are a helpful assistant that can interact with a computer "
            "workspace to solve programming tasks."
        )

    def check_anomaly(
        self,
        state: AgentState,
        action: ToolCall,
        obs: Observation,
    ) -> Optional[str]:
        """Anomaly detection hook. H3 overrides this."""
        return None

    def verify_step(
        self,
        state: AgentState,
        action: ToolCall,
        obs: Observation,
        model: Model,
        runtime: TaskRuntime,
    ) -> Optional[str]:
        """Self-verification hook. H3 overrides this."""
        return None

    def should_retry(self, obs: Observation, attempt: int) -> bool:
        """Retry policy hook. H2 and H3 override this."""
        return False

    def update_state(self, state: AgentState, step: Step) -> AgentState:
        """Default state update: append step and remember latest observation."""
        state.history.append(step)
        state.persistent_memory["last_observation"] = step.observation.output
        return state

    def _detect_drift(self, state: AgentState, threshold: float) -> bool:
        """Drift detection hook. H3 overrides this."""
        return False

    def _handle_drift(self, state: AgentState) -> AgentState:
        """Drift recovery hook. H3 overrides this."""
        return state

    def _handle_anomaly(
        self,
        state: AgentState,
        obs: Observation,
        anomaly: str,
    ) -> tuple[AgentState, Observation]:
        """Default anomaly response: annotate the observation only."""
        obs.error_message = f"anomaly_detected:{anomaly}"
        return state, obs

    TERMINAL_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

    @staticmethod
    def _check_has_changes(runtime) -> bool:
        """Check if the runtime has any uncommitted file changes."""
        try:
            obs = runtime.execute_tool(ToolCall(name="git_diff", arguments={}))
            return bool(obs.output and obs.output.strip())
        except Exception:
            return False

    @classmethod
    def _is_terminal_output(cls, raw_output: str) -> bool:
        if cls.TERMINAL_SENTINEL in raw_output:
            return True
        return "<final>" in raw_output or "```final" in raw_output

    @classmethod
    def _extract_final(cls, raw_output: str) -> str:
        if cls.TERMINAL_SENTINEL in raw_output:
            head, _, tail = raw_output.partition(cls.TERMINAL_SENTINEL)
            return tail.strip() or head.strip()
        if "<final>" in raw_output:
            return raw_output.split("<final>", 1)[1].split("</final>", 1)[0].strip()
        if "```final" in raw_output:
            return raw_output.split("```final", 1)[1].split("```", 1)[0].strip()
        return raw_output.strip()

    def _collect_metrics(self, state: AgentState) -> dict[str, Any]:
        history = state.history
        total_steps = len(history)
        error_indices = [idx for idx, step in enumerate(history) if step.observation.error]
        recovered = 0
        for idx in error_indices:
            if any(next_step.observation.error is None for next_step in history[idx + 1 :]):
                recovered += 1

        productive_steps = sum(1 for step in history if step.observation.error is None)
        verification_revisions = sum(
            1
            for step in history
            if step.verification is not None and step.verification.startswith("REVISE")
        )
        anomalies = sum(
            1
            for step in history
            if step.observation.error_message
            and step.observation.error_message.startswith("anomaly_detected:")
        )

        context_cap = int(self.config["context"]["max_tokens"])
        avg_context_tokens = (
            sum(state.prompt_token_estimates) / len(state.prompt_token_estimates)
            if state.prompt_token_estimates
            else 0.0
        )

        return {
            "harness_name": self.name,
            "harness_version": self.version,
            "n_steps": total_steps,
            "n_errors": len(error_indices),
            "n_retries": int(state.persistent_memory.get("retry_count", 0)),
            "n_parse_retries": int(state.persistent_memory.get("parse_retry_count", 0)),
            "n_verifications": sum(1 for step in history if step.verification is not None),
            "verification_revision_rate": (
                verification_revisions / total_steps if total_steps else 0.0
            ),
            "recovery_rate": recovered / len(error_indices) if error_indices else 1.0,
            "action_efficiency": productive_steps / total_steps if total_steps else 0.0,
            "avg_context_tokens_estimate": avg_context_tokens,
            "avg_context_utilization": (
                avg_context_tokens / context_cap if context_cap else 0.0
            ),
            "anomaly_count": anomalies,
        }
