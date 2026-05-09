"""H3 full harness: adds verification, anomaly detection, and rollback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Optional

from ..core import AgentState, ErrorType, Model, Observation, TaskRuntime, ToolCall
from .h2_improved import H2Improved, H2ImprovedV2


class H3Full(H2Improved):
    """Full closed-loop harness with explicit stability controls."""

    def verify_step(
        self,
        state: AgentState,
        action: ToolCall,
        obs: Observation,
        model: Model,
        runtime: TaskRuntime,
    ) -> Optional[str]:
        if not self.config["verification"].get("self_check", False):
            return None
        # Truncate observation for verification prompt
        obs_text = str(obs.output or "")
        if len(obs_text) > 1500:
            obs_text = obs_text[:750] + "\n[...truncated...]\n" + obs_text[-750:]
        verification_prompt = (
            "Review whether this tool action made progress toward the goal.\n"
            f"Goal summary: Fix the bug described in the task.\n"
            f"Action: {action.name} {json.dumps(action.arguments, ensure_ascii=True)[:300]}\n"
            f"Observation: {obs_text}\n"
            f"Error: {obs.error.value if obs.error else 'none'}\n\n"
            "Respond VERIFIED if the action succeeded or made reasonable progress "
            "(including exploration steps like reading files or running commands). "
            "Respond REVISE only if the action clearly failed or was destructive.\n"
            "One word then one short sentence."
        )
        try:
            response = model.generate(
                verification_prompt,
                system_prompt="You are a verification assistant. Default to VERIFIED unless there is a clear problem.",
                max_output_tokens=64,
            )
            # Store verification cost so the main loop can attach it to Step
            self._last_verify_usage = response.usage
            text = response.text.strip()
            if text:
                return text
        except Exception:
            pass
        # Default to VERIFIED — err on the side of not blocking
        return "VERIFIED"

    def check_anomaly(
        self,
        state: AgentState,
        action: ToolCall,
        obs: Observation,
    ) -> Optional[str]:
        detectors = self.config["verification"]["anomaly_detection"].get("detectors", [])
        # NOTE: "malformed_json" is intentionally not checked against tool
        # observations — tool outputs are file contents/command output and
        # routinely contain `{` without being JSON. This detector fires on
        # model-emitted tool calls, which are already validated upstream by
        # validate_action, so there's nothing for us to add here.
        if "empty_tool_output" in detectors and obs.output in (None, "", [], {}) and obs.error is None:
            return "empty_tool_output"
        if "repeated_action_loop" in detectors and self._is_repeated_action(state, action):
            return "repeated_action_loop"
        if "context_contradiction" in detectors and self._detects_contradiction(state, obs):
            return "context_contradiction"
        return None

    def _handle_anomaly(
        self,
        state: AgentState,
        obs: Observation,
        anomaly: str,
    ) -> tuple[AgentState, Observation]:
        obs.error = ErrorType.MALFORMED_OUTPUT if anomaly == "malformed_json" else ErrorType.UNKNOWN
        obs.error_message = f"anomaly_detected:{anomaly}"
        if (
            self.config["orchestration"].get("escalation_policy") == "checkpoint_rollback"
            and state.checkpoints
        ):
            depth = min(
                int(self.config["orchestration"].get("rollback_depth", 1)),
                len(state.checkpoints),
            )
            restored = state.checkpoints[-depth].snapshot()
            restored.checkpoints = list(state.checkpoints[:-depth])
            restored.persistent_memory["last_rollback_reason"] = anomaly
            return restored, obs
        return state, obs

    def _detect_drift(self, state: AgentState, threshold: float) -> bool:
        if len(state.history) < 6:
            return False
        early = state.history[:3]
        recent = state.history[-3:]
        early_hist = self._token_histogram(early)
        recent_hist = self._token_histogram(recent)
        return self._kl_divergence(recent_hist, early_hist) > threshold

    def _handle_drift(self, state: AgentState) -> AgentState:
        state.persistent_memory["drift_reset"] = {
            "step": state.step_count,
            "note": f"Context drift detected at step {state.step_count}. Re-anchor to the original goal.",
        }
        return state

    @staticmethod
    def _looks_like_malformed_json(output: object) -> bool:
        if not isinstance(output, str):
            return False
        if "{" not in output and "[" not in output:
            return False
        try:
            json.loads(output)
            return False
        except json.JSONDecodeError:
            return True

    @staticmethod
    def _is_repeated_action(state: AgentState, action: ToolCall) -> bool:
        if len(state.history) < 2:
            return False
        recent = state.history[-2:]
        signature = H3Full._action_signature(action)
        return all(H3Full._action_signature(step.action) == signature for step in recent)

    @staticmethod
    def _action_signature(action: ToolCall) -> str:
        payload = json.dumps({"tool": action.name, "args": action.arguments}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _detects_contradiction(state: AgentState, obs: Observation) -> bool:
        if not state.history or not isinstance(obs.output, str):
            return False
        last_output = str(state.history[-1].observation.output)
        if "success" in last_output.lower() and "error" in obs.output.lower():
            return True
        return False

    @staticmethod
    def _token_histogram(steps) -> Counter:
        tokens: list[str] = []
        for step in steps:
            text = f"{step.action.name} {step.observation.output}"
            tokens.extend(text.lower().split())
        return Counter(tokens)

    @staticmethod
    def _kl_divergence(p: Counter, q: Counter) -> float:
        vocab = set(p.keys()) | set(q.keys())
        if not vocab:
            return 0.0
        p_total = sum(p.values()) + len(vocab)
        q_total = sum(q.values()) + len(vocab)
        kl = 0.0
        for token in vocab:
            p_prob = (p.get(token, 0) + 1) / p_total
            q_prob = (q.get(token, 0) + 1) / q_total
            kl += p_prob * math.log(p_prob / q_prob)
        return kl


class H3FullV2(H3Full):
    """Current H3 policy with H2 anti-stall guidance and parser normalization."""

    def build_context(self, state: AgentState, runtime: TaskRuntime) -> str:
        base_context = super().build_context(state, runtime)
        v2_section = H2ImprovedV2._build_v2_policy_section(state).replace("H2", "H3")
        marker = "Next action (respond with a single JSON tool call):"
        if marker in base_context:
            return base_context.replace(marker, f"{v2_section}\n{marker}")
        return f"{base_context}\n{v2_section}"

    def system_prompt(self, runtime: TaskRuntime) -> str:
        return (
            super().system_prompt(runtime)
            + " This run uses H3 policy: use structured tool calls only, "
            "avoid excessive repeated searching/testing, and use H3 self-checks "
            "to validate a targeted source edit before submitting."
        )

    def validate_action(
        self,
        raw_output: str,
        runtime: TaskRuntime,
    ) -> tuple[Optional[ToolCall], Optional[ErrorType]]:
        action, parse_error = super().validate_action(raw_output, runtime)
        if action is not None:
            return action, None

        normalized = self._parse_structured_tool_output(raw_output, runtime)
        if normalized is not None:
            return normalized, None
        return None, parse_error

    def _handle_anomaly(
        self,
        state: AgentState,
        obs: Observation,
        anomaly: str,
    ) -> tuple[AgentState, Observation]:
        # Do not roll back past a source edit. For coding tasks, losing the
        # edit from model-visible history is more damaging than the anomaly.
        if any(
            step.action.name in {"replace_in_file", "write_file"}
            and step.observation.error is None
            for step in state.history
        ):
            obs.error = ErrorType.MALFORMED_OUTPUT if anomaly == "malformed_json" else ErrorType.UNKNOWN
            obs.error_message = f"anomaly_detected:{anomaly}"
            return state, obs
        return super()._handle_anomaly(state, obs, anomaly)

    @classmethod
    def _parse_structured_tool_output(
        cls,
        raw_output: str,
        runtime: TaskRuntime,
    ) -> Optional[ToolCall]:
        available_tools = {
            tool.name for tool in runtime.list_tool_specs(schema_style="minimal", exposure="all")
        }

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw_output):
            try:
                parsed, _ = decoder.raw_decode(raw_output[match.start() :])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            tool_call = cls._normalize_json_tool_call(parsed, available_tools)
            if tool_call is not None:
                return tool_call

        return cls._parse_xmlish_tool_call(raw_output, available_tools)

    @classmethod
    def _normalize_json_tool_call(
        cls,
        payload: object,
        available_tools: set[str],
    ) -> Optional[ToolCall]:
        if not isinstance(payload, dict):
            return None

        candidates: list[object] = [payload]
        for key in ("tool_call", "function_call", "action"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            candidates.extend(item for item in tool_calls if isinstance(item, dict))

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("tool") or candidate.get("name") or candidate.get("function")
            args = candidate.get("args", candidate.get("arguments", {}))
            if isinstance(args, str):
                try:
                    loaded_args = json.loads(args)
                    args = loaded_args if isinstance(loaded_args, dict) else {}
                except json.JSONDecodeError:
                    args = {}
            if isinstance(name, str) and name in available_tools and isinstance(args, dict):
                return ToolCall(name=name, arguments=cls._coerce_args(args))
        return None

    @classmethod
    def _parse_xmlish_tool_call(
        cls,
        raw_output: str,
        available_tools: set[str],
    ) -> Optional[ToolCall]:
        for block_match in re.finditer(
            r"<(?:tool|tool_call)>\s*(.*?)\s*</(?:tool|tool_call)>",
            raw_output,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            block = block_match.group(1).strip()
            tool_call = cls._parse_named_xml_tool(block, available_tools)
            if tool_call is not None:
                return tool_call
            tool_call = cls._parse_line_tool_block(block, available_tools)
            if tool_call is not None:
                return tool_call
        return None

    @classmethod
    def _parse_named_xml_tool(
        cls,
        block: str,
        available_tools: set[str],
    ) -> Optional[ToolCall]:
        name_match = re.search(r"<name>\s*([^<]+?)\s*</name>", block, flags=re.DOTALL | re.IGNORECASE)
        if not name_match:
            return None
        name = name_match.group(1).strip()
        if name not in available_tools:
            return None

        args: dict[str, object] = {}
        args_match = re.search(r"<args>\s*(.*?)\s*</args>", block, flags=re.DOTALL | re.IGNORECASE)
        args_block = args_match.group(1).strip() if args_match else block
        try:
            loaded = json.loads(args_block)
            if isinstance(loaded, dict):
                args.update(loaded)
        except json.JSONDecodeError:
            for key, value in re.findall(
                r"<([a-zA-Z_][\w-]*)>\s*(.*?)\s*</\1>",
                args_block,
                flags=re.DOTALL,
            ):
                if key != "name":
                    args[key] = value.strip()
        return ToolCall(name=name, arguments=cls._coerce_args(args))

    @classmethod
    def _parse_line_tool_block(
        cls,
        block: str,
        available_tools: set[str],
    ) -> Optional[ToolCall]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            return None
        name = lines[0]
        if name not in available_tools:
            return None
        args: dict[str, object] = {}
        current_key: Optional[str] = None
        for line in lines[1:]:
            key_match = re.match(r"^([a-zA-Z_][\w-]*)\s*:\s*(.*)$", line)
            if key_match:
                current_key = key_match.group(1)
                args[current_key] = key_match.group(2)
            elif current_key:
                args[current_key] = f"{args[current_key]}\n{line}"
        return ToolCall(name=name, arguments=cls._coerce_args(args))

    @staticmethod
    def _coerce_args(args: dict[str, object]) -> dict[str, object]:
        coerced: dict[str, object] = {}
        int_keys = {"start_line", "end_line", "max_entries", "count"}
        bool_keys = {"append", "staged"}
        aliases = {"cmd": "command"}
        for key, value in args.items():
            key = aliases.get(key, key)
            if isinstance(value, str):
                stripped = value.strip()
                if key in int_keys and re.fullmatch(r"-?\d+", stripped):
                    coerced[key] = int(stripped)
                elif key in bool_keys and stripped.lower() in {"true", "false"}:
                    coerced[key] = stripped.lower() == "true"
                else:
                    coerced[key] = stripped
            else:
                coerced[key] = value
        return coerced
