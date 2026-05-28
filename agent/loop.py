"""
IsingFlow ReAct Agent Loop
===========================
A minimal Reason → Act → Observe loop.

The LLM reasons about what tool to call next, we execute the tool,
feed the observation back, and repeat until the LLM emits FINAL ANSWER.

No framework required — the full loop is ~80 lines.
"""

from __future__ import annotations

import json
import re
import anthropic

from agent.prompts import SYSTEM_PROMPT, TOOL_CALL_FORMAT, REACT_PROMPT_TEMPLATE
from tools.design import design_circuit
from tools.optimize import optimize_circuit
from tools.simulate import simulate_circuit
from tools.hardware import execute_on_hardware
from tools.analyze import analyze_results
from config import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    AGENT_MAX_ITERATIONS,
    AGENT_STOP_TOKEN,
)

# ---------------------------------------------------------------------------
# Tool registry — maps name → callable
# ---------------------------------------------------------------------------
TOOLS: dict[str, callable] = {
    "design_circuit": design_circuit,
    "optimize_circuit": optimize_circuit,
    "simulate_circuit": simulate_circuit,
    "execute_on_hardware": execute_on_hardware,
    "analyze_results": analyze_results,
}

# ---------------------------------------------------------------------------
# Agent state passed between iterations
# ---------------------------------------------------------------------------
_CONTEXT: dict = {}  # stores intermediate tool outputs by tool name


def run(user_request: str, verbose: bool = True) -> str:
    """
    Run the IsingFlow agent on a user request.

    Args:
        user_request: Natural language problem description.
        verbose:      If True, print each Thought/Action/Observation step.

    Returns:
        The agent's final answer as a string.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages = [
        {
            "role": "user",
            "content": REACT_PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                tool_call_format=TOOL_CALL_FORMAT,
                user_request=user_request,
            ),
        }
    ]

    for iteration in range(AGENT_MAX_ITERATIONS):
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=messages,
        )

        llm_output = response.content[0].text

        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"THOUGHT/ACTION:\n{llm_output}")

        # Check for final answer
        if AGENT_STOP_TOKEN in llm_output:
            final = llm_output.split(AGENT_STOP_TOKEN, 1)[1].strip()
            if verbose:
                print(f"\nFINAL ANSWER:\n{final}")
            return final

        # Parse tool call
        tool_name, tool_input = _parse_action(llm_output)
        if tool_name is None:
            observation = (
                "ERROR: Could not parse a valid ACTION from your output. "
                "Please output 'ACTION: <tool_name>' followed by 'INPUT: <json>'."
            )
        else:
            observation = _call_tool(tool_name, tool_input, verbose)

        if verbose:
            print(f"OBSERVATION: {_summarize(observation)}")

        # Append assistant turn + observation
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"OBSERVATION: {json.dumps(observation, default=str)}"})

    return "Agent reached maximum iterations without a final answer."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> tuple[str | None, dict]:
    """Extract tool name and JSON input from LLM output."""
    action_match = re.search(r"ACTION:\s*(\w+)", text)
    input_match = re.search(r"INPUT:\s*(\{.*?\})", text, re.DOTALL)

    if not action_match:
        return None, {}

    tool_name = action_match.group(1).strip()
    tool_input = {}
    if input_match:
        try:
            tool_input = json.loads(input_match.group(1))
        except json.JSONDecodeError as e:
            return tool_name, {"_parse_error": str(e)}

    return tool_name, tool_input


def _call_tool(tool_name: str, tool_input: dict, verbose: bool) -> dict | str:
    """Execute a tool, threading context automatically."""
    if tool_name not in TOOLS:
        return f"ERROR: Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"

    if verbose:
        print(f"ACTION: {tool_name}({json.dumps(tool_input, default=str)})")

    try:
        fn = TOOLS[tool_name]

        # Thread context: downstream tools need outputs of upstream tools
        if tool_name == "optimize_circuit":
            result = fn(_CONTEXT.get("design_circuit", tool_input))
        elif tool_name == "simulate_circuit":
            result = fn(_CONTEXT.get("optimize_circuit", tool_input))
        elif tool_name == "execute_on_hardware":
            use_fake = tool_input.get("use_fake", False)
            result = fn(_CONTEXT.get("optimize_circuit", tool_input), use_fake=use_fake)
        elif tool_name == "analyze_results":
            # Use simulation or hardware result, whichever is available
            exec_result = _CONTEXT.get("execute_on_hardware") or _CONTEXT.get("simulate_circuit")
            if exec_result is None:
                return "ERROR: analyze_results requires simulate_circuit or execute_on_hardware to run first."
            # Propagate circuit_type into metadata for dispatcher
            if "circuit_type" in _CONTEXT.get("design_circuit", {}):
                exec_result["metadata"]["circuit_type"] = _CONTEXT["design_circuit"]["circuit_type"]
            result = fn(exec_result)
        else:
            result = fn(tool_input)

        _CONTEXT[tool_name] = result

        # Strip QuantumCircuit objects from observation (not serializable / not useful)
        return _strip_circuits(result)

    except Exception as e:
        return f"ERROR in {tool_name}: {type(e).__name__}: {e}"


def _strip_circuits(obj):
    """Recursively remove QuantumCircuit objects from a dict for JSON serialization."""
    from qiskit import QuantumCircuit
    if isinstance(obj, dict):
        return {k: _strip_circuits(v) for k, v in obj.items() if not isinstance(v, QuantumCircuit)}
    if isinstance(obj, list):
        return [_strip_circuits(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _summarize(observation) -> str:
    """Short summary of observation for verbose printing."""
    if isinstance(observation, str):
        return observation[:200]
    if isinstance(observation, dict):
        keys = list(observation.keys())
        return f"dict with keys: {keys}"
    return str(observation)[:200]


# Delayed import to avoid circular at module level
import numpy as np
