"""
Tool: design_circuit
=====================
Given a problem description dict from the agent, builds the appropriate
quantum circuit and returns it alongside metadata.

Supported problem types:
  - "grover":  unstructured search
  - "qubo":    QUBO/Ising optimization via QAOA
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from circuits.grover import build_grover_circuit, grover_circuit_info
from circuits.qaoa_qubo import build_qaoa_circuit, qaoa_circuit_info
from config import GROVER_DEFAULT_QUBITS, QAOA_DEFAULT_REPS


def design_circuit(problem: dict) -> dict:
    """
    Entry point for the agent's design_circuit tool.

    Args:
        problem: Dict with required key "type" and type-specific keys:

            For "grover":
                "target_states": list[str]  — bitstrings to search for
                "n_qubits":      int        — optional, defaults to config

            For "qubo":
                "Q":    list[list[float]]   — QUBO matrix (n x n)
                "reps": int                 — optional QAOA depth, defaults to config

    Returns:
        {
            "circuit":      QuantumCircuit,
            "circuit_type": str,
            "info":         dict,           # depth, gate count, etc.
            "metadata":     dict,           # problem-specific data for downstream tools
        }

    Raises:
        ValueError: on unknown problem type or missing keys.
    """
    problem_type = problem.get("type", "").lower()

    if problem_type == "grover":
        return _design_grover(problem)
    elif problem_type == "qubo":
        return _design_qubo(problem)
    else:
        raise ValueError(
            f"Unknown problem type '{problem_type}'. Supported: 'grover', 'qubo'."
        )


def _design_grover(problem: dict) -> dict:
    target_states = problem.get("target_states")
    if not target_states:
        raise ValueError("Grover problem requires 'target_states' list.")

    n_qubits = problem.get("n_qubits", len(target_states[0]))
    qc = build_grover_circuit(target_states=target_states, n_qubits=n_qubits)
    info = grover_circuit_info(qc)

    return {
        "circuit": qc,
        "circuit_type": "grover",
        "info": info,
        "metadata": {
            "target_states": target_states,
            "n_qubits": n_qubits,
        },
    }


def _design_qubo(problem: dict) -> dict:
    Q_raw = problem.get("Q")
    if Q_raw is None:
        raise ValueError("QUBO problem requires 'Q' matrix.")

    Q = np.array(Q_raw, dtype=float)
    reps = problem.get("reps", QAOA_DEFAULT_REPS)
    qc, J, h, offset = build_qaoa_circuit(Q=Q, reps=reps)
    info = qaoa_circuit_info(qc)

    return {
        "circuit": qc,
        "circuit_type": "qubo",
        "info": info,
        "metadata": {
            "Q": Q,
            "J": J,
            "h": h,
            "offset": offset,
            "reps": reps,
            "n_qubits": Q.shape[0],
        },
    }
