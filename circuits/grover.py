"""
Grover's Search Algorithm
=========================
Builds a Grover circuit for a given oracle (marked state).

The oracle is specified as a list of target bitstrings (e.g. ["101", "011"]).
The diffuser (inversion about the mean) is appended automatically.
"""

import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator
from config import GROVER_DEFAULT_QUBITS, GROVER_MAX_QUBITS


def build_grover_circuit(
    target_states: list[str],
    n_qubits: int = GROVER_DEFAULT_QUBITS,
    n_iterations: int | None = None,
) -> QuantumCircuit:
    """
    Build a Grover search circuit.

    Args:
        target_states: List of bitstrings to mark (e.g. ["101"]).
                       All strings must have length == n_qubits.
        n_qubits:      Number of qubits (search space = 2^n_qubits).
        n_iterations:  Number of Grover iterations. Defaults to optimal
                       floor(pi/4 * sqrt(N/M)) where N=2^n, M=len(targets).

    Returns:
        QuantumCircuit with measurement on all qubits.

    Raises:
        ValueError: on invalid inputs.
    """
    if n_qubits < 1 or n_qubits > GROVER_MAX_QUBITS:
        raise ValueError(
            f"n_qubits must be between 1 and {GROVER_MAX_QUBITS}, got {n_qubits}"
        )
    if not target_states:
        raise ValueError("target_states must be non-empty")
    for state in target_states:
        if len(state) != n_qubits or not all(b in "01" for b in state):
            raise ValueError(
                f"Each target state must be a bitstring of length {n_qubits}, got '{state}'"
            )

    n_targets = len(target_states)
    n_states = 2 ** n_qubits

    if n_iterations is None:
        n_iterations = max(1, math.floor(math.pi / 4 * math.sqrt(n_states / n_targets)))

    # Build oracle: phase-flip the marked states
    oracle = _build_oracle(target_states, n_qubits)

    # Grover operator = oracle + diffuser
    grover_op = grover_operator(oracle)

    # Full circuit: uniform superposition → Grover iterations → measure
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        qc.compose(grover_op, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


def _build_oracle(target_states: list[str], n_qubits: int) -> QuantumCircuit:
    """
    Phase-flip oracle: applies -1 phase to each target state.
    Uses a multi-controlled Z construction.
    """
    oracle = QuantumCircuit(n_qubits)

    for state in target_states:
        # Flip qubits where the target bit is 0, apply mcz, flip back
        zero_positions = [i for i, bit in enumerate(reversed(state)) if bit == "0"]
        if zero_positions:
            oracle.x(zero_positions)
        # Multi-controlled Z: phase flip when all qubits are |1>
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)
        if zero_positions:
            oracle.x(zero_positions)

    return oracle


def grover_circuit_info(qc: QuantumCircuit) -> dict:
    """Return a summary dict describing the circuit."""
    return {
        "n_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "gate_count": qc.size(),
        "circuit_str": str(qc.draw(output="text")),
    }
