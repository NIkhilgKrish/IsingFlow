"""
QAOA for QUBO / Ising Problems
================================
Converts a QUBO matrix to an Ising Hamiltonian and builds a QAOA circuit.

QUBO form:   minimize  x^T Q x        (x ∈ {0,1}^n)
Ising form:  minimize  sum_ij J_ij s_i s_j + sum_i h_i s_i   (s ∈ {-1,+1}^n)

The mapping x_i = (1 - s_i) / 2 converts between the two.
This is the same mathematical structure as NVIDIA's Ising solver targets.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from config import (
    QAOA_DEFAULT_REPS,
    QAOA_MAX_REPS,
    QAOA_DEFAULT_QUBITS,
    QAOA_MAX_QUBITS,
)


# ---------------------------------------------------------------------------
# QUBO → Ising conversion
# ---------------------------------------------------------------------------

def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert a QUBO matrix Q to Ising coefficients (J, h, offset).

    Args:
        Q: Upper-triangular or symmetric (n x n) QUBO matrix.

    Returns:
        J:      (n x n) coupling matrix (off-diagonal Ising interactions).
        h:      (n,) local field vector.
        offset: Constant energy offset.
    """
    Q = np.array(Q, dtype=float)
    # Symmetrize
    Q = (Q + Q.T) / 2
    n = Q.shape[0]

    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0.0

    for i in range(n):
        h[i] += Q[i, i] / 2
        offset += Q[i, i] / 2
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4
            h[i] += Q[i, j] / 4
            h[j] += Q[i, j] / 4
            offset += Q[i, j] / 4

    return J, h, offset


def ising_energy(J: np.ndarray, h: np.ndarray, bitstring: str) -> float:
    """
    Compute the Ising energy for a given spin configuration.

    Args:
        J:         (n x n) coupling matrix.
        h:         (n,) local field vector.
        bitstring: Binary string, e.g. "1001". '0' → s=+1, '1' → s=-1.

    Returns:
        Energy as a float.
    """
    s = np.array([1 - 2 * int(b) for b in bitstring], dtype=float)
    # Ising Hamiltonion @ spin state give energy. Basic QM
    energy = float(h @ s + s @ J @ s)
    return energy


# ---------------------------------------------------------------------------
# QAOA circuit
# ---------------------------------------------------------------------------

def build_qaoa_circuit(
    Q: np.ndarray,
    reps: int = QAOA_DEFAULT_REPS,
) -> tuple[QuantumCircuit, np.ndarray, np.ndarray, float]:
    """
    Build a QAOA circuit for a QUBO problem.

    Args:
        Q:    (n x n) QUBO matrix.
        reps: Number of QAOA layers (p parameter). Higher → better approximation.

    Returns:
        qc:     Parameterized QuantumCircuit (2*reps free parameters: gamma, beta).
        J:      Ising coupling matrix derived from Q.
        h:      Ising local fields derived from Q.
        offset: Constant energy offset.

    Raises:
        ValueError: on invalid inputs.
    """
    n = Q.shape[0]
    if n < 2 or n > QAOA_MAX_QUBITS:
        raise ValueError(f"Problem size must be between 2 and {QAOA_MAX_QUBITS}, got {n}")
    if reps < 1 or reps > QAOA_MAX_REPS:
        raise ValueError(f"reps must be between 1 and {QAOA_MAX_REPS}, got {reps}")

    J, h, offset = qubo_to_ising(Q)

    gamma = ParameterVector("γ", reps)  # cost layer angles
    beta = ParameterVector("β", reps)   # mixer layer angles

    qc = QuantumCircuit(n)

    # Initial state: uniform superposition
    qc.h(range(n))

    for p in range(reps):
        # --- Cost unitary: exp(-i * gamma_p * H_C) ---
        # ZZ interactions from J
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2 * gamma[p] * J[i, j], j)
                    qc.cx(i, j)
        # Z terms from h
        for i in range(n):
            if abs(h[i]) > 1e-10:
                qc.rz(2 * gamma[p] * h[i], i)

        # --- Mixer unitary: exp(-i * beta_p * H_B) ---
        for i in range(n):
            qc.rx(2 * beta[p], i)

    qc.measure_all()
    return qc, J, h, offset


def bind_qaoa_parameters(
    qc: QuantumCircuit,
    gamma_vals: list[float],
    beta_vals: list[float],
) -> QuantumCircuit:
    """
    Bind concrete angle values to a parameterized QAOA circuit.

    Args:
        qc:         Parameterized QAOA circuit from build_qaoa_circuit.
        gamma_vals: Cost layer angles, length == reps.
        beta_vals:  Mixer layer angles, length == reps.

    Returns:
        Bound (non-parameterized) QuantumCircuit ready for execution.
    """
    params = qc.parameters
    gamma_params = sorted([p for p in params if "γ" in p.name], key=lambda p: p.name)
    beta_params = sorted([p for p in params if "β" in p.name], key=lambda p: p.name)

    if len(gamma_vals) != len(gamma_params) or len(beta_vals) != len(beta_params):
        raise ValueError(
            f"Expected {len(gamma_params)} gamma and {len(beta_params)} beta values, "
            f"got {len(gamma_vals)} and {len(beta_vals)}"
        )

    binding = {p: v for p, v in zip(gamma_params, gamma_vals)}
    binding.update({p: v for p, v in zip(beta_params, beta_vals)})
    return qc.assign_parameters(binding)


def qaoa_circuit_info(qc: QuantumCircuit) -> dict:
    """Return a summary dict describing the circuit."""
    return {
        "n_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "gate_count": qc.size(),
        "n_parameters": qc.num_parameters,
        "circuit_str": str(qc.draw(output="text")),
    }
