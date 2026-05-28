"""
Tool: optimize_circuit
=======================
Transpiles and optimizes a circuit for a target backend.
For simulation, uses a FakeBackend noise model.
For hardware, connects to the actual IBM Quantum backend.

Also handles QAOA parameter optimization via classical optimizer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from circuits.qaoa_qubo import bind_qaoa_parameters, ising_energy
from config import (
    TRANSPILE_OPTIMIZATION_LEVEL,
    TRANSPILE_SEED,
    QAOA_OPTIMIZER,
    QAOA_MAX_ITER,
    QAOA_TOLERANCE,
    SIM_SHOTS,
    SIM_SEED,
)


def optimize_circuit(design_result: dict, backend=None) -> dict:
    """
    Transpile a circuit for a target backend and, for QUBO circuits,
    run classical parameter optimization.

    Args:
        design_result: Output dict from design_circuit tool.
        backend:       Qiskit backend. None → use AerSimulator (no noise).

    Returns:
        {
            "circuit":          Optimized (bound, transpiled) QuantumCircuit,
            "transpile_stats":  dict,
            "optimization":     dict | None,  # QAOA only: best params + energy
            "metadata":         dict,         # pass-through from design_result
        }
    """
    circuit_type = design_result["circuit_type"]
    qc = design_result["circuit"]
    metadata = design_result["metadata"]

    if backend is None:
        backend = AerSimulator()

    if circuit_type == "qubo":
        opt_result = _optimize_qaoa_params(qc, metadata, backend)
        bound_qc = opt_result["bound_circuit"]
    else:
        bound_qc = qc
        opt_result = None

    transpiled = transpile(
        bound_qc,
        backend=backend,
        optimization_level=TRANSPILE_OPTIMIZATION_LEVEL,
        seed_transpiler=TRANSPILE_SEED,
    )

    transpile_stats = {
        "depth_before": bound_qc.depth(),
        "depth_after": transpiled.depth(),
        "gate_count_before": bound_qc.size(),
        "gate_count_after": transpiled.size(),
    }

    return {
        "circuit": transpiled,
        "transpile_stats": transpile_stats,
        "optimization": opt_result,
        "metadata": metadata,
    }


def _optimize_qaoa_params(
    qc: QuantumCircuit,
    metadata: dict,
    backend,
) -> dict:
    """
    Run classical optimizer to find best QAOA angles.
    Returns best gamma/beta values, best energy, and bound circuit.
    """
    J = metadata["J"]
    h = metadata["h"]
    offset = metadata["offset"]
    reps = metadata["reps"]
    n_qubits = metadata["n_qubits"]

    sim = AerSimulator()

    def expected_energy(params: np.ndarray) -> float:
        gamma_vals = params[:reps].tolist()
        beta_vals = params[reps:].tolist()
        bound = bind_qaoa_parameters(qc, gamma_vals, beta_vals)
        transpiled = transpile(bound, backend=sim, optimization_level=1)
        job = sim.run(transpiled, shots=SIM_SHOTS, seed_simulator=SIM_SEED)
        counts = job.result().get_counts()
        total = sum(counts.values())
        energy = sum(
            (count / total) * (ising_energy(J, h, bitstring) + offset)
            for bitstring, count in counts.items()
        )
        return energy

    # Random initial parameters
    rng = np.random.default_rng(SIM_SEED)
    x0 = rng.uniform(0, 2 * np.pi, 2 * reps)

    result = minimize(
        expected_energy,
        x0,
        method=QAOA_OPTIMIZER,
        options={"maxiter": QAOA_MAX_ITER, "rhobeg": 0.5},
        tol=QAOA_TOLERANCE,
    )

    best_gamma = result.x[:reps].tolist()
    best_beta = result.x[reps:].tolist()
    bound_circuit = bind_qaoa_parameters(qc, best_gamma, best_beta)

    return {
        "best_gamma": best_gamma,
        "best_beta": best_beta,
        "best_energy": float(result.fun),
        "n_iterations": result.nit,
        "converged": result.success,
        "bound_circuit": bound_circuit,
    }
