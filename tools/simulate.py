"""
Tool: simulate_circuit
=======================
Runs a circuit on Qiskit Aer (local simulator).
Optionally applies a noise model from a fake IBM backend for realistic results.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from config import SIM_SHOTS, SIM_SEED, SIM_NOISE_MODEL, TRANSPILE_SEED


def simulate_circuit(
    optimize_result: dict,
    noise_model: NoiseModel | None = None,
) -> dict:
    """
    Simulate a circuit using Qiskit Aer.

    Args:
        optimize_result: Output dict from optimize_circuit tool.
        noise_model:     Optional Aer NoiseModel. If None and SIM_NOISE_MODEL
                         is False, uses ideal statevector simulation.

    Returns:
        {
            "counts":        dict[str, int],   # raw measurement counts
            "probabilities": dict[str, float], # normalized probabilities
            "top_states":    list[tuple],      # (bitstring, probability) sorted desc
            "shots":         int,
            "backend":       str,
            "metadata":      dict,             # pass-through
        }
    """
    qc: QuantumCircuit = optimize_result["circuit"]
    metadata = optimize_result["metadata"]

    if noise_model is not None or SIM_NOISE_MODEL:
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = AerSimulator(method="statevector")

    transpiled = transpile(qc, backend=backend, seed_transpiler=TRANSPILE_SEED)
    job = backend.run(transpiled, shots=SIM_SHOTS, seed_simulator=SIM_SEED)
    result = job.result()
    counts = result.get_counts()

    total = sum(counts.values())
    probabilities = {state: count / total for state, count in counts.items()}
    top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "counts": counts,
        "probabilities": probabilities,
        "top_states": top_states,
        "shots": SIM_SHOTS,
        "backend": "AerSimulator",
        "metadata": metadata,
    }
