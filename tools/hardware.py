"""
Tool: execute_on_hardware
==========================
Submits a circuit to a real IBM Quantum backend via Qiskit IBM Runtime.
Falls back gracefully with a clear error if credentials are missing.
"""

from __future__ import annotations

import os
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from config import (
    IBM_CHANNEL,
    IBM_INSTANCE,
    IBM_BACKEND,
    HARDWARE_SHOTS,
    HARDWARE_TIMEOUT_S,
)


def execute_on_hardware(optimize_result: dict, use_fake: bool = False) -> dict:
    """
    Execute a circuit on IBM Quantum hardware (or a fake backend for testing).

    Args:
        optimize_result: Output dict from optimize_circuit tool.
        use_fake:        If True, uses FakeSherbrooke instead of real hardware.
                         Useful for development without consuming QPU credits.

    Returns:
        {
            "counts":        dict[str, int],
            "probabilities": dict[str, float],
            "top_states":    list[tuple],
            "shots":         int,
            "backend":       str,
            "job_id":        str | None,
            "metadata":      dict,
        }

    Raises:
        RuntimeError: If IBM credentials are missing and use_fake is False.
    """
    qc: QuantumCircuit = optimize_result["circuit"]
    metadata = optimize_result["metadata"]

    if use_fake:
        return _run_fake_backend(qc, metadata)

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN environment variable not set. "
            "Set it or use use_fake=True for testing."
        )

    service = QiskitRuntimeService(
        channel=IBM_CHANNEL,
        token=token,
        instance=IBM_INSTANCE,
    )

    if IBM_BACKEND == "least_busy":
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=qc.num_qubits,
        )
    else:
        backend = service.backend(IBM_BACKEND)

    sampler = Sampler(backend)
    job = sampler.run([qc], shots=HARDWARE_SHOTS)
    result = job.result()[0]

    counts = result.data.meas.get_counts()
    total = sum(counts.values())
    probabilities = {state: count / total for state, count in counts.items()}
    top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "counts": counts,
        "probabilities": probabilities,
        "top_states": top_states,
        "shots": HARDWARE_SHOTS,
        "backend": backend.name,
        "job_id": job.job_id(),
        "metadata": metadata,
    }


def _run_fake_backend(qc: QuantumCircuit, metadata: dict) -> dict:
    """Run on FakeSherbrooke — realistic noise model, no QPU credits needed."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    fake_backend = FakeSherbrooke()
    noise_model = NoiseModel.from_backend(fake_backend)
    sim = AerSimulator(noise_model=noise_model)

    transpiled = transpile(qc, backend=fake_backend, optimization_level=3)
    job = sim.run(transpiled, shots=HARDWARE_SHOTS)
    counts = job.result().get_counts()
    total = sum(counts.values())
    probabilities = {state: count / total for state, count in counts.items()}
    top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "counts": counts,
        "probabilities": probabilities,
        "top_states": top_states,
        "shots": HARDWARE_SHOTS,
        "backend": "FakeSherbrooke (noise model)",
        "job_id": None,
        "metadata": metadata,
    }
