"""
Tool: analyze_results
======================
Interprets raw measurement counts from simulation or hardware.
Computes fidelity (for Grover), energy landscape (for QUBO), and
flags whether results pass quality thresholds.
"""

from __future__ import annotations

import numpy as np
from circuits.qaoa_qubo import ising_energy
from config import FIDELITY_THRESHOLD, TOP_BITSTRINGS, ENERGY_DECIMAL_PLACES


def analyze_results(execution_result: dict) -> dict:
    """
    Analyze measurement results from simulate_circuit or execute_on_hardware.

    Dispatches to the appropriate analysis function based on circuit type
    stored in metadata.

    Args:
        execution_result: Output dict from simulate_circuit or execute_on_hardware.

    Returns:
        Analysis dict — see _analyze_grover / _analyze_qubo for shape.
    """
    circuit_type = execution_result["metadata"].get("circuit_type", "")

    # circuit_type is propagated differently depending on path taken
    # fall back to checking metadata keys
    if circuit_type == "grover" or "target_states" in execution_result["metadata"]:
        return _analyze_grover(execution_result)
    elif circuit_type == "qubo" or "Q" in execution_result["metadata"]:
        return _analyze_qubo(execution_result)
    else:
        return _analyze_generic(execution_result)


def _analyze_grover(execution_result: dict) -> dict:
    """
    Grover analysis: compute success probability and fidelity.

    Fidelity = probability mass on the target states.
    """
    probabilities = execution_result["probabilities"]
    target_states = execution_result["metadata"].get("target_states", [])
    top_states = execution_result["top_states"][:TOP_BITSTRINGS]

    success_prob = sum(probabilities.get(t, 0.0) for t in target_states)
    passed = success_prob >= FIDELITY_THRESHOLD

    suggestions = []
    if not passed:
        suggestions.append(
            f"Success probability {success_prob:.3f} is below threshold "
            f"{FIDELITY_THRESHOLD}. Consider increasing Grover iterations or "
            f"verifying the oracle construction."
        )

    return {
        "circuit_type": "grover",
        "success_probability": round(success_prob, 4),
        "fidelity_threshold": FIDELITY_THRESHOLD,
        "passed": passed,
        "target_states": target_states,
        "top_states": top_states,
        "suggestions": suggestions,
        "backend": execution_result["backend"],
        "shots": execution_result["shots"],
    }


def _analyze_qubo(execution_result: dict) -> dict:
    """
    QUBO/Ising analysis: compute energy for each observed bitstring,
    identify the approximate ground state, and report the energy gap.
    """
    probabilities = execution_result["probabilities"]
    metadata = execution_result["metadata"]
    J = np.array(metadata["J"])
    h = np.array(metadata["h"])
    offset = metadata["offset"]

    energies = {
        bitstring: round(ising_energy(J, h, bitstring) + offset, ENERGY_DECIMAL_PLACES)
        for bitstring in probabilities
    }

    sorted_by_energy = sorted(energies.items(), key=lambda x: x[1])
    ground_state_bitstring, ground_state_energy = sorted_by_energy[0]

    # Most probable state (what the circuit actually returned)
    top_bitstring = execution_result["top_states"][0][0]
    top_energy = energies[top_bitstring]

    approx_ratio = None
    if abs(ground_state_energy) > 1e-10:
        approx_ratio = round(top_energy / ground_state_energy, 4)

    passed = top_bitstring == ground_state_bitstring

    suggestions = []
    if not passed:
        gap = round(top_energy - ground_state_energy, ENERGY_DECIMAL_PLACES)
        suggestions.append(
            f"Most probable state ({top_bitstring}, E={top_energy}) is not the "
            f"ground state ({ground_state_bitstring}, E={ground_state_energy}). "
            f"Energy gap: {gap}. Try increasing QAOA reps or running more shots."
        )
    if approx_ratio is not None and approx_ratio > 1.1:
        suggestions.append(
            f"Approximation ratio {approx_ratio:.3f} > 1.1. "
            f"Consider deeper QAOA circuit (increase reps)."
        )

    return {
        "circuit_type": "qubo",
        "ground_state": ground_state_bitstring,
        "ground_state_energy": ground_state_energy,
        "top_result": top_bitstring,
        "top_result_energy": top_energy,
        "approximation_ratio": approx_ratio,
        "passed": passed,
        "top_energies": sorted_by_energy[:TOP_BITSTRINGS],
        "suggestions": suggestions,
        "backend": execution_result["backend"],
        "shots": execution_result["shots"],
    }


def _analyze_generic(execution_result: dict) -> dict:
    """Fallback analysis: just report top states and probabilities."""
    return {
        "circuit_type": "unknown",
        "top_states": execution_result["top_states"][:TOP_BITSTRINGS],
        "shots": execution_result["shots"],
        "backend": execution_result["backend"],
        "passed": True,
        "suggestions": [],
    }
