"""
Tests for results analysis.
Validates fidelity calculation, energy ranking, and suggestion logic.
"""

import numpy as np
import pytest

from tools.analyze import analyze_results, _analyze_grover, _analyze_qubo
from config import FIDELITY_THRESHOLD


def _make_grover_result(probabilities: dict, target_states: list[str]) -> dict:
    top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return {
        "probabilities": probabilities,
        "top_states": top_states,
        "shots": 1024,
        "backend": "AerSimulator",
        "metadata": {"target_states": target_states, "circuit_type": "grover"},
    }


def _make_qubo_result(probabilities: dict, J, h, offset=0.0) -> dict:
    top_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return {
        "probabilities": probabilities,
        "top_states": top_states,
        "shots": 1024,
        "backend": "AerSimulator",
        "metadata": {
            "J": np.array(J),
            "h": np.array(h),
            "offset": offset,
            "circuit_type": "qubo",
        },
    }


# ---------------------------------------------------------------------------
# Grover analysis
# ---------------------------------------------------------------------------

class TestAnalyzeGrover:
    def test_perfect_result_passes(self):
        result = _make_grover_result({"101": 1.0}, ["101"])
        analysis = _analyze_grover(result)
        assert analysis["passed"] is True
        assert analysis["success_probability"] == pytest.approx(1.0)

    def test_low_fidelity_fails(self):
        # Target is "101" but only 50% probability
        result = _make_grover_result({"101": 0.5, "000": 0.5}, ["101"])
        analysis = _analyze_grover(result)
        assert analysis["passed"] is False
        assert len(analysis["suggestions"]) > 0

    def test_multi_target_success(self):
        result = _make_grover_result({"101": 0.5, "010": 0.5}, ["101", "010"])
        analysis = _analyze_grover(result)
        assert analysis["success_probability"] == pytest.approx(1.0)
        assert analysis["passed"] is True

    def test_threshold_boundary(self):
        # Exactly at threshold should pass
        result = _make_grover_result(
            {"101": FIDELITY_THRESHOLD, "000": 1 - FIDELITY_THRESHOLD},
            ["101"]
        )
        analysis = _analyze_grover(result)
        assert analysis["passed"] is True

    def test_output_keys_present(self):
        result = _make_grover_result({"101": 0.9, "000": 0.1}, ["101"])
        analysis = _analyze_grover(result)
        for key in ["circuit_type", "success_probability", "passed", "suggestions", "top_states"]:
            assert key in analysis


# ---------------------------------------------------------------------------
# QUBO/Ising analysis
# ---------------------------------------------------------------------------

class TestAnalyzeQUBO:
    def test_correct_ground_state_passes(self):
        # Simple 2-qubit antiferromagnet: ground state is "01" or "10"
        J = np.array([[0, 1.0], [0, 0]])
        h = np.zeros(2)
        # "01" should be lower energy than "00" for this J
        result = _make_qubo_result({"01": 0.9, "00": 0.1}, J, h)
        analysis = _analyze_qubo(result)
        assert analysis["top_result"] == "01"

    def test_energy_ranking_correct(self):
        J = np.zeros((2, 2))
        h = np.array([1.0, 1.0])  # prefers spin-down (s=-1, bit=1)
        result = _make_qubo_result({"11": 0.8, "00": 0.2}, J, h)
        analysis = _analyze_qubo(result)
        # "11" → s=[-1,-1], energy = -2. "00" → s=[+1,+1], energy = +2
        assert analysis["ground_state"] == "11"
        assert analysis["ground_state_energy"] < 0

    def test_suggestions_on_suboptimal_result(self):
        J = np.zeros((2, 2))
        h = np.array([1.0, 1.0])
        # Most probable is "00" but ground state is "11"
        result = _make_qubo_result({"00": 0.9, "11": 0.1}, J, h)
        analysis = _analyze_qubo(result)
        assert analysis["passed"] is False
        assert len(analysis["suggestions"]) > 0

    def test_output_keys_present(self):
        J = np.zeros((2, 2))
        h = np.zeros(2)
        result = _make_qubo_result({"00": 1.0}, J, h)
        analysis = _analyze_qubo(result)
        for key in ["circuit_type", "ground_state", "ground_state_energy",
                    "top_result", "passed", "suggestions"]:
            assert key in analysis


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestAnalyzeResultsDispatcher:
    def test_dispatches_grover(self):
        result = _make_grover_result({"101": 1.0}, ["101"])
        analysis = analyze_results(result)
        assert analysis["circuit_type"] == "grover"

    def test_dispatches_qubo(self):
        J = np.zeros((2, 2))
        h = np.zeros(2)
        result = _make_qubo_result({"00": 1.0}, J, h)
        analysis = analyze_results(result)
        assert analysis["circuit_type"] == "qubo"
