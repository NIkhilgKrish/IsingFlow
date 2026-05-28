"""
Tests for agent tools.
Uses mocks for IBM Quantum to avoid live API calls.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit

from tools.design import design_circuit
from tools.optimize import optimize_circuit
from tools.simulate import simulate_circuit


# ---------------------------------------------------------------------------
# design_circuit
# ---------------------------------------------------------------------------

class TestDesignCircuit:
    def test_grover_returns_circuit(self):
        result = design_circuit({"type": "grover", "target_states": ["101"], "n_qubits": 3})
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit_type"] == "grover"
        assert "info" in result
        assert "metadata" in result

    def test_qubo_returns_circuit(self):
        Q = [[1, -1], [-1, 1]]
        result = design_circuit({"type": "qubo", "Q": Q, "reps": 1})
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit_type"] == "qubo"
        assert "J" in result["metadata"]
        assert "h" in result["metadata"]

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown problem type"):
            design_circuit({"type": "vqe"})

    def test_grover_missing_targets_raises(self):
        with pytest.raises(ValueError):
            design_circuit({"type": "grover"})

    def test_qubo_missing_Q_raises(self):
        with pytest.raises(ValueError):
            design_circuit({"type": "qubo"})

    def test_grover_metadata_populated(self):
        result = design_circuit({"type": "grover", "target_states": ["11"], "n_qubits": 2})
        assert result["metadata"]["target_states"] == ["11"]
        assert result["metadata"]["n_qubits"] == 2

    def test_qubo_metadata_has_offset(self):
        Q = [[1, 0], [0, 1]]
        result = design_circuit({"type": "qubo", "Q": Q, "reps": 1})
        assert "offset" in result["metadata"]


# ---------------------------------------------------------------------------
# simulate_circuit (uses real Aer — fast, no API needed)
# ---------------------------------------------------------------------------

class TestSimulateCircuit:
    def _make_optimize_result(self, circuit_type="grover"):
        """Build a minimal optimize_result dict with a real transpiled circuit."""
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        if circuit_type == "grover":
            design = design_circuit({"type": "grover", "target_states": ["11"], "n_qubits": 2})
            metadata = design["metadata"]
            metadata["circuit_type"] = "grover"
        else:
            Q = np.array([[1, -1], [-1, 1]])
            design = design_circuit({"type": "qubo", "Q": Q.tolist(), "reps": 1})
            from circuits.qaoa_qubo import bind_qaoa_parameters
            bound = bind_qaoa_parameters(design["circuit"], [0.5], [0.3])
            design["circuit"] = bound
            metadata = design["metadata"]
            metadata["circuit_type"] = "qubo"

        backend = AerSimulator()
        transpiled = transpile(design["circuit"], backend=backend)
        return {"circuit": transpiled, "metadata": metadata}

    def test_grover_returns_counts(self):
        opt_result = self._make_optimize_result("grover")
        result = simulate_circuit(opt_result)
        assert "counts" in result
        assert "probabilities" in result
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_top_states_sorted(self):
        opt_result = self._make_optimize_result("grover")
        result = simulate_circuit(opt_result)
        probs = [p for _, p in result["top_states"]]
        assert probs == sorted(probs, reverse=True)

    def test_shots_correct(self):
        from config import SIM_SHOTS
        opt_result = self._make_optimize_result("grover")
        result = simulate_circuit(opt_result)
        assert result["shots"] == SIM_SHOTS
