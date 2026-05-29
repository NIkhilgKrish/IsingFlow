"""
Tests for circuit generators.
Validates structure, not numerical results — Qiskit's correctness is not our job.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from circuits.grover import build_grover_circuit, grover_circuit_info
from circuits.qaoa_qubo import (
    build_qaoa_circuit,
    qubo_to_ising,
    bind_qaoa_parameters,
    ising_energy,
)
from config import GROVER_MAX_QUBITS, QAOA_MAX_QUBITS, QAOA_MAX_REPS


# ---------------------------------------------------------------------------
# Grover
# ---------------------------------------------------------------------------

class TestGroverCircuit:
    def test_returns_quantum_circuit(self):
        qc = build_grover_circuit(["101"], n_qubits=3)
        assert isinstance(qc, QuantumCircuit)

    def test_correct_qubit_count(self):
        for n in [2, 3, 4]:
            qc = build_grover_circuit(["1" * n], n_qubits=n)
            assert qc.num_qubits == n

    def test_has_measurements(self):
        qc = build_grover_circuit(["101"], n_qubits=3)
        assert qc.num_clbits == 3

    def test_multiple_targets(self):
        qc = build_grover_circuit(["101", "010"], n_qubits=3)
        assert isinstance(qc, QuantumCircuit)

    def test_invalid_qubit_count_raises(self):
        with pytest.raises(ValueError):
            build_grover_circuit(["1"], n_qubits=GROVER_MAX_QUBITS + 1)

    def test_empty_targets_raises(self):
        with pytest.raises(ValueError):
            build_grover_circuit([], n_qubits=3)

    def test_mismatched_target_length_raises(self):
        with pytest.raises(ValueError):
            build_grover_circuit(["10"], n_qubits=3)  # length 2 != 3

    def test_circuit_info_keys(self):
        qc = build_grover_circuit(["101"], n_qubits=3)
        info = grover_circuit_info(qc)
        assert "depth" in info
        assert "gate_count" in info
        assert info["n_qubits"] == 3


# ---------------------------------------------------------------------------
# QUBO / QAOA
# ---------------------------------------------------------------------------

class TestQUBOToIsing:
    def test_identity_qubo(self):
        """Diagonal QUBO (no coupling) → zero J, h = -0.5 per qubit."""
        Q = np.eye(3)
        J, h, offset = qubo_to_ising(Q)
        assert np.allclose(J, 0), "No off-diagonal coupling expected"
        assert np.allclose(h, -0.5 * np.ones(3)), "h should be -Q_ii/2"
        assert offset == pytest.approx(1.5)

    def test_symmetry(self):
        """Non-symmetric Q should be symmetrized."""
        Q = np.array([[1, 2], [0, 1]], dtype=float)
        J1, h1, o1 = qubo_to_ising(Q)
        Q_sym = (Q + Q.T) / 2
        J2, h2, o2 = qubo_to_ising(Q_sym)
        assert np.allclose(J1, J2)
        assert np.allclose(h1, h2)

    def test_known_2qubit(self):
        """Q = [[0,-1],[-1,0]]: ground state x=[1,1], energy=-1 (upper triangular convention)."""
        Q = np.array([[0, -1], [-1, 0]], dtype=float)
        J, h, offset = qubo_to_ising(Q)
        # Verify energy at each state matches QUBO (upper triangular: f = -x0*x1)
        assert ising_energy(J, h, "11") + offset == pytest.approx(-1.0)  # x=[1,1], minimum
        assert ising_energy(J, h, "00") + offset == pytest.approx(0.0)   # x=[0,0]


class TestQAOACircuit:
    def test_returns_quantum_circuit(self):
        Q = np.array([[1, -1], [-1, 1]], dtype=float)
        qc, J, h, offset = build_qaoa_circuit(Q, reps=1)
        assert isinstance(qc, QuantumCircuit)

    def test_qubit_count_matches_problem_size(self):
        for n in [2, 3, 4]:
            Q = np.eye(n)
            qc, _, _, _ = build_qaoa_circuit(Q, reps=1)
            assert qc.num_qubits == n

    def test_parameter_count(self):
        """Should have 2*reps free parameters (gamma + beta)."""
        Q = np.eye(3)
        for reps in [1, 2, 3]:
            qc, _, _, _ = build_qaoa_circuit(Q, reps=reps)
            assert qc.num_parameters == 2 * reps

    def test_reps_out_of_range_raises(self):
        Q = np.eye(2)
        with pytest.raises(ValueError):
            build_qaoa_circuit(Q, reps=QAOA_MAX_REPS + 1)

    def test_too_large_raises(self):
        Q = np.eye(QAOA_MAX_QUBITS + 1)
        with pytest.raises(ValueError):
            build_qaoa_circuit(Q, reps=1)

    def test_bind_parameters(self):
        Q = np.eye(3)
        qc, _, _, _ = build_qaoa_circuit(Q, reps=2)
        bound = bind_qaoa_parameters(qc, [0.5, 0.5], [0.3, 0.3])
        assert bound.num_parameters == 0

    def test_bind_wrong_length_raises(self):
        Q = np.eye(3)
        qc, _, _, _ = build_qaoa_circuit(Q, reps=2)
        with pytest.raises(ValueError):
            bind_qaoa_parameters(qc, [0.5], [0.3, 0.3])


class TestIsingEnergy:
    def test_all_spin_up(self):
        """All zeros bitstring → all spins +1."""
        J = np.zeros((2, 2))
        h = np.array([1.0, 1.0])
        energy = ising_energy(J, h, "00")
        assert energy == pytest.approx(2.0)

    def test_all_spin_down(self):
        J = np.zeros((2, 2))
        h = np.array([1.0, 1.0])
        energy = ising_energy(J, h, "11")
        assert energy == pytest.approx(-2.0)

    def test_coupled_system(self):
        """Two antiferromagnetically coupled spins: ground state is "01" or "10"."""
        J = np.array([[0, 1.0], [0, 0]])
        h = np.zeros(2)
        e_aligned = ising_energy(J, h, "00")      # s = [+1, +1]
        e_antialigned = ising_energy(J, h, "01")  # s = [+1, -1]
        assert e_antialigned < e_aligned
