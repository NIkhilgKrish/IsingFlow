# IsingFlow

An agentic quantum computing pipeline for Grover search and QUBO/Ising optimization.

IsingFlow combines an LLM-driven ReAct agent with Qiskit-based quantum circuit design, noise-aware optimization, and execution on real IBM Quantum hardware. Problems are specified in natural language; the agent autonomously designs, simulates, evaluates, and iterates.

---

## What It Does

Given a problem in natural language (e.g. *"Find the minimum energy configuration of this 4-variable Ising Hamiltonian"*), IsingFlow:

1. **Designs** a quantum circuit — Grover's search or QAOA for QUBO/Ising
2. **Optimizes** — classical COBYLA optimizer finds the best QAOA angles; Qiskit transpiler reduces gate count for the target backend
3. **Simulates** on Qiskit Aer (fast, local, no credits consumed)
4. **Evaluates** — fidelity for Grover, Ising ground state energy for QUBO; flags quality issues and suggests improvements
5. **Executes** on real IBM Quantum hardware if simulation passes quality threshold
6. **Reports** — final answer with energy landscape, approximation ratio, and backend metadata

---

## Motivation

QUBO (Quadratic Unconstrained Binary Optimization) and the Ising model are mathematically equivalent. NVIDIA's quantum platform — including the NVIDIA Ising solver and cuQuantum — targets exactly this class of problems. IsingFlow is designed to sit upstream of hardware execution, automating the circuit design and optimization workflow that researchers currently do by hand.

---

## Architecture

```
IsingFlow/
├── agent/
│   ├── loop.py          # ReAct agent core (~80 lines, no framework)
│   └── prompts.py       # System prompt and tool descriptions
├── tools/
│   ├── design.py        # Circuit design tool
│   ├── optimize.py      # Transpilation + QAOA parameter optimization
│   ├── simulate.py      # Qiskit Aer simulation
│   ├── hardware.py      # IBM Quantum execution
│   └── analyze.py       # Fidelity / energy analysis + suggestions
├── circuits/
│   ├── grover.py        # Grover's search algorithm
│   └── qaoa_qubo.py     # QAOA for QUBO/Ising problems
├── tests/
│   ├── test_circuits.py
│   ├── test_tools.py
│   └── test_analyze.py
├── notebooks/
│   └── walkthrough.ipynb
├── config.py            # All magic numbers and configuration
└── requirements.txt
```

The agent uses a hand-rolled ReAct loop (Reason → Act → Observe) with no external framework. Each tool is a plain Python function — the agent calls them by name, threads context automatically, and iterates until the quality threshold is met or a final answer is produced.

---

## Supported Problems

**Grover Search**
Unstructured search over `2^n` items. Specify target bitstrings; the agent builds an optimal oracle and determines the number of Grover iterations automatically.

**QUBO / Ising Optimization**
Provide a QUBO matrix `Q`; IsingFlow converts it to Ising form (`J`, `h`) and solves via QAOA. The QUBO ↔ Ising mapping is:

```
x_i = (1 - s_i) / 2
```

This is the same formulation used by NVIDIA's Ising solver.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY="your_key_here"
export IBM_QUANTUM_TOKEN="your_token_here"   # optional, for real hardware
```

### 3. Run the agent

```python
from agent.loop import run

# Grover search
run("Find the state '101' in a 3-qubit search space.")

# QUBO / Ising optimization
run("""
Solve the QUBO problem with matrix:
Q = [[1, -2], [-2, 1]]
Find the binary assignment x ∈ {0,1}^2 that minimizes x^T Q x.
""")
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Explore the notebook

```bash
jupyter notebook notebooks/walkthrough.ipynb
```

---

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `QAOA_DEFAULT_REPS` | 2 | QAOA circuit depth (higher = better approximation) |
| `SIM_SHOTS` | 4096 | Measurement shots for simulation |
| `HARDWARE_SHOTS` | 2048 | Measurement shots on real QPU |
| `FIDELITY_THRESHOLD` | 0.80 | Min success probability before accepting Grover results |
| `TRANSPILE_OPTIMIZATION_LEVEL` | 3 | Qiskit transpiler aggressiveness (0-3) |
| `IBM_BACKEND` | `"least_busy"` | Target backend; set to a specific name to fix it |

---

## Results

Example output for a 4-variable Max-Cut QUBO on `ibm_brisbane`:

```
Ground state:  1010  (E = -2.000000)
Top result:    1010  (E = -2.000000)
Approx ratio:  1.000
Backend:       ibm_brisbane
Shots:         2048
```

---

## Dependencies

- [Qiskit](https://qiskit.org/) — circuit construction, transpilation, Aer simulation
- [Qiskit IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime) — IBM Quantum access
- [Anthropic Python SDK](https://github.com/anthropic/anthropic-sdk-python) — LLM backbone
- [SciPy](https://scipy.org/) — COBYLA optimizer for QAOA parameters
