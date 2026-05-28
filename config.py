"""
IsingFlow Configuration
=======================
Central config for all magic numbers, model names, and runtime defaults.
Override any value via environment variable or by editing this file.
"""

import os

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = "claude-opus-4-6"          # Anthropic model to use
LLM_MAX_TOKENS: int = 4096                    # Max tokens per LLM response
LLM_TEMPERATURE: float = 0.2                  # Low temp for deterministic tool use

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
AGENT_MAX_ITERATIONS: int = 10               # Hard stop on ReAct iterations
AGENT_STOP_TOKEN: str = "FINAL ANSWER:"     # Token that ends the loop

# ---------------------------------------------------------------------------
# Grover's algorithm
# ---------------------------------------------------------------------------
GROVER_DEFAULT_QUBITS: int = 3               # Default search space size (2^n items)
GROVER_MAX_QUBITS: int = 10                  # Safety cap (simulator performance)

# ---------------------------------------------------------------------------
# QAOA / QUBO
# ---------------------------------------------------------------------------
QAOA_DEFAULT_REPS: int = 2                   # QAOA circuit depth (p parameter)
QAOA_MAX_REPS: int = 5                       # Cap to keep circuits shallow
QAOA_DEFAULT_QUBITS: int = 4                 # Default problem size
QAOA_MAX_QUBITS: int = 12                    # Safety cap
QAOA_OPTIMIZER: str = "COBYLA"              # Classical optimizer for VQE-style loop
QAOA_MAX_ITER: int = 500                     # Max classical optimizer iterations
QAOA_TOLERANCE: float = 1e-6                 # Convergence tolerance

# ---------------------------------------------------------------------------
# Simulation (Qiskit Aer)
# ---------------------------------------------------------------------------
SIM_SHOTS: int = 4096                        # Measurement shots for simulation
SIM_SEED: int = 42                           # RNG seed for reproducibility
SIM_NOISE_MODEL: bool = False                # Apply fake backend noise in simulation

# ---------------------------------------------------------------------------
# IBM Quantum hardware
# ---------------------------------------------------------------------------
IBM_CHANNEL: str = "ibm_quantum"             # IBM Quantum channel
IBM_INSTANCE: str = "ibm-q/open/main"       # Default open access instance
IBM_BACKEND: str = "least_busy"             # "least_busy" or specific backend name
HARDWARE_SHOTS: int = 2048                   # Measurement shots on real hardware
HARDWARE_TIMEOUT_S: int = 300               # Max seconds to wait for job result

# ---------------------------------------------------------------------------
# Circuit optimization (transpilation)
# ---------------------------------------------------------------------------
TRANSPILE_OPTIMIZATION_LEVEL: int = 3       # Qiskit transpiler level (0-3)
TRANSPILE_SEED: int = 42                     # Seed for deterministic transpilation

# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------
FIDELITY_THRESHOLD: float = 0.80            # Min fidelity to consider run successful
TOP_BITSTRINGS: int = 5                      # How many top measurement outcomes to report
ENERGY_DECIMAL_PLACES: int = 6              # Precision for Ising energy values
