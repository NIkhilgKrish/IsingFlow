import sys
sys.path.insert(0, ".")
from agent.loop import run

# Start simple: 2-variable Max-Cut QUBO, simulation only
run("""
Solve this QUBO problem using simulation only (do not run on hardware):
Q = [[0, -1], [-1, 0]]
Find the binary assignment x in {0,1}^2 that minimizes x^T Q x.
Report the ground state, its energy, and the approximation ratio.
""")