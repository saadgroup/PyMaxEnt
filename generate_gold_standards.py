#!/usr/bin/env python
"""
generate_gold_standards.py
--------------------------
Run this script to generate (or rebless) the gold standard lambda values
used by the regression suite.  Results are saved to gold_standards.npz.

Usage:
    python generate_gold_standards.py

When to re-run:
    - First time setup
    - After intentionally changing the algorithm and verifying the new
      output is correct (rebless)
"""

import warnings
import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm

try:
    from pymaxent import reconstruct, moments_c
except ImportError:
    from src.pymaxent import reconstruct, moments_c

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Distribution definitions (mirror of examples.ipynb)
# ---------------------------------------------------------------------------

def gauss(x):
    sigma, mu = 0.2, 1.0
    A = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def gauss2(x):
    s0, m0 = 1.0/14.0, 1.0/4.0;  A0 = 1.0/(2.0*s0*np.sqrt(2*np.pi))
    s1, m1 = 1.0/20.0, 2.0/4.0;  A1 = 1.0/(2.0*s1*np.sqrt(2*np.pi))
    return (A0*np.exp(-(x-m0)**2/(2*s0**2))
          + A1*np.exp(-(x-m1)**2/(2*s1**2)))

def gauss3(x):
    s0, m0 = 1.0/14.0, 1.0/4.0;  A0 = 1.0/(2.0*s0*np.sqrt(2*np.pi))
    s1, m1 = 1.0/20.0, 2.0/4.0;  A1 = 1.0/(2.0*s1*np.sqrt(2*np.pi))
    s2, m2 = 1.0/20.0, 3.0/4.0;  A2 = 1.0/(2.0*s1*np.sqrt(2*np.pi))
    return (A0*np.exp(-(x-m0)**2/(2*s0**2))
          + A1*np.exp(-(x-m1)**2/(2*s1**2))
          + A2*np.exp(-(x-m2)**2/(2*s2**2)))

def beta_dist(x):
    a, b = 3, 9
    btm = quad(lambda X: X**(a-1)*(1-X)**(b-1), 0, 1)[0]
    return x**(a-1) * (1-x)**(b-1) / btm

def beta2_dist(x):
    a = b = 0.5
    btm = quad(lambda X: X**(a-1)*(1-X)**(b-1), 0, 1)[0]
    return x**(a-1) * (1-x)**(b-1) / btm

_ld = lognorm([0.25], loc=0.2)
def lognorm_pdf(x):
    return np.squeeze(_ld.pdf(x))

# ---------------------------------------------------------------------------
# Run all 13 cases and collect lambdas
# ---------------------------------------------------------------------------

gs = {}

print("Generating gold standards...")

# --- Discrete ---
mu = np.array([1]);  x = np.array([1,2,3,4,5,6])
sol, lam = reconstruct(mu=mu, rndvar=x)
gs["D1_lambdas"] = lam;  gs["D1_sol"] = sol
print(f"  D1  lambdas={lam}")

mu = [1, 3.5];  x = [1,2,3,4,5,6]
sol, lam = reconstruct(mu=mu, rndvar=x)
gs["D2_lambdas"] = lam;  gs["D2_sol"] = sol
print(f"  D2  lambdas={lam}")

pi = np.array([1,1,1,2,3,4]) / 12.0;  x = [1,2,3,4,5,6]
mu = [1, np.sum(pi * np.array(x))]
sol, lam = reconstruct(mu=mu, rndvar=x)
gs["D3_lambdas"] = lam;  gs["D3_sol"] = sol
print(f"  D3  lambdas={lam}")

# --- Continuous ---
mu = moments_c(gauss, 3, bnds=[0,2])
_, lam = reconstruct(mu=mu, bnds=[0,2])
gs["C1_lambdas"] = lam
print(f"  C1  lambdas={lam}")

mu = moments_c(gauss2, 5, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C2_lambdas"] = lam
print(f"  C2  lambdas={lam}")

mu = moments_c(gauss2, 10, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C3_lambdas"] = lam
print(f"  C3  lambdas={lam}")

mu = moments_c(gauss3, 5, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C4_lambdas"] = lam
print(f"  C4  lambdas={lam}")

mu = moments_c(gauss3, 13, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C5_lambdas"] = lam
print(f"  C5  lambdas={lam}")

mu = moments_c(beta_dist, 3, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C6a_lambdas"] = lam
print(f"  C6a lambdas={lam}")

mu = moments_c(beta_dist, 5, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C6b_lambdas"] = lam
print(f"  C6b lambdas={lam}")

mu = moments_c(beta2_dist, 3, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C7a_lambdas"] = lam
print(f"  C7a lambdas={lam}")

mu = moments_c(beta2_dist, 5, bnds=[0,1])
_, lam = reconstruct(mu=mu, bnds=[0,1])
gs["C7b_lambdas"] = lam
print(f"  C7b lambdas={lam}")

mu = moments_c(lognorm_pdf, 3, bnds=[0,5])
_, lam = reconstruct(mu=mu, bnds=[0,5])
gs["C8a_lambdas"] = lam
print(f"  C8a lambdas={lam}")

mu = moments_c(lognorm_pdf, 5, bnds=[0,5])
_, lam = reconstruct(mu=mu, bnds=[0,5])
gs["C8b_lambdas"] = lam
print(f"  C8b lambdas={lam}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

np.savez("gold_standards.npz", **gs)
print("\nSaved to gold_standards.npz")
