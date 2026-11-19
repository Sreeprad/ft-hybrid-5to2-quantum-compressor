# Fault-Tolerant Quantum Circuit Design of Hybrid Approximate-Exact 5:2 Compressors

Official simulation, verification, and reproducibility repository for the manuscript:
> **Fault-Tolerant Quantum Circuit Design of Hybrid Approximate-Exact 5:2 Compressors for Arithmetic Applications**
> Manda V S A L Sreeprad, Aravindhan Alagarsamy, R. S. Ernest Ravindran, Yu-Chen Hu, Gian Carlo Cardarilli, and Seok-Bum Ko.

## Overview

This repository provides a hardware-locked Clifford+T synthesis and quantum statevector validation suite for 5:2 quantum compressors. To eliminate heuristic transpiler variance, all Toffoli (`CCX`) gates are manually decomposed into mathematically verified, ancilla-free 7-T Clifford+T sequences (`optimization_level=0`).

The benchmark evaluates **9 architectures** across the T-count–fidelity Pareto frontier:
1. **Exact-Parallel (3-FA Baseline)** — 42 T-gates, 6 ancillas
2. **Exact-InPlace (Serial Low-Ancilla)** — 42 T-gates, 4 ancillas
3. **Single-AND Hybrid** — 35 T-gates, 6 ancillas
4. **Proposed Hybrid-1** — 28 T-gates (**33.3% T-count reduction**), 6 ancillas
5. **Proposed Hybrid-1-LA (Wire-Aliased)** — 28 T-gates (**33.3% T-count**, **16.7% ancilla**, **27.5% CNOT reduction**), 5 ancillas
6. **Published Pairwise-AND AFA-5:2** — 21 T-gates, 4 ancillas
7. **Hybrid-2 (2-Stage Zero-Toffoli)** — 14 T-gates, 6 ancillas
8. **Extreme Zero-T Clifford Routing** — 0 T-gates, 4 ancillas
9. **NISQ AQANC No-Carry Truncation** — 0 T-gates, 4 ancillas

## Verified Headline Results (Locked Standard 7-T Model)

| Architecture | T-Count | T-Depth (Std / Amy) | CNOT (Std / Amy) | Ancillas | Total Qubits | $F_3$ (3-Out) | $F_4$ (4-Out) | Trace Dist. ($D_3$) | ER (%) | Bias | NMED |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Exact-Parallel (3-FA) | 42 | 8 / 6 | 51 / 57 | 6 | 13 | 1.000000 | 1.000000 | 0.000000 | 0.00 | +0.00 | 0.0000 |
| Exact-InPlace (Serial) | 42 | 16 / 12 | 51 / 57 | 4 | 11 | 1.000000 | 1.000000 | 0.000000 | 0.00 | +0.00 | 0.0000 |
| Single-AND Hybrid | 35 | 8 / 6 | 43 / 48 | 6 | 13 | 0.918199 | 0.916379 | 0.250000 | 25.00 | -0.50 | 0.0714 |
| **Proposed Hybrid-1** | **28** | **8 / 6** | **38 / 42** | **6** | **13** | **0.984123** | **0.982172** | **0.125000** | **25.00** | **+0.00** | **0.0714** |
| **Proposed Hybrid-1-LA** | **28** | **8 / 6** | **37 / 41** | **5** | **12** | **0.984123** | **0.982172** | **0.125000** | **25.00** | **+0.00** | **0.0714** |
| Published AFA-5:2 | 21 | 4 / 3 | 25 / 28 | 4 | 11 | 0.841138 | 0.687780 | 0.375000 | 68.75 | -1.50 | 0.2143 |
| Hybrid-2 (14T) | 14 | 8 / 6 | 25 / 27 | 6 | 13 | 0.984123 | 0.958406 | 0.125000 | 37.50 | +0.00 | 0.1071 |
| Extreme Zero-T Route | 0 | 0 / 0 | 10 / 10 | 4 | 11 | 0.984123 | 0.886521 | 0.125000 | 45.31 | +0.00 | 0.1339 |
| NISQ AQANC No-Carry | 0 | 0 / 0 | 7 / 7 | 4 | 11 | 0.187500 | 0.051920 | 0.812500 | 93.75 | -3.00 | 0.4286 |

## Verified Simulation Environment

All published numbers were generated under an ideal, deterministic, noiseless statevector configuration (`AerSimulator(method='statevector')`, `noise_model=None`, `optimization_level=0`, peak memory < 10 MB):

- **Python:** `3.13.18` (Linux x86_64)
- **Qiskit:** `2.6.2`
- **Qiskit-Aer:** `0.18.4`
- **NumPy:** `2.4.6`
- **SciPy:** `1.16.3`

## Quick Start (Reproduce All Tables & Figures)

```bash
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/ft-hybrid-5to2-quantum-compressor.git
cd ft-hybrid-5to2-quantum-compressor
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Verify 7-T unitaries and reproduce Table 1, Amy sensitivity check, and Table 2
python benchmark.py

# 2. Generate all 300-DPI PNG and vector PDF figures
python generate_figures.py
