#!/usr/bin/env python3
"""
Master Reproducibility & Benchmark Runner.

Executes unitary verification, full superposition (H^7|0>^7) statevector
simulation across all 9 5:2 compressor architectures under both locked 7-T
Toffoli decompositions, and exports results to results/benchmark_results.json
and results/benchmark_table1.csv.
"""

import csv
import gc
import json
import os
import platform
import sys
import tracemalloc

import numpy as np
import qiskit
import scipy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from circuits import (
    BENCHMARK_BUILDERS,
    apply_toffoli_amy_7t,
    apply_toffoli_standard_7t,
    compute_t_depth,
    verify_toffoli_decompositions,
)

try:
    import qiskit_aer
    from qiskit_aer import AerSimulator

    AER_AVAILABLE = True
    AER_VERSION = qiskit_aer.__version__
    SIMULATOR = AerSimulator(method="statevector")
except ImportError:
    AER_AVAILABLE = False
    AER_VERSION = "Not installed (fallback: qiskit.quantum_info.Statevector)"
    SIMULATOR = None


def exact_trace_distance(rho, sigma) -> float:
    """Compute exact trace distance 0.5 * Tr|rho - sigma| via Hermitian eigvalsh."""
    diff = np.asarray(rho.data, dtype=np.complex128) - np.asarray(sigma.data, dtype=np.complex128)
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def run_statevector(qc: QuantumCircuit) -> Statevector:
    """Deterministic, noiseless statevector simulation (optimization_level=0)."""
    if AER_AVAILABLE:
        qc_sim = qc.copy()
        qc_sim.save_statevector()
        res = SIMULATOR.run(qc_sim, shots=1).result()
        return Statevector(res.get_statevector(qc_sim))
    return Statevector.from_instruction(qc)


def extract_classical_accuracy(sv: Statevector, out4_indices: list) -> dict:
    """Extract exhaustive 128-state arithmetic error metrics from the superposition statevector."""
    probs = np.abs(sv.data) ** 2
    nz = np.where(probs > 1e-12)[0]
    assert len(nz) == 128, f"Expected 128 basis states, got {len(nz)}"
    assert np.allclose(probs[nz], 1.0 / 128.0, atol=1e-9), "Non-uniform input probability detected!"

    i_c1, i_c2, i_sum, i_car = out4_indices
    errors = []
    seen_inputs = set()
    for idx in nz:
        inp_bits = [(idx >> k) & 1 for k in range(7)]
        seen_inputs.add(tuple(inp_bits))
        exact_val = sum(inp_bits)
        c1 = (idx >> i_c1) & 1
        c2 = (idx >> i_c2) & 1
        sm = (idx >> i_sum) & 1
        cr = (idx >> i_car) & 1
        approx_val = sm + 2 * (c1 + c2 + cr)
        errors.append(approx_val - exact_val)

    assert len(seen_inputs) == 128, "Input register was not preserved cleanly!"
    errors = np.array(errors, dtype=np.float64)
    er = float(np.mean(errors != 0) * 100.0)
    me = float(np.mean(errors))
    med = float(np.mean(np.abs(errors)))
    nmed = float(med / 7.0)
    max_ed = int(np.max(np.abs(errors)))
    return {"ER_pct": er, "Mean_Err": me, "MED": med, "NMED": nmed, "Max_ED": max_ed}


def evaluate_suite(toffoli_fn):
    results = []
    exact_dm3, exact_dm4 = None, None
    prob_tables = {}

    for label, builder in BENCHMARK_BUILDERS:
        qc, out3, out4, meta = builder(toffoli_fn)
        assert out3 == sorted(out3) and out4 == sorted(out4), f"Unsorted outputs in {label}"

        ops = qc.count_ops()
        t_cnt = int(ops.get("t", 0) + ops.get("tdg", 0))
        t_dep = int(compute_t_depth(qc))
        cx_cnt = int(ops.get("cx", 0))
        h_cnt = int(ops.get("h", 0)) - 7
        tot_depth = int(qc.depth())
        n_total = qc.num_qubits
        n_anc = n_total - 7
        st_vol = n_total * t_dep

        sv = run_statevector(qc)
        trace3 = [i for i in range(n_total) if i not in out3]
        trace4 = [i for i in range(n_total) if i not in out4]
        dm3 = partial_trace(sv, trace3)
        dm4 = partial_trace(sv, trace4)

        if label == "Exact-Parallel (3-FA)":
            exact_dm3, exact_dm4 = dm3, dm4

        fid3 = float(state_fidelity(exact_dm3, dm3))
        td3 = exact_trace_distance(exact_dm3, dm3)
        fid4 = float(state_fidelity(exact_dm4, dm4))
        td4 = exact_trace_distance(exact_dm4, dm4)
        cls = extract_classical_accuracy(sv, out4)

        if label in ("Exact-Parallel (3-FA)", "Proposed Hybrid-1", "Proposed Hybrid-1-LA"):
            prob_tables[label] = {
                "p3": np.diag(dm3.data).real.tolist(),
                "p4": np.diag(dm4.data).real.tolist(),
            }

        results.append({
            "Model": label,
            "T_Count": t_cnt,
            "T_Depth": t_dep,
            "CNOT": cx_cnt,
            "H_Logic": h_cnt,
            "Total_Depth": tot_depth,
            "Ancillas": n_anc,
            "Workspace_Anc": meta["workspace_anc"],
            "Output_Anc": meta["output_anc"],
            "Garbage": meta["garbage"],
            "Total_Qubits": n_total,
            "ST_Volume": st_vol,
            "Fid_3out": fid3,
            "TD_3out": td3,
            "Fid_4out": fid4,
            "TD_4out": td4,
            **cls,
        })

        del sv, dm3, dm4, qc
        gc.collect()

    return results, prob_tables


def main():
    os.makedirs("results", exist_ok=True)
    tracemalloc.start()

    print("=" * 96)
    print("0. ENVIRONMENT & REPRODUCIBILITY METADATA")
    print("=" * 96)
    print(f"  Python Version     : {sys.version.split()[0]} ({platform.platform()})")
    print(f"  Qiskit Version     : {qiskit.__version__}")
    print(f"  Qiskit-Aer Version : {AER_VERSION}")
    print(f"  NumPy Version      : {np.__version__}")
    print(f"  SciPy Version      : {scipy.__version__}")
    print("  Simulator Config   : AerSimulator(method='statevector'), noiseless, optimization_level=0")

    print("\n" + "=" * 96)
    print("1. UNITARY VERIFICATION OF LOCKED 7-T TOFFOLI DECOMPOSITIONS")
    print("=" * 96)
    verify_toffoli_decompositions()

    print("\n" + "=" * 96)
    print("2. PRIMARY BENCHMARK — STANDARD LOCKED 7-T DECOMPOSITION (6 CNOT, T-DEPTH 4 / TOFFOLI)")
    print("=" * 96)
    res_std, probs_std = evaluate_suite(apply_toffoli_standard_7t)

    header = (
        f"{'Architecture':<23} | {'T':>3} | {'TD':>2} | {'CNOT':>4} | {'Anc':>3} | {'TotQ':>4} | "
        f"{'Fid(3)':>8} | {'TDst(3)':>7} | {'Fid(4)':>8} | {'TDst(4)':>7} | {'ER%':>6} | {'Bias':>6} | {'NMED':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in res_std:
        print(
            f"{r['Model']:<23} | {r['T_Count']:>3} | {r['T_Depth']:>2} | {r['CNOT']:>4} | "
            f"{r['Ancillas']:>3} | {r['Total_Qubits']:>4} | {r['Fid_3out']:>8.6f} | "
            f"{r['TD_3out']:>7.6f} | {r['Fid_4out']:>8.6f} | {r['TD_4out']:>7.6f} | "
            f"{r['ER_pct']:>6.2f} | {r['Mean_Err']:>+6.2f} | {r['NMED']:>6.4f}"
        )

    print("\n" + "=" * 96)
    print("3. SENSITIVITY CHECK — AMY ET AL. (2013) DEPTH-OPTIMAL 7-T TOFFOLI (7 CNOT, T-DEPTH 3)")
    print("=" * 96)
    res_amy, _ = evaluate_suite(apply_toffoli_amy_7t)
    h2 = f"{'Architecture':<23} | {'T':>3} | {'T-Depth (Std->Amy)':>18} | {'CNOT (Std->Amy)':>15} | {'ST-Vol (Amy)':>12} | {'Fid(3) Check':>12}"
    print(h2)
    print("-" * len(h2))
    for rs, ra in zip(res_std, res_amy):
        print(
            f"{ra['Model']:<23} | {ra['T_Count']:>3} | "
            f"{rs['T_Depth']:>7} -> {ra['T_Depth']:<8} | "
            f"{rs['CNOT']:>5} -> {ra['CNOT']:<7} | "
            f"{ra['ST_Volume']:>12} | {ra['Fid_3out']:>12.6f}"
        )

    print("\n" + "=" * 96)
    print("4. TABLE 2 — 3-OUTPUT PROBABILITY DISTRIBUTION (Cout1, Cout2, Sum)")
    print("=" * 96)
    p_ex3 = probs_std["Exact-Parallel (3-FA)"]["p3"]
    p_hy3 = probs_std["Proposed Hybrid-1"]["p3"]
    print(f"{'Basis State |Sum,C2,C1>':<24} | {'Exact Prob':>10} | {'Hybrid Prob':>11} | {'|Delta|':>9}")
    print("-" * 64)
    for i in range(8):
        b = format(i, "03b")
        print(f"  |{b}>{'':<18} | {p_ex3[i]:>10.5f} | {p_hy3[i]:>11.5f} | {abs(p_ex3[i] - p_hy3[i]):>9.5f}")
    print(f"  Sum of |Delta| / 2 (Check Trace Distance) = {0.5 * sum(abs(a - b) for a, b in zip(p_ex3, p_hy3)):.6f}")

    json_path = os.path.join("results", "benchmark_results.json")
    csv_path = os.path.join("results", "benchmark_table1.csv")

    with open(json_path, "w") as f:
        json.dump({"standard_7t": res_std, "amy_7t": res_amy, "table2_probs": probs_std}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res_std[0].keys())
        writer.writeheader()
        writer.writerows(res_std)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("\n" + "=" * 96)
    print(f"DONE! Saved '{json_path}' and '{csv_path}'.")
    print(f"Peak Python Memory Tracked: {peak_mem / (1024 * 1024):.2f} MB")
    print("=" * 96)


if __name__ == "__main__":
    main()
