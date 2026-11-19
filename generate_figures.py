#!/usr/bin/env python3
"""
Publication-Grade Figure Generator (300-DPI PNG + Embedded-Font Vector PDF).
Reads 'results/benchmark_results.json' produced by benchmark.py and writes
figures to 'results/figures/'.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from circuits import apply_toffoli_standard_7t

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RESULTS_JSON = os.path.join("results", "benchmark_results.json")
FIG_DIR = os.path.join("results", "figures")


def load_results():
    if not os.path.exists(RESULTS_JSON):
        sys.exit(f"ERROR: '{RESULTS_JSON}' not found. Run 'python benchmark.py' first.")
    with open(RESULTS_JSON, "r") as f:
        return json.load(f)


def make_pareto_figure(res_std):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)

    pareto_names = [
        "Extreme Zero-T Route",
        "Hybrid-2 (14T)",
        "Proposed Hybrid-1-LA",
        "Exact-Parallel (3-FA)",
    ]
    by_name = {r["Model"]: r for r in res_std}

    px = [by_name[n]["T_Count"] for n in pareto_names]
    py = [by_name[n]["Fid_4out"] for n in pareto_names]
    ax1.plot(px, py, color="#1f4e79", linestyle="-", linewidth=1.8, zorder=2, label="Pareto Frontier (T-Count vs. $F_4$)")

    style_map = {
        "Exact-Parallel (3-FA)": ("#c0392b", "s", 80, (-65, -16), "Exact-Parallel\n(42T, 6 Anc)"),
        "Exact-InPlace (Serial)": ("#8e44ad", "D", 70, (-88, -28), "Exact-InPlace\n(42T, 4 Anc, TD=16)"),
        "Single-AND Hybrid": ("#d35400", "X", 85, (-55, -24), "Single-AND Hybrid\n(35T, Dominated)"),
        "Proposed Hybrid-1": ("#27ae60", "o", 85, (-12, -26), "Proposed Hybrid-1\n(28T, 6 Anc)"),
        "Proposed Hybrid-1-LA": ("#16a085", "*", 175, (-72, 10), "Proposed Hybrid-1-LA\n(28T, 5 Anc, $F_4$=0.982)"),
        "Published AFA-5:2": ("#7f8c8d", "v", 80, (8, -6), "Pairwise-AND AFA\n(21T, TD=4, Biased)"),
        "Hybrid-2 (14T)": ("#2980b9", "^", 85, (8, -14), "Hybrid-2 (14T)\n($F_4$=0.958)"),
        "Extreme Zero-T Route": ("#34495e", "P", 85, (8, -12), "Extreme Zero-T\n(0T, $F_4$=0.887)"),
    }

    for name, (col, mk, sz, offset, txt) in style_map.items():
        r = by_name[name]
        x, y = r["T_Count"], r["Fid_4out"]
        x_plot = x - 0.5 if name == "Exact-InPlace (Serial)" else x
        y_plot = y - 0.006 if name == "Proposed Hybrid-1" else y
        ax1.scatter(x_plot, y_plot, c=col, marker=mk, s=sz, edgecolors="black", linewidths=0.7, zorder=4)
        ax1.annotate(
            txt,
            xy=(x_plot, y_plot),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.6,
            fontweight="bold" if "Proposed" in name else "normal",
            color="#0e6251" if "Proposed" in name else "#222222",
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6),
        )

    ax1.annotate("", xy=(28, 1.015), xytext=(42, 1.015), arrowprops=dict(arrowstyle="<->", color="#16a085", lw=1.5))
    ax1.text(35, 1.021, "−33.3% T-Count", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#16a085")
    ax1.text(
        0.98, 0.04,
        "Note: NISQ AQANC No-Carry (0T, $F_4$=0.052, ER=93.8%)\nlies far below the plotted fidelity axis.",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )
    ax1.set_xlim(-3, 46)
    ax1.set_ylim(0.64, 1.055)
    ax1.set_xlabel("Fault-Tolerant T-Count (Locked 7-T Toffoli Model)")
    ax1.set_ylabel("Full 4-Output State Fidelity ($F_4$)")
    ax1.set_title("(a) T-Count vs. Quantum State Fidelity Pareto Frontier")
    ax1.legend(loc="lower left", frameon=True)

    comp_models = [
        ("Exact-Parallel (3-FA)", "Exact\n(42T)"),
        ("Single-AND Hybrid", "Single-AND\n(35T)"),
        ("Proposed Hybrid-1-LA", "Proposed\nHybrid-1-LA\n(28T)"),
        ("Published AFA-5:2", "Pairwise-AND\nAFA (21T)"),
        ("Hybrid-2 (14T)", "Hybrid-2\n(14T)"),
        ("Extreme Zero-T Route", "Extreme\nZero-T (0T)"),
        ("NISQ AQANC No-Carry", "NISQ\nAQANC (0T)"),
    ]
    labels = [lbl for _, lbl in comp_models]
    nmed_vals = [by_name[m]["NMED"] * 100.0 for m, _ in comp_models]
    er_vals = [by_name[m]["ER_pct"] for m, _ in comp_models]
    bias_vals = [by_name[m]["Mean_Err"] for m, _ in comp_models]

    x_idx = np.arange(len(labels))
    w = 0.36
    colors_nmed = ["#16a085" if "Proposed" in m else "#3b7ea1" for m, _ in comp_models]
    colors_er = ["#76d7c4" if "Proposed" in m else "#b0c4de" for m, _ in comp_models]

    ax2.bar(x_idx - w / 2, er_vals, width=w, color=colors_er, edgecolor="black", linewidth=0.6, label="Error Rate — ER (%)")
    ax2.bar(x_idx + w / 2, nmed_vals, width=w, color=colors_nmed, edgecolor="black", linewidth=0.6, label="NMED (%)")

    for i, (er, nm, bs) in enumerate(zip(er_vals, nmed_vals, bias_vals)):
        ax2.text(x_idx[i] - w / 2, er + 1.5, f"{er:.1f}%", ha="center", va="bottom", fontsize=6.8)
        ax2.text(x_idx[i] + w / 2, nm + 1.5, f"{nm:.1f}%", ha="center", va="bottom", fontsize=6.8)
        bias_lbl = "Bias=0" if abs(bs) < 1e-9 else f"Bias={bs:+.1f}"
        ax2.text(
            x_idx[i], max(er, nm) + 8.5, bias_lbl,
            ha="center", va="bottom", fontsize=6.9,
            fontweight="bold" if abs(bs) < 1e-9 else "normal",
            color="#0e6251" if abs(bs) < 1e-9 else "#922b21",
        )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels, fontsize=7.6)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Error Metric (%)")
    ax2.set_title("(b) Classical Error Rate, NMED, and Mean Error Bias")
    ax2.legend(loc="upper left", frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_pareto_frontier.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] fig_pareto_frontier.pdf / .png")


def make_toffoli_figure():
    q = QuantumRegister(3, "q")
    qc = QuantumCircuit(q)
    apply_toffoli_standard_7t(qc, q[0], q[1], q[2])

    fig = qc.draw(output="mpl", fold=-1, scale=0.95)
    fig.suptitle(
        "Hardware-Locked Ancilla-Free 7-T Toffoli Decomposition (T-Count = 7, T-Depth = 4, CNOT = 6, H = 2, Ancillas = 0)",
        fontsize=9.5, y=0.98,
    )
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_toffoli_7t.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] fig_toffoli_7t.pdf / .png")


def make_circuit_stages_figure():
    q1 = QuantumRegister(5, "q")
    qc1 = QuantumCircuit(q1)
    qc1.cx(0, 3); qc1.cx(1, 3); qc1.cx(2, 3); qc1.cx(2, 4)

    q2 = QuantumRegister(5, "q")
    qc2 = QuantumCircuit(q2)
    qc2.cx(0, 3); qc2.cx(1, 3); qc2.cx(2, 3)
    qc2.ccx(0, 1, 4); qc2.cx(0, 1); qc2.ccx(1, 2, 4); qc2.cx(0, 1)

    q_in = QuantumRegister(7, "in")
    q_int = QuantumRegister(2, "int")
    q_out = QuantumRegister(4, "out")
    qc_full = QuantumCircuit(q_in, q_int, q_out)
    x1, x2, x3, x4, x5, cin1, cin2 = q_in

    qc_full.cx(x1, q_int[0]); qc_full.cx(x2, q_int[0]); qc_full.cx(x3, q_int[0]); qc_full.cx(x3, q_out[0])
    qc_full.barrier(label="Stage 1 (0T)")
    qc_full.cx(x4, q_int[1]); qc_full.cx(x5, q_int[1]); qc_full.cx(q_int[0], q_int[1])
    qc_full.ccx(x4, x5, q_out[1]); qc_full.cx(x4, x5); qc_full.ccx(x5, q_int[0], q_out[1]); qc_full.cx(x4, x5)
    qc_full.barrier(label="Stage 2 (14T)")
    qc_full.cx(cin1, q_out[2]); qc_full.cx(cin2, q_out[2]); qc_full.cx(q_int[1], q_out[2])
    qc_full.ccx(cin1, cin2, q_out[3]); qc_full.cx(cin1, cin2); qc_full.ccx(cin2, q_int[1], q_out[3]); qc_full.cx(cin1, cin2)

    specs = [
        (qc1, "fig_stage1_zero_toffoli", "Stage 1: Zero-Toffoli Approximate Front-End (0 T-Gates, 4 CNOTs)", 0.9, False),
        (qc2, "fig_stage23_exact_fa", "Stages 2 & 3: Exact Reversible Full Adder Cell (2 Toffolis = 14 T-Gates, 5 CNOTs)", 0.9, False),
        (qc_full, "fig_hybrid_full_circuit", "Complete 13-Qubit Proposed Hybrid-1 5:2 Compressor (4 Toffolis = 28 T-Gates, 38 CNOTs, 6 Ancillas)", 0.75, True),
    ]
    for circ, fname, title, sc, use_suptitle in specs:
        fig = circ.draw(output="mpl", fold=-1, scale=sc)
        if use_suptitle:
            fig.suptitle(title, fontsize=10.0, y=0.99)
        else:
            fig.axes[0].set_title(title, fontsize=9.5)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG_DIR, f"{fname}.{ext}"), bbox_inches="tight")
        plt.close(fig)
    print("  [SAVED] fig_stage1_zero_toffoli, fig_stage23_exact_fa, fig_hybrid_full_circuit (.pdf/.png)")


def make_prob_and_error_figure(probs_std):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    p_ex = probs_std["Exact-Parallel (3-FA)"]["p3"]
    p_hy = probs_std["Proposed Hybrid-1"]["p3"]
    states = [f"$|{format(i, '03b')}\\rangle$" for i in range(8)]
    x = np.arange(8)
    w = 0.38

    ax1.bar(x - w / 2, p_ex, width=w, color="#2c3e50", edgecolor="black", lw=0.6, label="Exact Baseline (42 T)")
    ax1.bar(x + w / 2, p_hy, width=w, color="#16a085", edgecolor="black", lw=0.6, label="Proposed Hybrid-1 (28 T)")
    ax1.axhline(0.125, color="#16a085", linestyle=":", lw=1.0, alpha=0.8)

    for i in range(8):
        ax1.text(x[i] - w / 2, p_ex[i] + 0.004, f"{p_ex[i]:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax1.text(x[i] + w / 2, p_hy[i] + 0.004, f"{p_hy[i]:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(states, fontsize=8.5)
    ax1.set_ylim(0, 0.215)
    ax1.set_ylabel("Output Measurement Probability")
    ax1.set_xlabel("Isolated 3-Qubit Output State $|\\mathrm{Sum}, C_{\\mathrm{out}2}, C_{\\mathrm{out}1}\\rangle$")
    ax1.set_title("(a) Output Probability Distribution ($|\\Delta| = 0.03125$, $D = 0.125$)")
    ax1.legend(loc="upper right", frameon=True)

    hyb_counts = [16, 96, 16]
    and_counts = [32, 96, 0]
    xb = np.arange(3)
    wb = 0.34

    ax2.bar(xb - wb / 2, hyb_counts, width=wb, color="#16a085", edgecolor="black", lw=0.6, label="Proposed Hybrid-1 (28 T, Bias = 0.00)")
    ax2.bar(xb + wb / 2, and_counts, width=wb, color="#d35400", edgecolor="black", lw=0.6, alpha=0.85, label="Single-AND Hybrid (35 T, Bias = −0.50)")

    for i in range(3):
        ax2.text(xb[i] - wb / 2, hyb_counts[i] + 2.0, f"{hyb_counts[i]}/128\n({hyb_counts[i] / 1.28:.1f}%)", ha="center", va="bottom", fontsize=7.5, color="#0e6251", fontweight="bold")
        ax2.text(xb[i] + wb / 2, and_counts[i] + 2.0, f"{and_counts[i]}/128\n({and_counts[i] / 1.28:.1f}%)", ha="center", va="bottom", fontsize=7.5, color="#a04000")

    ax2.set_xticks(xb)
    ax2.set_xticklabels(["Error = −2", "Exact (Error = 0)", "Error = +2"], fontsize=8.8)
    ax2.set_ylim(0, 120)
    ax2.set_ylabel("Number of Input Combinations (out of $2^7 = 128$)")
    ax2.set_xlabel("Arithmetic Error Distance ($V_{\\mathrm{approx}} - V_{\\mathrm{exact}}$)")
    ax2.set_title("(b) Symmetric Zero-Bias Error Distribution Across All 128 States")
    ax2.legend(loc="upper right", frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_prob_and_error.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  [SAVED] fig_prob_and_error.pdf / .png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_results()
    print("=== GENERATING PUBLICATION-QUALITY FIGURES (300 DPI + VECTOR PDF) ===")
    make_pareto_figure(data["standard_7t"])
    make_toffoli_figure()
    make_circuit_stages_figure()
    make_prob_and_error_figure(data["table2_probs"])
    print(f"=== ALL FIGURES SAVED TO '{FIG_DIR}/' ===")


if __name__ == "__main__":
    main()
