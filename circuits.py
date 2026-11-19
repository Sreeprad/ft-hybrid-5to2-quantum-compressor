#!/usr/bin/env python3
"""
Fault-Tolerant Clifford+T 5:2 Quantum Compressor Circuit Library.

Contains:
  1. Hardware-locked, ancilla-free 7-T Toffoli decompositions:
       - Standard 7-T (6 CNOT, T-depth 4, 2 H, 0 ancillas)
       - Amy et al. (2013) 7-T (7 CNOT, T-depth 3, 2 H, 0 ancillas)
  2. Exact DAG-based circuit T-depth analyzer.
  3. Unitary verification routine against Qiskit's CCXGate().
  4. Nine 5:2 quantum compressor benchmark architectures spanning the
     T-count vs. State Fidelity Pareto frontier.
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CCXGate
from qiskit.quantum_info import Operator


# ==============================================================================
# 1. LOCKED 7-T TOFFOLI DECOMPOSITIONS & DAG T-DEPTH ANALYZER
# ==============================================================================
def apply_toffoli_standard_7t(qc: QuantumCircuit, c1, c2, t) -> None:
    """
    Standard ancilla-free Clifford+T Toffoli decomposition (Nielsen & Chuang):
    T-count = 7, T-depth = 4, CNOT = 6, H = 2, Ancillas = 0.
    """
    qc.h(t)
    qc.cx(c2, t)
    qc.tdg(t)
    qc.cx(c1, t)
    qc.t(t)
    qc.cx(c2, t)
    qc.tdg(t)
    qc.cx(c1, t)
    qc.t(c2)
    qc.t(t)
    qc.cx(c1, c2)
    qc.h(t)
    qc.t(c1)
    qc.tdg(c2)
    qc.cx(c1, c2)


def apply_toffoli_amy_7t(qc: QuantumCircuit, c1, c2, t) -> None:
    """
    Depth-optimal ancilla-free Clifford+T Toffoli (Amy et al., IEEE TCAD 2013):
    T-count = 7, T-depth = 3, CNOT = 7, H = 2, Ancillas = 0.
    """
    qc.h(t)
    qc.t(c1)
    qc.t(c2)
    qc.t(t)
    qc.cx(c2, c1)
    qc.cx(t, c2)
    qc.cx(c1, t)
    qc.tdg(c1)
    qc.tdg(c2)
    qc.t(t)
    qc.cx(c1, t)
    qc.cx(c2, c1)
    qc.tdg(c1)
    qc.cx(t, c2)
    qc.cx(t, c1)
    qc.h(t)


def compute_t_depth(qc: QuantumCircuit) -> int:
    """
    Exact DAG-based circuit T-depth: longest path of non-Clifford {T, Tdg}
    layers respecting all multi-qubit Clifford data dependencies.
    """
    qubit_layer = {q: 0 for q in qc.qubits}
    for inst in qc.data:
        qargs = inst.qubits
        if not qargs:
            continue
        cur = max(qubit_layer[q] for q in qargs)
        nxt = cur + 1 if inst.operation.name in ("t", "tdg") else cur
        for q in qargs:
            qubit_layer[q] = nxt
    return max(qubit_layer.values()) if qubit_layer else 0


def verify_toffoli_decompositions() -> None:
    """Assert exact unitary equivalence of both 7-T decompositions to CCXGate()."""
    ref_op = Operator(CCXGate())
    specs = [
        ("Standard 7-T (6 CNOT, T-depth 4)", apply_toffoli_standard_7t, 7, 4, 6),
        ("Amy et al. 7-T (7 CNOT, T-depth 3)", apply_toffoli_amy_7t, 7, 3, 7),
    ]
    for name, fn, exp_t, exp_td, exp_cx in specs:
        qc = QuantumCircuit(3)
        fn(qc, 0, 1, 2)
        ops = qc.count_ops()
        t_cnt = ops.get("t", 0) + ops.get("tdg", 0)
        td = compute_t_depth(qc)
        cx_cnt = ops.get("cx", 0)
        is_exact = Operator(qc).equiv(ref_op)
        assert is_exact, f"Unitary mismatch in {name}!"
        assert (t_cnt, td, cx_cnt) == (exp_t, exp_td, exp_cx), (
            f"Gate count mismatch in {name}: T={t_cnt}, TD={td}, CX={cx_cnt}"
        )
        print(
            f"  [VERIFIED] {name}: Unitary Exact = {is_exact} | "
            f"T={t_cnt}, T-depth={td}, CNOT={cx_cnt}, H={ops.get('h', 0)}"
        )


# ==============================================================================
# 2. REVERSIBLE FULL ADDER PRIMITIVES
# ==============================================================================
def exact_reversible_fa(qc: QuantumCircuit, a, b, c, sum_target, cout_target, toffoli_fn) -> None:
    """Out-of-place Exact Reversible Full Adder (2 Toffolis, 5 CNOTs)."""
    qc.cx(a, sum_target)
    qc.cx(b, sum_target)
    qc.cx(c, sum_target)
    toffoli_fn(qc, a, b, cout_target)
    qc.cx(a, b)
    toffoli_fn(qc, b, c, cout_target)
    qc.cx(a, b)


def exact_reversible_fa_inplace_sum(qc: QuantumCircuit, a, b, c_inplace, cout_target, toffoli_fn) -> None:
    """In-place Exact Reversible Full Adder (2 Toffolis, 3 CNOTs, 0 intermediate ancilla)."""
    toffoli_fn(qc, a, b, cout_target)
    qc.cx(a, b)
    toffoli_fn(qc, b, c_inplace, cout_target)
    qc.cx(b, c_inplace)
    qc.cx(a, b)


# ==============================================================================
# 3. NINE 5:2 COMPRESSOR BENCHMARK ARCHITECTURES
#    Standard Interface:
#      Inputs  (7): x1, x2, x3, x4, x5, cin1, cin2 (weight 2^0)
#      Outputs (4): Cout1 (2^1), Cout2 (2^1), Sum (2^0), Carry (2^1)
#    All returned output index lists are strictly ascending for aligned partial trace.
# ==============================================================================
def build_exact_parallel(toffoli_fn):
    """[1] Exact Baseline (3-FA Parallel Cascade)."""
    q_in = QuantumRegister(7, "in")
    q_int = QuantumRegister(2, "int")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_int, q_out, name="Exact-Parallel")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    exact_reversible_fa(qc, x1, x2, x3, q_int[0], q_out[0], toffoli_fn)
    exact_reversible_fa(qc, x4, x5, q_int[0], q_int[1], q_out[1], toffoli_fn)
    exact_reversible_fa(qc, cin1, cin2, q_int[1], q_out[2], q_out[3], toffoli_fn)

    meta = {"workspace_anc": 2, "output_anc": 4, "toffoli_anc": 0, "garbage": 2}
    return qc, [9, 10, 11], [9, 10, 11, 12], meta


def build_exact_inplace(toffoli_fn):
    """[2] Exact Low-Ancilla Serial 3-FA (In-place intermediate sum + zero-T input restore)."""
    q_in = QuantumRegister(7, "in")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_out, name="Exact-InPlace")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    exact_reversible_fa_inplace_sum(qc, x1, x2, x3, q_out[0], toffoli_fn)
    exact_reversible_fa_inplace_sum(qc, x4, x5, x3, q_out[1], toffoli_fn)
    exact_reversible_fa(qc, cin1, cin2, x3, q_out[2], q_out[3], toffoli_fn)
    qc.cx(x1, x3)
    qc.cx(x2, x3)
    qc.cx(x4, x3)
    qc.cx(x5, x3)

    meta = {"workspace_anc": 0, "output_anc": 4, "toffoli_anc": 0, "garbage": 0}
    return qc, [7, 8, 9], [7, 8, 9, 10], meta


def build_single_and_hybrid(toffoli_fn):
    """[3] Single-AND Stage-1 Hybrid (Cout1 = x1 & x2 [1 Toffoli] + 2 Exact FAs)."""
    q_in = QuantumRegister(7, "in")
    q_int = QuantumRegister(2, "int")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_int, q_out, name="Single-AND-Hybrid")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    qc.cx(x1, q_int[0]); qc.cx(x2, q_int[0]); qc.cx(x3, q_int[0])
    toffoli_fn(qc, x1, x2, q_out[0])
    exact_reversible_fa(qc, x4, x5, q_int[0], q_int[1], q_out[1], toffoli_fn)
    exact_reversible_fa(qc, cin1, cin2, q_int[1], q_out[2], q_out[3], toffoli_fn)

    meta = {"workspace_anc": 2, "output_anc": 4, "toffoli_anc": 0, "garbage": 2}
    return qc, [9, 10, 11], [9, 10, 11, 12], meta


def build_proposed_hybrid_1(toffoli_fn):
    """[4] Proposed Hybrid-1 (Zero-Toffoli Stage 1 + 2 Exact FAs)."""
    q_in = QuantumRegister(7, "in")
    q_int = QuantumRegister(2, "int")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_int, q_out, name="Proposed-Hybrid-1")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    qc.cx(x1, q_int[0]); qc.cx(x2, q_int[0]); qc.cx(x3, q_int[0])
    qc.cx(x3, q_out[0])
    exact_reversible_fa(qc, x4, x5, q_int[0], q_int[1], q_out[1], toffoli_fn)
    exact_reversible_fa(qc, cin1, cin2, q_int[1], q_out[2], q_out[3], toffoli_fn)

    meta = {"workspace_anc": 2, "output_anc": 4, "toffoli_anc": 0, "garbage": 2}
    return qc, [9, 10, 11], [9, 10, 11, 12], meta


def build_proposed_hybrid_1_la(toffoli_fn):
    """[5] Proposed Hybrid-1-LA (Wire-Aliased Low-Ancilla Variant: x3 acts as Cout1)."""
    q_in = QuantumRegister(7, "in")
    q_int = QuantumRegister(2, "int")
    q_out = QuantumRegister(3, "out")
    qc = QuantumCircuit(q_in, q_int, q_out, name="Proposed-Hybrid-1-LA")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    qc.cx(x1, q_int[0]); qc.cx(x2, q_int[0]); qc.cx(x3, q_int[0])
    exact_reversible_fa(qc, x4, x5, q_int[0], q_int[1], q_out[0], toffoli_fn)
    exact_reversible_fa(qc, cin1, cin2, q_int[1], q_out[1], q_out[2], toffoli_fn)

    meta = {"workspace_anc": 2, "output_anc": 3, "toffoli_anc": 0, "garbage": 2}
    return qc, [2, 9, 10], [2, 9, 10, 11], meta


def build_published_afa_52(toffoli_fn):
    """[6] Published Pairwise-AND Approximate Full-Adder 5:2 Compressor mapped to Clifford+T."""
    q_in = QuantumRegister(7, "in")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_out, name="Published-AFA-5:2")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    for q in q_in:
        qc.cx(q, q_out[2])
    toffoli_fn(qc, x1, x2, q_out[0])
    toffoli_fn(qc, x4, cin1, q_out[1])
    toffoli_fn(qc, x5, cin2, q_out[3])

    meta = {"workspace_anc": 0, "output_anc": 4, "toffoli_anc": 0, "garbage": 0}
    return qc, [7, 8, 9], [7, 8, 9, 10], meta


def build_hybrid_2(toffoli_fn):
    """[7] Hybrid-2 (2-Stage Zero-Toffoli + 1 Exact FA — 14 T-gate Pareto point)."""
    q_in = QuantumRegister(7, "in")
    q_int = QuantumRegister(2, "int")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_int, q_out, name="Hybrid-2-(14T)")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    qc.cx(x1, q_int[0]); qc.cx(x2, q_int[0]); qc.cx(x3, q_int[0])
    qc.cx(x3, q_out[0])
    qc.cx(x4, q_int[1]); qc.cx(x5, q_int[1]); qc.cx(q_int[0], q_int[1])
    qc.cx(x5, q_out[1])
    exact_reversible_fa(qc, cin1, cin2, q_int[1], q_out[2], q_out[3], toffoli_fn)

    meta = {"workspace_anc": 2, "output_anc": 4, "toffoli_anc": 0, "garbage": 2}
    return qc, [9, 10, 11], [9, 10, 11, 12], meta


def build_extreme_zero_t_routing(toffoli_fn):
    """[8] Extreme Zero-T Clifford Routing (all 3 carries wire-routed, 0 T-gates)."""
    q_in = QuantumRegister(7, "in")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_out, name="Extreme-ZeroT-Route")
    for i in range(7):
        qc.h(q_in[i])

    x1, x2, x3, x4, x5, cin1, cin2 = q_in
    for q in q_in:
        qc.cx(q, q_out[2])
    qc.cx(x3, q_out[0])
    qc.cx(x5, q_out[1])
    qc.cx(cin2, q_out[3])

    meta = {"workspace_anc": 0, "output_anc": 4, "toffoli_anc": 0, "garbage": 0}
    return qc, [7, 8, 9], [7, 8, 9, 10], meta


def build_nisq_nocarry_aqanc(toffoli_fn):
    """[9] NISQ Zero-T No-Carry Truncation (AQANC-style baseline)."""
    q_in = QuantumRegister(7, "in")
    q_out = QuantumRegister(4, "out")
    qc = QuantumCircuit(q_in, q_out, name="NISQ-AQANC-NoCarry")
    for i in range(7):
        qc.h(q_in[i])

    for q in q_in:
        qc.cx(q, q_out[2])

    meta = {"workspace_anc": 0, "output_anc": 4, "toffoli_anc": 0, "garbage": 0}
    return qc, [7, 8, 9], [7, 8, 9, 10], meta


BENCHMARK_BUILDERS = [
    ("Exact-Parallel (3-FA)", build_exact_parallel),
    ("Exact-InPlace (Serial)", build_exact_inplace),
    ("Single-AND Hybrid", build_single_and_hybrid),
    ("Proposed Hybrid-1", build_proposed_hybrid_1),
    ("Proposed Hybrid-1-LA", build_proposed_hybrid_1_la),
    ("Published AFA-5:2", build_published_afa_52),
    ("Hybrid-2 (14T)", build_hybrid_2),
    ("Extreme Zero-T Route", build_extreme_zero_t_routing),
    ("NISQ AQANC No-Carry", build_nisq_nocarry_aqanc),
]
