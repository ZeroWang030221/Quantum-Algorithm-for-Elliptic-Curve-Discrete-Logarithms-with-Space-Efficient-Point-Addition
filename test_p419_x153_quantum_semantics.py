"""Gate-level semantic regression for the revised circuit at p=419, x=153.

This test does *not* call a classical EEA implementation to produce the circuit
output.  It builds the repository's actual forward/inverse Qiskit instructions,
recursively executes every emitted primitive gate (including H, measurement,
reset, and classically conditioned Z/CZ), and only uses Python's ``pow`` as an
independent assertion oracle.

Two tests are performed:

1. Basis-state forward semantics:
       |153>|0>|0> -> |241>|0>|Gamma(153)>
   over several independent measurement trajectories.

2. Coherent round trip with an external reference qubit:
       (|0>|153> + |1>|155>)/sqrt(2)
         -- EEA --> -- EEA^dagger -->
       (|0>|153> + |1>|155>)/sqrt(2)
   with the large workspace and every shared ancilla returned to zero.
   This catches relative-phase/coherence errors that a layout-only or
   basis-cleanup-only test cannot detect.

When real Qiskit is installed, the production modules are imported against real
Qiskit circuit objects.  In restricted environments without Qiskit, the script
uses ``mini_qiskit_runtime`` only as a circuit data model; the same independent
quantum-state trajectory engine still executes the emitted gate definitions.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import cmath
import math
import os
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Import the production code.  The fallback is only a Qiskit-compatible circuit
# container; it contains no EEA or arithmetic semantics.
# ---------------------------------------------------------------------------
FORCE_MINI_QISKIT = os.environ.get("P419_FORCE_MINI_QISKIT", "0") == "1"

if not FORCE_MINI_QISKIT:
    try:
        import qiskit  # type: ignore
        USING_REAL_QISKIT = True
    except Exception:
        import mini_qiskit_runtime as _mini
        _mini.install_as_qiskit()
        import qiskit  # type: ignore
        USING_REAL_QISKIT = False
else:
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()
    import qiskit  # type: ignore
    USING_REAL_QISKIT = False

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # type: ignore

import under1000_eea_shared_s835_fastdual_wrapped as shared_eea


P = 419
X0 = 153
X1 = 155  # default second branch
BRANCH_BIT = 1
N = P.bit_length()
EXPECTED_UNIFORM_T_MAX = 60
T_MAX = int(shared_eea.shared_eea_layout(N).T_max)
if T_MAX != EXPECTED_UNIFORM_T_MAX:
    raise AssertionError(
        f"current production code reports T_max={T_MAX} for n={N}; "
        f"expected the revised-paper uniform schedule {EXPECTED_UNIFORM_T_MAX}"
    )
COMPUTATIONAL_STEPS = 56


# ---------------------------------------------------------------------------
# Generic circuit access helpers (work with real Qiskit and mini runtime).
# ---------------------------------------------------------------------------
def _items(circuit):
    for item in circuit.data:
        if hasattr(item, "operation"):
            yield item.operation, tuple(item.qubits), tuple(item.clbits)
        else:  # old Qiskit tuple form
            op, qargs, cargs = item
            yield op, tuple(qargs), tuple(cargs)


def _find_index(circuit, bit) -> int:
    return int(circuit.find_bit(bit).index)


def _normalize_name(op) -> str:
    base = getattr(op, "base_name", None)
    name = str(base if base is not None else op.name).lower()
    while name.endswith("_dg"):
        name = name[:-3]
    aliases = {"cnot": "cx", "tof": "ccx", "toffoli": "ccx"}
    return aliases.get(name, name)


def _condition_value(condition, classical: Sequence[int], c_map: Sequence[int]) -> bool:
    if condition is None:
        return True
    lhs, expected = condition
    # Clbit-like object.
    if hasattr(lhs, "register") and hasattr(lhs, "index"):
        # ``lhs`` belongs to the current definition; c_map translates it.
        local_idx = lhs.register._bits.index(lhs) if hasattr(lhs.register, "_bits") else int(lhs.index)
        # The register may not start at local index zero in the circuit.  Callers
        # replace Clbit conditions by a mapped global index before this function
        # when real-Qiskit metadata is available.  For mini-qiskit, find via map.
        if hasattr(lhs, "_semantic_local_index"):
            local_idx = int(lhs._semantic_local_index)
        if local_idx >= len(c_map):
            raise IndexError(f"condition local clbit index {local_idx} out of range {len(c_map)}")
        return int(classical[c_map[local_idx]]) == int(expected)
    # ClassicalRegister-like condition: Qiskit interprets little endian.
    try:
        bits = list(lhs)
    except TypeError as exc:
        raise TypeError(f"Unsupported condition lhs: {lhs!r}") from exc
    value = 0
    for i, bit in enumerate(bits):
        local_idx = int(getattr(bit, "index", i))
        value |= int(classical[c_map[local_idx]]) << i
    return value == int(expected)


# ---------------------------------------------------------------------------
# Sparse pure-state dynamic-circuit trajectory simulator.
# ---------------------------------------------------------------------------
@dataclass
class SimulationStats:
    primitive_counts: Counter = field(default_factory=Counter)
    composite_calls: int = 0
    max_support: int = 0
    measurements: int = 0
    resets: int = 0


@dataclass
class TrajectoryResult:
    state: Dict[int, complex]
    classical: List[int]
    measurement_record: List[int]
    measurement_probabilities: List[float]
    stats: SimulationStats


class SparseTrajectorySimulator:
    """Execute the emitted quantum circuit, not a classical arithmetic model."""

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        *,
        initial_state: Optional[Mapping[int, complex]] = None,
        seed: int = 0,
        forced_measurements: Optional[Sequence[int]] = None,
        atol: float = 1e-12,
    ):
        self.num_qubits = int(num_qubits)
        self.num_clbits = int(num_clbits)
        self.state: Dict[int, complex] = dict(initial_state or {0: 1.0 + 0.0j})
        self.classical = [0] * self.num_clbits
        self.rng = random.Random(seed)
        self.forced = list(forced_measurements) if forced_measurements is not None else None
        self.forced_pos = 0
        self.record: List[int] = []
        self.measurement_probabilities: List[float] = []
        self.stats = SimulationStats(max_support=len(self.state))
        self.atol = float(atol)
        self._normalize()

    def _normalize(self):
        norm2 = sum(abs(a) ** 2 for a in self.state.values())
        if norm2 <= self.atol:
            raise AssertionError("quantum state has zero norm")
        scale = 1.0 / math.sqrt(norm2)
        self.state = {b: a * scale for b, a in self.state.items() if abs(a) > self.atol}
        self.stats.max_support = max(self.stats.max_support, len(self.state))

    def _permute(self, transform):
        out: Dict[int, complex] = {}
        for basis, amp in self.state.items():
            dst = transform(basis)
            out[dst] = out.get(dst, 0.0j) + amp
        self.state = {b: a for b, a in out.items() if abs(a) > self.atol}
        self.stats.max_support = max(self.stats.max_support, len(self.state))

    def _phase(self, phase_fn):
        self.state = {b: a * phase_fn(b) for b, a in self.state.items()}

    def _x(self, q: int):
        mask = 1 << q
        self._permute(lambda b: b ^ mask)

    def _cx(self, c: int, t: int):
        cm, tm = 1 << c, 1 << t
        self._permute(lambda b: b ^ tm if (b & cm) else b)

    def _ccx(self, controls: Sequence[int], target: int):
        masks = [1 << c for c in controls]
        tm = 1 << target
        self._permute(lambda b: b ^ tm if all(b & m for m in masks) else b)

    def _swap(self, a: int, b: int):
        if a == b:
            return
        am, bm = 1 << a, 1 << b
        def f(x):
            ba = 1 if x & am else 0
            bb = 1 if x & bm else 0
            return x ^ am ^ bm if ba != bb else x
        self._permute(f)

    def _h(self, q: int):
        mask = 1 << q
        s = 1.0 / math.sqrt(2.0)
        out: Dict[int, complex] = {}
        for basis, amp in self.state.items():
            if basis & mask:
                b0 = basis ^ mask
                out[b0] = out.get(b0, 0.0j) + amp * s
                out[basis] = out.get(basis, 0.0j) - amp * s
            else:
                b1 = basis ^ mask
                out[basis] = out.get(basis, 0.0j) + amp * s
                out[b1] = out.get(b1, 0.0j) + amp * s
        self.state = {b: a for b, a in out.items() if abs(a) > self.atol}
        self.stats.max_support = max(self.stats.max_support, len(self.state))

    def _measure(self, q: int, c: int):
        mask = 1 << q
        p1 = sum(abs(a) ** 2 for b, a in self.state.items() if b & mask)
        p1 = min(1.0, max(0.0, p1))
        if self.forced is not None:
            if self.forced_pos >= len(self.forced):
                raise AssertionError("forced measurement record is too short")
            outcome = int(self.forced[self.forced_pos])
            self.forced_pos += 1
        else:
            outcome = 1 if self.rng.random() < p1 else 0
        prob = p1 if outcome else (1.0 - p1)
        if prob <= self.atol:
            raise AssertionError(
                f"forced/implied measurement outcome {outcome} has probability {prob:.3e}"
            )
        self.state = {
            b: a / math.sqrt(prob)
            for b, a in self.state.items()
            if bool(b & mask) == bool(outcome)
        }
        self.classical[c] = outcome
        self.record.append(outcome)
        self.measurement_probabilities.append(prob)
        self.stats.measurements += 1
        self.stats.max_support = max(self.stats.max_support, len(self.state))

    def _reset(self, q: int):
        mask = 1 << q
        # In this codebase reset immediately follows a measurement of the same
        # qubit, so it is already classical.  The implementation below is still
        # a valid trajectory reset for a general sparse pure state.
        p1 = sum(abs(a) ** 2 for b, a in self.state.items() if b & mask)
        if p1 > self.atol and p1 < 1.0 - self.atol:
            outcome = 1 if self.rng.random() < p1 else 0
            prob = p1 if outcome else 1.0 - p1
            self.state = {
                b: a / math.sqrt(prob)
                for b, a in self.state.items()
                if bool(b & mask) == bool(outcome)
            }
        elif p1 >= 1.0 - self.atol:
            outcome = 1
        else:
            outcome = 0
        if outcome:
            self._x(q)
        self.stats.resets += 1

    def _condition_is_true(self, op, definition, c_map: Sequence[int]) -> bool:
        cond = getattr(op, "condition", None)
        if cond is None:
            return True
        lhs, expected = cond
        # Resolve a single Clbit by asking its owning definition for the local
        # flat index.  This works for real Qiskit and the mini data model.
        if not hasattr(lhs, "__iter__") or hasattr(lhs, "register"):
            local = _find_index(definition, lhs)
            return self.classical[c_map[local]] == int(expected)
        value = 0
        for i, bit in enumerate(lhs):
            local = _find_index(definition, bit)
            value |= self.classical[c_map[local]] << i
        return value == int(expected)

    def _run_circuit(self, circuit, q_map: Sequence[int], c_map: Sequence[int]):
        for op, qargs, cargs in _items(circuit):
            qids = [q_map[_find_index(circuit, q)] for q in qargs]
            cids = [c_map[_find_index(circuit, c)] for c in cargs]
            name = _normalize_name(op)
            definition = getattr(op, "definition", None)

            # Qiskit 2.x represents real-time classical feed-forward with an
            # IfElseOp.  Execute exactly one block using the current classical
            # state.  The block bit order follows the operation qargs/cargs.
            if name == "if_else" and hasattr(op, "blocks"):
                self.stats.composite_calls += 1
                take_true = self._condition_is_true(op, circuit, c_map)
                blocks = list(op.blocks)
                selected = blocks[0] if take_true else (blocks[1] if len(blocks) > 1 else None)
                if selected is not None:
                    if selected.num_qubits != len(qids) or selected.num_clbits != len(cids):
                        raise AssertionError(
                            f"bad IfElseOp mapping: block q/c={selected.num_qubits}/{selected.num_clbits}, "
                            f"mapped={len(qids)}/{len(cids)}"
                        )
                    self._run_circuit(selected, qids, cids)
                continue

            if not self._condition_is_true(op, circuit, c_map):
                continue

            if definition is not None and name not in {
                "x", "cx", "ccx", "mcx", "swap", "h", "z", "cz",
                "measure", "reset", "barrier", "id",
            }:
                self.stats.composite_calls += 1
                if len(qids) != int(op.num_qubits) or len(cids) != int(op.num_clbits):
                    raise AssertionError(f"bad mapping for composite {op.name}")
                self._run_circuit(definition, qids, cids)
                continue

            self.stats.primitive_counts[name] += 1
            if name == "x":
                self._x(qids[0])
            elif name == "cx":
                self._cx(qids[0], qids[1])
            elif name in {"ccx", "mcx"}:
                self._ccx(qids[:-1], qids[-1])
            elif name == "swap":
                self._swap(qids[0], qids[1])
            elif name == "h":
                self._h(qids[0])
            elif name == "z":
                qm = 1 << qids[0]
                self._phase(lambda b, m=qm: -1.0 if (b & m) else 1.0)
            elif name == "cz":
                m0, m1 = 1 << qids[0], 1 << qids[1]
                self._phase(lambda b, a=m0, d=m1: -1.0 if (b & a and b & d) else 1.0)
            elif name == "measure":
                self._measure(qids[0], cids[0])
            elif name == "reset":
                self._reset(qids[0])
            elif name in {"barrier", "id"}:
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported primitive leaf {op.name!r}; definition={definition!r}"
                )

    def run(self, circuit) -> TrajectoryResult:
        if circuit.num_qubits != self.num_qubits or circuit.num_clbits != self.num_clbits:
            raise ValueError("simulator/circuit width mismatch")
        self._run_circuit(circuit, list(range(self.num_qubits)), list(range(self.num_clbits)))
        if self.forced is not None and self.forced_pos != len(self.forced):
            raise AssertionError(
                f"forced measurement record has {len(self.forced)-self.forced_pos} unused outcomes"
            )
        self._normalize()
        return TrajectoryResult(
            state=dict(self.state),
            classical=list(self.classical),
            measurement_record=list(self.record),
            measurement_probabilities=list(self.measurement_probabilities),
            stats=self.stats,
        )


# ---------------------------------------------------------------------------
# Test-circuit construction and assertions.
# ---------------------------------------------------------------------------
@dataclass
class TestCircuit:
    circuit: QuantumCircuit
    ref: Optional[QuantumRegister]
    X: QuantumRegister
    A: QuantumRegister
    S: QuantumRegister
    c_fwd: ClassicalRegister
    c_inv: Optional[ClassicalRegister]


def _set_le_constant(qc: QuantumCircuit, reg: QuantumRegister, value: int):
    for i, q in enumerate(reg):
        if (int(value) >> i) & 1:
            qc.x(q)


def build_forward_circuit(x: int = X0) -> TestCircuit:
    layout = shared_eea.shared_eea_layout(N, T_max=T_MAX)
    X = QuantumRegister(N, "X")
    A = QuantumRegister(N, "A")
    S = QuantumRegister(layout.s_qubits, "S")
    c_fwd = ClassicalRegister(N, "c_fwd")
    qc = QuantumCircuit(X, A, S, c_fwd, name="P419_X153_FORWARD_SEMANTICS")
    _set_le_constant(qc, X, x)
    fwd = shared_eea.eea_forward_shared_instruction(N, P, T_max=T_MAX, lazy_definition=True)
    qc.append(fwd, [*X, *A, *S], [*c_fwd])
    return TestCircuit(qc, None, X, A, S, c_fwd, None)


def build_coherent_forward_circuit() -> TestCircuit:
    layout = shared_eea.shared_eea_layout(N, T_max=T_MAX)
    ref = QuantumRegister(1, "ref")
    X = QuantumRegister(N, "X")
    A = QuantumRegister(N, "A")
    S = QuantumRegister(layout.s_qubits, "S")
    c_fwd = ClassicalRegister(N, "c_fwd")
    qc = QuantumCircuit(ref, X, A, S, c_fwd, name="P419_COHERENT_FORWARD_SEMANTICS")
    _set_le_constant(qc, X, X0)
    qc.h(ref[0])
    qc.cx(ref[0], X[BRANCH_BIT])
    fwd = shared_eea.eea_forward_shared_instruction(N, P, T_max=T_MAX, lazy_definition=True)
    qc.append(fwd, [*X, *A, *S], [*c_fwd])
    return TestCircuit(qc, ref, X, A, S, c_fwd, None)


def build_coherent_roundtrip_circuit() -> TestCircuit:
    layout = shared_eea.shared_eea_layout(N, T_max=T_MAX)
    ref = QuantumRegister(1, "ref")
    X = QuantumRegister(N, "X")
    A = QuantumRegister(N, "A")
    S = QuantumRegister(layout.s_qubits, "S")
    c_fwd = ClassicalRegister(N, "c_fwd")
    c_inv = ClassicalRegister(N, "c_inv")
    qc = QuantumCircuit(ref, X, A, S, c_fwd, c_inv, name="P419_COHERENT_FORWARD_INVERSE")

    # (|0>|153> + |1>|155>)/sqrt(2), with X1=X0 xor 2.
    _set_le_constant(qc, X, X0)
    qc.h(ref[0])
    qc.cx(ref[0], X[BRANCH_BIT])

    fwd = shared_eea.eea_forward_shared_instruction(N, P, T_max=T_MAX, lazy_definition=True)
    inv = shared_eea.eea_inverse_shared_instruction(N, P, T_max=T_MAX, lazy_definition=True)
    qc.append(fwd, [*X, *A, *S], [*c_fwd])
    qc.append(inv, [*X, *A, *S], [*c_inv])
    return TestCircuit(qc, ref, X, A, S, c_fwd, c_inv)


def _reg_value(basis: int, circuit, reg: QuantumRegister) -> int:
    out = 0
    for i, q in enumerate(reg):
        out |= ((basis >> _find_index(circuit, q)) & 1) << i
    return out


def _reg_is_zero(basis: int, circuit, reg: QuantumRegister) -> bool:
    return all(((basis >> _find_index(circuit, q)) & 1) == 0 for q in reg)


def _state_fidelity(a: Mapping[int, complex], b: Mapping[int, complex]) -> float:
    keys = set(a) | set(b)
    overlap = sum(complex(a.get(k, 0.0j)).conjugate() * complex(b.get(k, 0.0j)) for k in keys)
    na = sum(abs(v) ** 2 for v in a.values())
    nb = sum(abs(v) ** 2 for v in b.values())
    return float(abs(overlap) ** 2 / (na * nb))


def _canonical_state(state: Mapping[int, complex], atol=1e-10) -> Dict[int, complex]:
    state = {k: complex(v) for k, v in state.items() if abs(v) > atol}
    if not state:
        return {}
    pivot = min(state)
    phase = state[pivot] / abs(state[pivot])
    return {k: v / phase for k, v in state.items()}


def _prepared_roundtrip_expected(tc: TestCircuit) -> Dict[int, complex]:
    assert tc.ref is not None
    q0 = 0
    # Branch 0: ref=0, X=X0.
    for i, q in enumerate(tc.X):
        if (X0 >> i) & 1:
            q0 |= 1 << _find_index(tc.circuit, q)
    # Branch 1: ref=1, X=X1.
    q1 = 1 << _find_index(tc.circuit, tc.ref[0])
    for i, q in enumerate(tc.X):
        if (X1 >> i) & 1:
            q1 |= 1 << _find_index(tc.circuit, q)
    s = 1.0 / math.sqrt(2.0)
    return {q0: s + 0.0j, q1: s + 0.0j}


def run_basis_forward_shots(shots: int, seed: int, verbose: bool) -> Tuple[TrajectoryResult, List[List[int]]]:
    tc = build_forward_circuit(X0)
    records: List[List[int]] = []
    first: Optional[TrajectoryResult] = None
    expected = pow(X0, -1, P)

    for shot in range(shots):
        sim = SparseTrajectorySimulator(tc.circuit.num_qubits, tc.circuit.num_clbits, seed=seed + shot)
        result = sim.run(tc.circuit)
        records.append(result.measurement_record)
        if first is None:
            first = result
        if len(result.state) != 1:
            raise AssertionError(f"basis forward shot {shot}: support is {len(result.state)}, expected 1")
        basis = next(iter(result.state))
        got = _reg_value(basis, tc.circuit, tc.X)
        if got != expected:
            raise AssertionError(f"basis forward shot {shot}: X={got}, expected {expected}")
        if (X0 * got) % P != 1:
            raise AssertionError(f"basis forward shot {shot}: {X0}*{got} mod {P} != 1")
        if not _reg_is_zero(basis, tc.circuit, tc.A):
            raise AssertionError(f"basis forward shot {shot}: large workspace A is not zero")
        if _reg_is_zero(basis, tc.circuit, tc.S):
            raise AssertionError(
                "basis forward shot unexpectedly has S=0; the forward map should retain Gamma(x)"
            )
        if verbose:
            probs = result.measurement_probabilities
            print(
                f"  forward shot {shot:02d}: X={got}, measurements={len(probs)}, "
                f"min_branch_prob={min(probs) if probs else 1.0:.6f}"
            )
    assert first is not None
    return first, records


def run_coherent_forward(verbose: bool) -> TrajectoryResult:
    tc = build_coherent_forward_circuit()
    assert tc.ref is not None
    sampled = SparseTrajectorySimulator(
        tc.circuit.num_qubits, tc.circuit.num_clbits, seed=777
    ).run(tc.circuit)
    replay = SparseTrajectorySimulator(
        tc.circuit.num_qubits,
        tc.circuit.num_clbits,
        seed=999,
        forced_measurements=sampled.measurement_record,
    ).run(tc.circuit)
    if len(replay.state) != 2:
        raise AssertionError(
            f"coherent forward support={len(replay.state)}, expected 2; the reference may have decohered"
        )
    expected_by_ref = {0: pow(X0, -1, P), 1: pow(X1, -1, P)}
    seen = {}
    magnitudes = []
    for basis, amp in replay.state.items():
        ref_bit = (basis >> _find_index(tc.circuit, tc.ref[0])) & 1
        x_value = _reg_value(basis, tc.circuit, tc.X)
        seen[ref_bit] = x_value
        magnitudes.append(abs(amp))
        if not _reg_is_zero(basis, tc.circuit, tc.A):
            raise AssertionError("coherent forward: A is not zero")
    if seen != expected_by_ref:
        raise AssertionError(f"coherent forward map got {seen}, expected {expected_by_ref}")
    for mag in magnitudes:
        if abs(mag - 1.0 / math.sqrt(2.0)) > 1e-10:
            raise AssertionError(f"coherent forward branch magnitude {mag} is not 1/sqrt(2)")
    if verbose:
        print(
            "  coherent forward: "
            f"ref=0 -> {seen[0]}, ref=1 -> {seen[1]}, "
            f"support={len(replay.state)}, measurements={len(replay.measurement_record)}"
        )
    return replay


def run_coherent_roundtrip(records: Sequence[Sequence[int]], verbose: bool) -> TrajectoryResult:
    tc = build_coherent_roundtrip_circuit()
    expected = _prepared_roundtrip_expected(tc)
    last: Optional[TrajectoryResult] = None

    # Forward+inverse has twice as many dynamic measurements as one forward
    # trajectory.  Generate independent records by first allowing the simulator
    # to sample, then replay the exact record to make the test deterministic.
    for i in range(max(1, min(8, len(records)))):
        sampled = SparseTrajectorySimulator(
            tc.circuit.num_qubits, tc.circuit.num_clbits, seed=1000 + i
        ).run(tc.circuit)
        replay = SparseTrajectorySimulator(
            tc.circuit.num_qubits,
            tc.circuit.num_clbits,
            seed=9999,
            forced_measurements=sampled.measurement_record,
        ).run(tc.circuit)
        fidelity = _state_fidelity(replay.state, expected)
        if fidelity < 1.0 - 1e-10:
            got = _canonical_state(replay.state)
            exp = _canonical_state(expected)
            raise AssertionError(
                f"coherent round trip trajectory {i}: fidelity={fidelity:.12f}\n"
                f"got={got}\nexpected={exp}"
            )
        for basis in replay.state:
            if not _reg_is_zero(basis, tc.circuit, tc.A):
                raise AssertionError(f"coherent trajectory {i}: A not zero")
            if not _reg_is_zero(basis, tc.circuit, tc.S):
                raise AssertionError(f"coherent trajectory {i}: S not zero")
        if len(replay.state) != 2:
            raise AssertionError(
                f"coherent trajectory {i}: support={len(replay.state)}, expected 2; coherence may be lost"
            )
        if verbose:
            print(
                f"  coherent trajectory {i:02d}: fidelity={fidelity:.12f}, "
                f"support={len(replay.state)}, measurements={len(replay.measurement_record)}"
            )
        last = replay
    assert last is not None
    return last


def main() -> None:
    global X0, X1, BRANCH_BIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=419153)
    parser.add_argument("--x0", type=int, default=X0)
    parser.add_argument("--x1", type=int, default=X1)
    parser.add_argument("--require-real-qiskit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.require_real_qiskit and not USING_REAL_QISKIT:
        raise RuntimeError("--require-real-qiskit was requested, but Qiskit is unavailable")
    if not (1 <= args.x0 < P and 1 <= args.x1 < P):
        raise ValueError(f"x0 and x1 must lie in [1,{P-1}]")
    diff = int(args.x0) ^ int(args.x1)
    if diff == 0 or (diff & (diff - 1)):
        raise ValueError("the coherent branches must differ in exactly one input bit")
    X0, X1 = int(args.x0), int(args.x1)
    BRANCH_BIT = diff.bit_length() - 1

    r0, r1 = P, min(X0, P - X0)
    weighted = 0
    while r1:
        q, r2 = divmod(r0, r1)
        weighted += int(q).bit_length()
        r0, r1 = r1, r2
    exact_steps = 4 * weighted
    if X0 == 153 and exact_steps != COMPUTATIONAL_STEPS:
        raise AssertionError(
            f"p=419,x=153 must require {COMPUTATIONAL_STEPS} computational microsteps, got {exact_steps}"
        )

    expected0 = pow(X0, -1, P)
    expected1 = pow(X1, -1, P)
    print(f"p=419, x0={X0}, x1={X1} gate-level quantum semantic regression")
    print(f"Qiskit circuit objects: {'real Qiskit' if USING_REAL_QISKIT else 'mini data-model fallback'}")
    print("Execution engine: sparse quantum-state dynamic-circuit trajectories")
    print("Classical EEA is not used to produce the circuit output.\n")

    forward, records = run_basis_forward_shots(args.shots, args.seed, args.verbose)
    coherent_forward = run_coherent_forward(args.verbose)
    coherent = run_coherent_roundtrip(records, args.verbose)

    print(
        f"[PASS] Forward semantic map: X={X0} -> X={expected0} and "
        f"{X0}*{expected0} mod {P} = 1"
    )
    print(
        "[PASS] Coherent forward map preserves both reference branches: "
        f"{X0} -> {expected0} and {X1} -> {expected1}"
    )
    print("[PASS] Forward large workspace A is clean; S retains the forward EEA state")
    print("[PASS] Coherent reference-entangled forward+inverse round trip has fidelity 1")
    print("[PASS] After round trip, X is restored and every A/S quantum ancilla is zero")
    print(f"[INFO] Uniform Algorithm-3 schedule: {T_MAX} microsteps")
    print(f"[INFO] Computational microsteps for p=419,x0={X0}: {exact_steps}")
    print(f"[INFO] Forward dynamic measurements per trajectory: {forward.stats.measurements}")
    print(f"[INFO] Round-trip dynamic measurements per trajectory: {coherent.stats.measurements}")
    print(f"[INFO] Round-trip maximum sparse support: {coherent.stats.max_support}")
    print(f"[INFO] Round-trip recursively executed composite calls: {coherent.stats.composite_calls}")
    print("[INFO] Round-trip primitive gate counts:")
    for name, count in sorted(coherent.stats.primitive_counts.items()):
        print(f"       {name:8s} {count}")


if __name__ == "__main__":
    main()
