#!/usr/bin/env python3
"""ECDSAfail-inspired trusted test utilities for emitted Qiskit circuits.

This module deliberately separates the *circuit producer* from the *test
interpreter*.  It recursively lowers an already-built Qiskit circuit into a
small, fail-closed operation stream and executes many computational-basis
shots in parallel while tracking:

* classical output bits,
* measurement-dependent phase kickback,
* every quantum ancilla,
* classical feed-forward conditions, and
* a deterministic circuit-derived test corpus.

The design mirrors the public ecdsa.fail harness philosophy, but it is not a
binary-compatible replacement for that Rust/KMX harness.  In particular, the
current Figure-14 builder has a compile-time classical offset point instead of
four Google-ABI input registers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

USING_REAL_QISKIT = True
try:
    import qiskit  # type: ignore
except Exception:
    USING_REAL_QISKIT = False
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()
    import qiskit  # type: ignore


ALIASES = {"cnot": "cx", "tof": "ccx", "toffoli": "ccx"}
PRIMITIVES = {
    "x", "z", "cx", "cz", "ccx", "ccz", "mcx", "swap",
    "h", "measure", "reset", "barrier", "id", "delay",
}
IGNORED = {"barrier", "id", "delay"}


def items(circuit):
    for item in circuit.data:
        if hasattr(item, "operation"):
            yield item.operation, tuple(item.qubits), tuple(item.clbits)
        else:
            op, qargs, cargs = item
            yield op, tuple(qargs), tuple(cargs)


def bit_index(circuit, bit) -> int:
    return int(circuit.find_bit(bit).index)


def op_name(op) -> str:
    base = getattr(op, "base_name", None)
    name = str(base if base is not None else op.name).lower()
    while name.endswith("_dg"):
        name = name[:-3]
    return ALIASES.get(name, name)


@dataclass(frozen=True)
class FlatOp:
    kind: str
    qargs: Tuple[int, ...] = ()
    cargs: Tuple[int, ...] = ()
    expected: int = 1

    def canonical_line(self) -> str:
        qs = ",".join(str(x) for x in self.qargs)
        cs = ",".join(str(x) for x in self.cargs)
        return f"{self.kind}|q={qs}|c={cs}|e={self.expected}\n"


def _condition_spec(condition, owner, c_map: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
    if condition is None:
        return (), 1
    lhs, expected = condition
    # Single Clbit.
    try:
        is_register = hasattr(lhs, "__iter__") and not hasattr(lhs, "register")
    except Exception:
        is_register = False
    if not is_register:
        local = bit_index(owner, lhs)
        return (int(c_map[local]),), int(expected)
    cids = tuple(int(c_map[bit_index(owner, b)]) for b in lhs)
    return cids, int(expected)


def flatten_circuit_fail_closed(circuit) -> List[FlatOp]:
    """Recursively lower ``circuit`` to approved leaves and condition markers.

    Unknown/opaque leaves raise immediately.  IfElseOps are represented by
    explicit PUSH_IF/POP_IF markers, so the resulting stream itself contains
    the feed-forward semantics used by the independent interpreter.
    """
    out: List[FlatOp] = []

    def emit_circuit(circ, q_map: Sequence[int], c_map: Sequence[int]) -> None:
        for op, qargs, cargs in items(circ):
            qids = tuple(int(q_map[bit_index(circ, q)]) for q in qargs)
            cids = tuple(int(c_map[bit_index(circ, c)]) for c in cargs)
            name = op_name(op)

            if name == "if_else" and hasattr(op, "blocks"):
                cond_cids, expected = _condition_spec(getattr(op, "condition", None), circ, c_map)
                if not cond_cids:
                    raise RuntimeError("IfElseOp without a resolvable classical condition")
                blocks = list(op.blocks)
                if blocks:
                    out.append(FlatOp("push_if", cargs=cond_cids, expected=expected))
                    emit_circuit(blocks[0], qids, cids)
                    out.append(FlatOp("pop_if"))
                if len(blocks) > 1:
                    width = len(cond_cids)
                    complement = ((1 << width) - 1) ^ expected if width > 1 else 1 - expected
                    out.append(FlatOp("push_if", cargs=cond_cids, expected=complement))
                    emit_circuit(blocks[1], qids, cids)
                    out.append(FlatOp("pop_if"))
                continue

            condition = getattr(op, "condition", None)
            if condition is not None:
                cond_cids, expected = _condition_spec(condition, circ, c_map)
                out.append(FlatOp("push_if", cargs=cond_cids, expected=expected))

            definition = getattr(op, "definition", None)
            if definition is not None and name not in PRIMITIVES:
                if len(qids) != int(op.num_qubits) or len(cids) != int(op.num_clbits):
                    raise RuntimeError(f"bad composite mapping for {op.name}")
                emit_circuit(definition, qids, cids)
            elif name in PRIMITIVES:
                out.append(FlatOp(name, qargs=qids, cargs=cids))
            else:
                raise RuntimeError(
                    f"opaque/unsupported Qiskit leaf {op.name!r}; definition={definition!r}"
                )

            if condition is not None:
                out.append(FlatOp("pop_if"))

    emit_circuit(circuit, list(range(circuit.num_qubits)), list(range(circuit.num_clbits)))
    return out


def canonical_flat_opstream_bytes(num_qubits: int, num_clbits: int, ops: Sequence[FlatOp]) -> bytes:
    ops = list(ops)
    header = (
        "qiskit-ecdsafail-style-opstream-v1\n"
        f"num_qubits={int(num_qubits)}\n"
        f"num_clbits={int(num_clbits)}\n"
        f"num_ops={len(ops)}\n"
    ).encode()
    return header + "".join(op.canonical_line() for op in ops).encode()


def canonical_opstream_bytes(circuit, ops: Optional[Sequence[FlatOp]] = None) -> bytes:
    ops = list(ops) if ops is not None else flatten_circuit_fail_closed(circuit)
    return canonical_flat_opstream_bytes(circuit.num_qubits, circuit.num_clbits, ops)


def circuit_fingerprint(circuit, ops: Optional[Sequence[FlatOp]] = None) -> str:
    return hashlib.sha256(canonical_opstream_bytes(circuit, ops)).hexdigest()


def flat_opstream_fingerprint(num_qubits: int, num_clbits: int, ops: Sequence[FlatOp]) -> str:
    return hashlib.sha256(canonical_flat_opstream_bytes(num_qubits, num_clbits, ops)).hexdigest()


class DeterministicMaskStream:
    """Counter-mode SHAKE256 masks derived from the actual lowered op stream."""

    def __init__(self, seed_material: bytes, ncases: int):
        self.seed = hashlib.sha256(seed_material).digest()
        self.ncases = int(ncases)
        self.nbytes = (self.ncases + 7) // 8
        self.counter = 0
        self.allmask = (1 << self.ncases) - 1 if self.ncases else 0

    def next_mask(self, label: bytes = b"hmr") -> int:
        h = hashlib.shake_256()
        h.update(b"qiskit-ecdsafail-style-rng-v1")
        h.update(self.seed)
        h.update(label)
        h.update(self.counter.to_bytes(8, "little"))
        self.counter += 1
        return int.from_bytes(h.digest(self.nbytes), "little") & self.allmask


def derive_case_integers(fingerprint_hex: str, modulus: int, count: int, *, domain: str) -> List[int]:
    """Derive values in ``[1, modulus-1]`` from an artifact fingerprint."""
    if modulus <= 2:
        raise ValueError("modulus must exceed 2")
    out: List[int] = []
    for i in range(int(count)):
        h = hashlib.shake_256()
        h.update(b"qiskit-ecdsafail-style-cases-v1")
        h.update(bytes.fromhex(fingerprint_hex))
        h.update(domain.encode())
        h.update(i.to_bytes(8, "little"))
        out.append(1 + int.from_bytes(h.digest(64), "little") % (modulus - 1))
    return out


def encode_values(values: Sequence[int], width: int) -> List[int]:
    masks = [0] * int(width)
    for case_i, value in enumerate(values):
        for bit in range(int(width)):
            if (int(value) >> bit) & 1:
                masks[bit] |= 1 << case_i
    return masks


@dataclass
class BatchStats:
    primitive_emitted: Dict[str, int]
    measurement_ops: int = 0
    executed_condition_nodes: int = 0


class PhaseBatchSimulator:
    """Bit-parallel basis simulator with ecdsa.fail-style phase tracking.

    It is exact for the gate family emitted by this repository.  An H followed
    by measurement is interpreted as X-basis demolition: the outcome is a
    deterministic artifact-derived random mask and the phase is updated by
    ``old_value & outcome``, matching the HMR semantics used by the public Rust
    harness.  General unmeasured superpositions are rejected.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        ncases: int,
        *,
        seed_material: bytes,
        initial_qubit_masks: Optional[Mapping[int, int]] = None,
        initial_clbit_masks: Optional[Mapping[int, int]] = None,
    ):
        self.ncases = int(ncases)
        self.allmask = (1 << self.ncases) - 1 if self.ncases else 0
        self.q = [0] * int(num_qubits)
        self.c = [0] * int(num_clbits)
        for k, v in (initial_qubit_masks or {}).items():
            self.q[int(k)] = int(v) & self.allmask
        for k, v in (initial_clbit_masks or {}).items():
            self.c[int(k)] = int(v) & self.allmask
        self.phase = 0
        self.pending_h = [0] * int(num_qubits)
        self.rng = DeterministicMaskStream(seed_material, self.ncases)
        self.active = self.allmask
        self.active_stack: List[int] = []
        self.stats = BatchStats(primitive_emitted={})

    def _condition_mask(self, cids: Sequence[int], expected: int) -> int:
        if not cids:
            return self.allmask
        truth = self.allmask
        for i, cid in enumerate(cids):
            bit = self.c[int(cid)]
            truth &= bit if ((int(expected) >> i) & 1) else (self.allmask ^ bit)
        return truth

    def run_ops(self, ops: Sequence[FlatOp]) -> None:
        for op_i, op in enumerate(ops):
            kind = op.kind
            if kind == "push_if":
                self.active_stack.append(self.active)
                self.active &= self._condition_mask(op.cargs, op.expected)
                self.stats.executed_condition_nodes += 1
                continue
            if kind == "pop_if":
                if not self.active_stack:
                    raise RuntimeError(f"condition stack underflow at op {op_i}")
                self.active = self.active_stack.pop()
                continue

            self.stats.primitive_emitted[kind] = self.stats.primitive_emitted.get(kind, 0) + 1
            active = self.active
            if active == 0:
                continue
            qs = op.qargs
            cs = op.cargs

            if kind == "x":
                self.q[qs[0]] ^= active
            elif kind == "z":
                self.phase ^= active & self.q[qs[0]]
            elif kind == "cx":
                self.q[qs[1]] ^= active & self.q[qs[0]]
            elif kind == "cz":
                self.phase ^= active & self.q[qs[0]] & self.q[qs[1]]
            elif kind == "ccx":
                self.q[qs[2]] ^= active & self.q[qs[0]] & self.q[qs[1]]
            elif kind == "ccz":
                self.phase ^= active & self.q[qs[0]] & self.q[qs[1]] & self.q[qs[2]]
            elif kind == "mcx":
                mask = active
                for q in qs[:-1]:
                    mask &= self.q[q]
                self.q[qs[-1]] ^= mask
            elif kind == "swap":
                d = active & (self.q[qs[0]] ^ self.q[qs[1]])
                self.q[qs[0]] ^= d
                self.q[qs[1]] ^= d
            elif kind == "h":
                overlap = self.pending_h[qs[0]] & active
                if overlap:
                    raise RuntimeError(
                        f"unsupported second H before measurement on q{qs[0]} at op {op_i}"
                    )
                self.pending_h[qs[0]] |= active
            elif kind == "measure":
                if len(cs) != 1:
                    raise RuntimeError("measure must have exactly one clbit")
                q, c = qs[0], cs[0]
                pending = self.pending_h[q] & active
                ordinary = active & (self.allmask ^ pending)
                random_outcome = self.rng.next_mask(b"hmr") & pending
                deterministic_outcome = self.q[q] & ordinary
                outcome = random_outcome | deterministic_outcome
                # H-basis demolition phase: (-1)^(old_q * measurement_outcome).
                self.phase ^= self.q[q] & random_outcome
                self.c[c] = (self.c[c] & (self.allmask ^ active)) | outcome
                self.q[q] &= self.allmask ^ pending
                self.pending_h[q] &= self.allmask ^ active
                self.stats.measurement_ops += 1
            elif kind == "reset":
                q = qs[0]
                if self.pending_h[q] & active:
                    raise RuntimeError(f"reset of q{q} while H is still pending")
                self.q[q] &= self.allmask ^ active
            elif kind in IGNORED:
                pass
            else:
                raise RuntimeError(f"unsupported flat primitive {kind!r}")

        if self.active_stack:
            raise RuntimeError("unterminated condition stack")
        pending = 0
        for mask in self.pending_h:
            pending |= mask
        if pending:
            raise RuntimeError("circuit ended with an H that was not measured")

    def run_circuit(self, circuit, ops: Optional[Sequence[FlatOp]] = None) -> List[FlatOp]:
        flat = list(ops) if ops is not None else flatten_circuit_fail_closed(circuit)
        self.run_ops(flat)
        return flat

    def register_values(self, circuit, register) -> List[int]:
        out = [0] * self.ncases
        for bit_i, q in enumerate(register):
            mask = self.q[bit_index(circuit, q)]
            while mask:
                lsb = mask & -mask
                case_i = lsb.bit_length() - 1
                out[case_i] |= 1 << bit_i
                mask ^= lsb
        return out

    def register_zero_mask(self, circuit, register) -> int:
        occupied = 0
        for q in register:
            occupied |= self.q[bit_index(circuit, q)]
        return self.allmask ^ occupied

    def qubits_zero_mask(self, qids: Iterable[int]) -> int:
        occupied = 0
        for q in qids:
            occupied |= self.q[int(q)]
        return self.allmask ^ occupied


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
