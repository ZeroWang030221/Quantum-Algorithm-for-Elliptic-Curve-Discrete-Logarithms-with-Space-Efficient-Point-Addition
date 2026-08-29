from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

import argparse
import gc
import json
import math
import re
import shutil
import subprocess
import time
import sys

try:
    from qiskit import transpile
    from qiskit.circuit import ClassicalRegister, Gate, Instruction, QuantumCircuit, QuantumRegister, Qubit
    from qiskit.qasm2 import dumps as qasm2_dumps
    from qiskit.circuit.library import ZGate, CZGate
except ImportError:  # Formula/test mode can still build X/CX/CCX circuits without Qiskit.
    transpile = None
    qasm2_dumps = None

    class _FallbackBit:
        def __init__(self, register: "QuantumRegister", index: int):
            self.register = register
            self.index = index

        def __repr__(self) -> str:
            return f"{self.register.name}[{self.index}]"

    class ClassicalRegister:  # type: ignore[no-redef]
        def __init__(self, size: int, name: str):
            self.size = int(size)
            self.name = name
            self._bits = [None for _ in range(self.size)]

        def __len__(self) -> int:
            return self.size

        def __iter__(self):
            return iter(self._bits)

        def __getitem__(self, item):
            return self._bits[item]

    class QuantumRegister:  # type: ignore[no-redef]
        def __init__(self, size: int, name: str):
            self.size = int(size)
            self.name = name
            self._bits = [_FallbackBit(self, i) for i in range(self.size)]

        def __len__(self) -> int:
            return self.size

        def __iter__(self):
            return iter(self._bits)

        def __getitem__(self, item):
            return self._bits[item]

    Qubit = _FallbackBit  # type: ignore

    class _FallbackInstruction:
        def __init__(self, name: str):
            self.name = name
            self.definition = None
            self.num_qubits = {"x": 1, "cx": 2, "ccx": 3}.get(name, 0)
            self.num_clbits = 0
            self.params = []

    Instruction = _FallbackInstruction  # type: ignore
    Gate = _FallbackInstruction  # type: ignore

    class _FallbackItem:
        def __init__(self, operation, qubits):
            self.operation = operation
            self.qubits = tuple(qubits)
            self.clbits = tuple()

        def __iter__(self):
            yield self.operation
            yield self.qubits
            yield self.clbits

    class _FallbackFindBit:
        def __init__(self, index: int):
            self.index = index

    class QuantumCircuit:  # type: ignore[no-redef]
        def __init__(self, *regs, name: str = "qc"):
            self.qregs = list(regs)
            self.name = name
            self.qubits = [q for reg in self.qregs for q in reg]
            self.num_qubits = len(self.qubits)
            self.data = []
            self._index = {q: i for i, q in enumerate(self.qubits)}

        def find_bit(self, qubit):
            return _FallbackFindBit(self._index[qubit])

        def x(self, q):
            self.data.append(_FallbackItem(_FallbackInstruction("x"), [q]))

        def cx(self, c, t):
            self.data.append(_FallbackItem(_FallbackInstruction("cx"), [c, t]))

        def ccx(self, a, b, t):
            self.data.append(_FallbackItem(_FallbackInstruction("ccx"), [a, b, t]))

        def append(self, gate, qargs):
            if getattr(gate, "definition", None) is None and getattr(gate, "name", "").lower() in {"x", "cx", "ccx"}:
                self.data.append(_FallbackItem(gate, list(qargs)))
            elif getattr(gate, "definition", None) is not None:
                self.data.append(_FallbackItem(gate, list(qargs)))
            else:
                raise RuntimeError("Fallback QuantumCircuit only supports x/cx/ccx or defined composite gates")

        def to_gate(self, label: Optional[str] = None):
            g = _FallbackInstruction(label or self.name)
            g.definition = self
            g.num_qubits = self.num_qubits
            return g


# Revised universal four-phase EEA growth constants (Appendix A.1).
# One unit of weighted quotient cost grows the continuant by at least lambda.
EEA_BASE = 2.0 + math.sqrt(3.0)
EEA_LAMBDA = EEA_BASE ** (1.0 / 3.0)
C_EEA = 1.0 / math.log2(EEA_LAMBDA)
# Refined finite-prefix correction used by the revised active-window proof.
DELTA_EEA = -math.log(((4.0 * math.sqrt(3.0) - 3.0) / 13.0) * (EEA_LAMBDA ** 2), EEA_LAMBDA)

def paper_len_width(n: int) -> int:
    """Width of l_t, l_q and l_r' with the unary/sign-free endpoint unitary.

    For cryptographic sizes this is floor(log2(n))+1.  The max with
    ceil(log2(n+4)) keeps the same exact endpoint range for the very small
    n used by the regression tests, where floor(log2(n))+1 may not encode
    positions up to n+3.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return max(math.floor(math.log2(n)) + 1, math.ceil(math.log2(n + 4)))


def paper_shift_width(n: int) -> int:
    """Base paper width of the circular shift pointer l_s."""
    return paper_len_width(n)


def terminal_safe_shift_width(n: int, T_max: Optional[int] = None) -> int:
    """Width of the low shift word when one borrowed epoch bit is available.

    ``l_s`` retains the paper width whenever the fixed-horizon padding fits in
    the two-word counter formed by ``l_s`` and one borrowed clean auxiliary
    qubit.  The auxiliary is used only as a terminal-padding epoch bit and is
    reversibly folded back into ``l_s`` before the EEA interface is exposed.
    A low-word bit is added only for finite-size edge cases where even this
    two-word representation would wrap.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if T_max is None:
        T_max = 4 * math.ceil(C_EEA * n)
    max_padding = max(0, int(T_max) - 4 * int(n))
    width = paper_shift_width(n)
    while max_padding >= (1 << (width + 1)):
        width += 1
    return width


def _n_config_entry(n: int) -> dict:
    T_max = 4 * math.ceil(C_EEA * n)
    return {
        "len_width": paper_len_width(n),
        "shift_width": terminal_safe_shift_width(n, T_max),
        "T_max": T_max,
    }


N_CONFIG = {n: _n_config_entry(n) for n in (64, 128, 160, 192, 224, 256, 384, 512)}

PRIMITIVE_OPS = {"ccx", "cx", "x", "h", "z", "cz", "measure", "reset"}
_INST_COUNT_CACHE: dict[tuple, Counter] = {}


def _require_qiskit() -> None:
    if QuantumCircuit is Any or QuantumRegister is Any or transpile is None:
        raise RuntimeError("Qiskit is required for circuit construction/export modes. Install with: pip install qiskit")



# Measurement-based uncomputation support
# ---------------------------------------
# In the paper/resource model, unary-iteration ANDs are uncomputed by measuring
# the temporary AND bit in the Hadamard basis and applying the corresponding
# feed-forward phase correction.  This removes the reverse Toffoli from the
# Toffoli count, but produces a dynamic circuit containing H/measure/reset and
# classically controlled phase corrections.  The reversible X/CX/CCX path is
# still available for the strict Toffoli-network tests.
MEASUREMENT_UNCOMPUTE = False


def set_measurement_uncompute(enabled: bool) -> None:
    """Select coherent or measurement-assisted unary-AND cleanup.

    Circuit factories in several modules are memoized and depend on this mode.
    Invalidate every loaded LRU cache when the mode changes; otherwise a dynamic
    MBU definition can be reused in the literal unitary inverse (or vice versa).
    """
    global MEASUREMENT_UNCOMPUTE
    enabled = bool(enabled)
    changed = enabled != bool(MEASUREMENT_UNCOMPUTE)
    MEASUREMENT_UNCOMPUTE = enabled
    _INST_COUNT_CACHE.clear()
    if not changed:
        return
    import sys
    # Clear this module first without calling another module's wrapper, which
    # would recurse back here.
    clear = globals().get("clear_gate_construction_caches")
    if callable(clear):
        clear()
    for module_name in (
        "eea_circuit_s835_fastdual",
        "eea_circuit_s835_lowaux",
        "under1000_eea_shared_s835_fastdual_wrapped",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for obj in tuple(vars(module).values()):
            cache_clear = getattr(obj, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


def _block_circuit(*regs, name: str) -> QuantumCircuit:
    # Classical bits are added lazily only to blocks that actually perform a
    # measurement-based AND uncompute.  This avoids making ordinary reversible
    # helper gates expect unnecessary clbits.
    return QuantumCircuit(*regs, name=name)


def _finalize_block(qc: QuantumCircuit):
    """Return a reusable helper while preserving dynamic control flow exactly.

    Pure reversible helpers are packed as :class:`~qiskit.circuit.Gate` objects.
    Dynamic helpers are kept as ``QuantumCircuit`` objects and composed directly
    by :func:`_append_with_optional_clbits`.  This avoids rebuilding a dynamic
    definition through ``to_instruction()`` and therefore preserves the exact
    measurement-bit identities and ``IfElseOp`` feed-forward structure in
    Qiskit 2.x.
    """
    if getattr(qc, "num_clbits", 0):
        return qc
    return qc.to_gate()


def _active_measure_bit(qc: QuantumCircuit):
    if not getattr(qc, "clbits", None):
        qc.add_register(ClassicalRegister(1, "m"))
    return qc.clbits[0]


def _conditional_z(qc: QuantumCircuit, mbit, q: Qubit) -> None:
    """Apply the feed-forward Z correction iff the measured bit is one.

    Qiskit 2.x removed the legacy ``Instruction.condition`` execution model in
    favour of explicit control flow.  Using ``if_test`` ensures that the
    condition is part of the circuit IR and survives composition, conversion to
    an Instruction, lowering, and serialization.  The bundled mini circuit
    model implements the same context-manager interface for offline tests.
    """
    with qc.if_test((mbit, True)):
        qc.z(q)


def _conditional_cz(qc: QuantumCircuit, mbit, a: Qubit, b: Qubit) -> None:
    """Apply the measurement-dependent CZ correction using explicit control flow."""
    with qc.if_test((mbit, True)):
        qc.cz(a, b)


def _mb_uncompute_and(qc: QuantumCircuit, a: Qubit, b: Qubit, target: Qubit) -> None:
    """Measurement-based uncompute for target = a AND b.

    Starting with target coherently equal to a*b, this maps target back to |0>
    using H, measurement, reset, and a classically controlled CZ(a,b) phase
    correction.  It contributes zero CCX gates in the dynamic circuit.
    """
    m = _active_measure_bit(qc)
    qc.h(target)
    qc.measure(target, m)
    qc.reset(target)
    _conditional_cz(qc, m, a, b)


def _append_with_optional_clbits(parent: QuantumCircuit, block, qargs: Sequence[Qubit]) -> None:
    if isinstance(block, QuantumCircuit):
        clbits = []
        if len(block.clbits):
            if not getattr(parent, "clbits", None):
                parent.add_register(ClassicalRegister(1, "m"))
            clbits = [parent.clbits[0]] * len(block.clbits)
        parent.compose(block, qubits=list(qargs), clbits=clbits, inplace=True)
    else:
        if getattr(block, "num_clbits", 0):
            if not getattr(parent, "clbits", None):
                parent.add_register(ClassicalRegister(1, "m"))
            parent.append(block, list(qargs), [parent.clbits[0]] * block.num_clbits)
        else:
            parent.append(block, list(qargs))

# Basic reversible arithmetic and controls
def _toggle_x_for_constant(qc: QuantumCircuit, reg: Sequence[Qubit], value: int) -> None:
    n = len(reg)
    mask = value % (1 << n)
    for i in range(n):
        if (mask >> i) & 1:
            qc.x(reg[i])


def xor_const_into_reg(
    qc: QuantumCircuit,
    reg: Sequence[Qubit],
    value: int,
    ctrl: Optional[Qubit] = None,
) -> None:
    """XOR a classical constant into a little-endian register."""
    n = len(reg)
    mask = value % (1 << n)
    for i in range(n):
        if (mask >> i) & 1:
            if ctrl is None:
                qc.x(reg[i])
            else:
                qc.cx(ctrl, reg[i])


def xor_const_into_reg_controls(
    qc: QuantumCircuit,
    reg: Sequence[Qubit],
    value: int,
    ctrls: Sequence[Qubit] = (),
    scratch: Sequence[Qubit] = (),
) -> None:
    """XOR a classical constant into a little-endian register under controls."""
    ctrls = list(ctrls)
    mask = value % (1 << len(reg))
    for i, q in enumerate(reg):
        if ((mask >> i) & 1) == 0:
            continue
        if len(ctrls) == 0:
            qc.x(q)
        elif len(ctrls) == 1:
            qc.cx(ctrls[0], q)
        elif len(ctrls) == 2:
            qc.ccx(ctrls[0], ctrls[1], q)
        else:
            mcx_vchain(qc, ctrls, q, scratch)


def _maj(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit) -> None:
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def _uma(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit) -> None:
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)


def _maj_inv(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit) -> None:
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(a, b)


def _uma_inv(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit) -> None:
    qc.cx(c, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def cuccaro_add_mod_2n_gate(n: int, name: str = "ADD_Z") -> Gate:
    """Cuccaro in-place adder: |a>|b>|c=0>|z=0> -> |a>|a+b>|0>|carry>."""
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c0 = QuantumRegister(1, "c")
    z = QuantumRegister(1, "z")
    qc = QuantumCircuit(a, b, c0, z, name=name)

    c = c0[0]
    _maj(qc, a[0], b[0], c)
    for i in range(1, n):
        _maj(qc, a[i], b[i], a[i - 1])
    qc.cx(a[n - 1], z[0])
    for i in reversed(range(1, n)):
        _uma(qc, a[i], b[i], a[i - 1])
    _uma(qc, a[0], b[0], c)
    return qc.to_gate()


def cuccaro_add_mod_2n_no_z_gate(n: int, name: str = "ADD_NO_Z") -> Gate:
    """Scratch-clean modulo-2^n Cuccaro adder: |a>|b>|0> -> |a>|a+b mod 2^n>|0>.

    This is the same MAJ/UMA ripple-carry adder, but the overflow line is not
    materialized.  It is the correct primitive for affine transformations on
    endpoint/length registers that must restore every scratch bit before the
    enclosing block exits.
    """
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c0 = QuantumRegister(1, "c")
    qc = QuantumCircuit(a, b, c0, name=name)

    c = c0[0]
    _maj(qc, a[0], b[0], c)
    for i in range(1, n):
        _maj(qc, a[i], b[i], a[i - 1])
    for i in reversed(range(1, n)):
        _uma(qc, a[i], b[i], a[i - 1])
    _uma(qc, a[0], b[0], c)
    return qc.to_gate()


def cuccaro_sub_mod_2n_no_z_gate(n: int, name: str = "SUB_NO_Z") -> Gate:
    return cuccaro_add_mod_2n_no_z_gate(n, name=name).inverse()


def cuccaro_sub_mod_2n_gate(n: int, name: str = "SUB_Z") -> Gate:
    return cuccaro_add_mod_2n_gate(n, name=name).inverse()


def add_const_mod_2n(qc: QuantumCircuit, reg: Sequence[Qubit], value: int, scratch: Sequence[Qubit]) -> None:
    """Scratch-clean in-place map reg <- reg + value (mod 2^len(reg))."""
    n = len(reg)
    if len(scratch) < n + 1:
        raise ValueError(f"add_const_mod_2n needs >= {n + 1} scratch qubits, got {len(scratch)}")
    const = list(scratch[:n])
    carry = scratch[n]
    _toggle_x_for_constant(qc, const, value)
    qc.append(cuccaro_add_mod_2n_no_z_gate(n), const + list(reg) + [carry])
    _toggle_x_for_constant(qc, const, value)


def sub_const_mod_2n(qc: QuantumCircuit, reg: Sequence[Qubit], value: int, scratch: Sequence[Qubit]) -> None:
    """Scratch-clean in-place map reg <- reg - value (mod 2^len(reg))."""
    n = len(reg)
    if len(scratch) < n + 1:
        raise ValueError(f"sub_const_mod_2n needs >= {n + 1} scratch qubits, got {len(scratch)}")
    const = list(scratch[:n])
    carry = scratch[n]
    _toggle_x_for_constant(qc, const, value)
    qc.append(cuccaro_sub_mod_2n_no_z_gate(n), const + list(reg) + [carry])
    _toggle_x_for_constant(qc, const, value)


def inc_mod2n_uncontrolled(qc: QuantumCircuit, reg: Sequence[Qubit], anc: Sequence[Qubit]) -> None:
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.x(reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"inc_mod2n_uncontrolled needs {n - 1} ancillas, got {len(anc)}")
    c = list(anc)
    qc.cx(reg[0], c[0])
    for i in range(1, n - 1):
        qc.ccx(reg[i], c[i - 1], c[i])
    qc.cx(c[n - 2], reg[n - 1])
    for i in range(n - 2, 0, -1):
        qc.ccx(reg[i], c[i - 1], c[i])
        qc.cx(c[i - 1], reg[i])
    qc.cx(reg[0], c[0])
    qc.x(reg[0])


def dec_mod2n_uncontrolled(qc: QuantumCircuit, reg: Sequence[Qubit], anc: Sequence[Qubit]) -> None:
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.x(reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"dec_mod2n_uncontrolled needs {n - 1} ancillas, got {len(anc)}")
    b = list(anc)
    qc.x(reg[0])
    qc.cx(reg[0], b[0])
    qc.x(reg[0])
    for i in range(1, n - 1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])
    qc.cx(b[n - 2], reg[n - 1])
    for i in range(n - 2, 0, -1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])
        qc.cx(b[i - 1], reg[i])
    qc.x(reg[0])
    qc.cx(reg[0], b[0])
    qc.x(reg[0])
    qc.x(reg[0])


def inc_mod2n_1ctrl(qc: QuantumCircuit, ctrl: Qubit, reg: Sequence[Qubit], anc: Sequence[Qubit]) -> None:
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.cx(ctrl, reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"inc_mod2n_1ctrl needs >= {n - 1} ancillas, got {len(anc)}")
    c = list(anc)
    qc.ccx(ctrl, reg[0], c[0])
    qc.cx(ctrl, reg[0])
    for i in range(1, n - 1):
        qc.ccx(reg[i], c[i - 1], c[i])
    qc.cx(c[n - 2], reg[n - 1])
    for i in range(n - 2, 0, -1):
        qc.ccx(reg[i], c[i - 1], c[i])
        qc.cx(c[i - 1], reg[i])
    qc.cx(ctrl, reg[0])
    qc.ccx(ctrl, reg[0], c[0])
    qc.cx(ctrl, reg[0])


def dec_mod2n_1ctrl(qc: QuantumCircuit, ctrl: Qubit, reg: Sequence[Qubit], anc: Sequence[Qubit]) -> None:
    n = len(reg)
    if n == 0:
        return
    if n == 1:
        qc.cx(ctrl, reg[0])
        return
    if len(anc) < n - 1:
        raise ValueError(f"dec_mod2n_1ctrl needs >= {n - 1} ancillas, got {len(anc)}")
    b = list(anc)
    qc.x(reg[0])
    qc.ccx(ctrl, reg[0], b[0])
    qc.x(reg[0])
    qc.cx(ctrl, reg[0])
    for i in range(1, n - 1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])
    qc.cx(b[n - 2], reg[n - 1])
    for i in range(n - 2, 0, -1):
        qc.x(reg[i])
        qc.ccx(reg[i], b[i - 1], b[i])
        qc.x(reg[i])
        qc.cx(b[i - 1], reg[i])
    qc.cx(ctrl, reg[0])
    qc.x(reg[0])
    qc.ccx(ctrl, reg[0], b[0])
    qc.x(reg[0])
    qc.cx(ctrl, reg[0])


def mcx_vchain(qc: QuantumCircuit, ctrls: Sequence[Qubit], target: Qubit, ancillas: Sequence[Qubit]) -> None:
    """Multi-controlled X using a clean v-chain.  All ancillas are restored."""
    ctrls = list(ctrls)
    m = len(ctrls)
    if m == 0:
        qc.x(target)
        return
    if m == 1:
        qc.cx(ctrls[0], target)
        return
    if m == 2:
        qc.ccx(ctrls[0], ctrls[1], target)
        return
    need = m - 2
    if len(ancillas) < need:
        raise ValueError(f"mcx_vchain needs >= {need} ancillas, got {len(ancillas)}")
    ancillas = list(ancillas)
    qc.ccx(ctrls[0], ctrls[1], ancillas[0])
    for i in range(2, m - 1):
        qc.ccx(ctrls[i], ancillas[i - 2], ancillas[i - 1])
    qc.ccx(ctrls[m - 1], ancillas[m - 3], target)
    for i in range(m - 2, 1, -1):
        if MEASUREMENT_UNCOMPUTE:
            _mb_uncompute_and(qc, ctrls[i], ancillas[i - 2], ancillas[i - 1])
        else:
            qc.ccx(ctrls[i], ancillas[i - 2], ancillas[i - 1])
    if MEASUREMENT_UNCOMPUTE:
        _mb_uncompute_and(qc, ctrls[0], ctrls[1], ancillas[0])
    else:
        qc.ccx(ctrls[0], ctrls[1], ancillas[0])


def cswap_toffoli(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit) -> None:
    qc.cx(b, a)
    qc.ccx(ctrl, a, b)
    qc.cx(b, a)


def controlled_rotate_right_by_two(qc: QuantumCircuit, ctrl: Qubit, reg: Sequence[Qubit]) -> None:
    """Controlled cyclic rotation by two positions to the right.

    This implements exactly the same permutation as the older adjacent-swap
    cascade

        for i in range(len(reg) - 3, -1, -1):
            cswap(reg[i], reg[i + 1])
            cswap(reg[i + 1], reg[i + 2])

    but uses the cycle decomposition of the permutation instead.  For a
    register of length m it uses m - gcd(m, 2) controlled swaps, rather than
    2(m-2).  All gates are real X/CX/CCX gates after recursive expansion; no
    resource-only formula gate is introduced.
    """
    r = list(reg)
    m = len(r)
    if m <= 2:
        return

    # Old value at position j moves to j+2 (mod m).  For each cycle of this
    # permutation, the transpositions (c0,c1), (c0,c2), ... implement the
    # desired cyclic move with one controlled swap per transposition.
    seen = [False] * m
    for start in range(m):
        if seen[start]:
            continue
        cycle: list[int] = []
        j = start
        while not seen[j]:
            seen[j] = True
            cycle.append(j)
            j = (j + 2) % m
        if len(cycle) <= 1:
            continue
        pivot = cycle[0]
        for j in cycle[1:]:
            cswap_toffoli(qc, ctrl, r[pivot], r[j])


def const_minus_inplace(qc: QuantumCircuit, reg: Sequence[Qubit], const: int, scratch: Sequence[Qubit]) -> None:
    """Scratch-clean affine involution reg <- const - reg (mod 2^w).

    This prepares endpoint registers such as R=n+2-(ell_s-1) and
    B=n+2-(ell_rp-1) under the paper's truth-minus-one encoding.  The
    implementation is (~x)+1+const, using only scratch-clean increment and
    add-constant primitives, so invoking the same map again truly restores both
    the register and all clean scratch bits.
    """
    w = len(reg)
    if len(scratch) < max(w + 1, w - 1):
        raise ValueError("const_minus_inplace needs scratch for increment and clean constant adder")
    for q in reg:
        qc.x(q)
    inc_mod2n_uncontrolled(qc, reg, scratch[: max(0, w - 1)])
    add_const_mod_2n(qc, reg, const, scratch[: w + 1])


def compute_eq_const(
    qc: QuantumCircuit,
    reg: Sequence[Qubit],
    const: int,
    flag: Qubit,
    scratch: Sequence[Qubit],
) -> None:
    """Toggle flag iff little-endian reg equals const.  Calling twice uncomputes."""
    w = len(reg)
    const %= 1 << w
    for i in range(w):
        if ((const >> i) & 1) == 0:
            qc.x(reg[i])
    mcx_vchain(qc, list(reg), flag, scratch)
    for i in range(w):
        if ((const >> i) & 1) == 0:
            qc.x(reg[i])


def compute_control(
    qc: QuantumCircuit,
    conditions: Sequence[tuple[Qubit, int]],
    out: Qubit,
    scratch: Sequence[Qubit],
) -> None:
    """Toggle out by the AND of literals (qubit == value).  Call again to uncompute."""
    controls = []
    for q, val in conditions:
        if val not in (0, 1):
            raise ValueError("condition value must be 0 or 1")
        if val == 0:
            qc.x(q)
        controls.append(q)
    mcx_vchain(qc, controls, out, scratch)
    for q, val in reversed(list(conditions)):
        if val == 0:
            qc.x(q)


def _make_condition_into(
    qc: QuantumCircuit,
    conditions: Sequence[tuple[Qubit, int]],
    out: Qubit,
    scratch: Sequence[Qubit],
) -> None:
    """Compatibility alias used by the Algorithm-3 scheduler."""
    compute_control(qc, conditions, out, scratch)


# Unary iteration
def unary_depth(num_labels: int) -> int:
    if num_labels <= 1:
        return 0
    return math.ceil(math.log2(num_labels - 1)) + 1


def _split_bit(labels: Sequence[int]) -> int:
    lo = min(labels)
    hi = max(labels)
    width = max(1, hi.bit_length())
    for b in reversed(range(width)):
        vals = {((x >> b) & 1) for x in labels}
        if len(vals) == 2:
            return b
    raise ValueError("cannot split a singleton label set")


def _and_with_index_bit(
    qc: QuantumCircuit,
    g: Qubit,
    bit: Qubit,
    h: Qubit,
    bit_value: int,
) -> None:
    if bit_value == 0:
        qc.x(bit)
    qc.ccx(g, bit, h)
    if bit_value == 0:
        qc.x(bit)


def _uncompute_and_with_index_bit(
    qc: QuantumCircuit,
    g: Qubit,
    bit: Qubit,
    h: Qubit,
    bit_value: int,
) -> None:
    if bit_value == 0:
        qc.x(bit)
    if MEASUREMENT_UNCOMPUTE:
        _mb_uncompute_and(qc, g, bit, h)
    else:
        qc.ccx(g, bit, h)
    if bit_value == 0:
        qc.x(bit)


def unary_iteration(
    qc: QuantumCircuit,
    *,
    index_reg: Sequence[Qubit],
    labels: Sequence[int],
    ctrl: Qubit,
    ancillas: Sequence[Qubit],
    leaf_fn: Callable[[int, Qubit], None],
    order: Literal["inc", "dec"] = "inc",
) -> None:
    """Pruned unary iteration over labels, as in our paper Eqs. (4)-(7)."""
    labels = sorted(set(labels))
    if not labels:
        return
    need = unary_depth(len(labels))
    if len(ancillas) < need:
        raise ValueError(f"unary_iteration needs {need} ancillas, got {len(ancillas)}")

    def rec(sub_labels: list[int], g: Qubit, depth: int) -> None:
        if len(sub_labels) == 1:
            leaf_fn(sub_labels[0], g)
            return
        b = _split_bit(sub_labels)
        if b >= len(index_reg):
            raise ValueError(f"index register too small for label bit {b}")
        zero_labels = [x for x in sub_labels if ((x >> b) & 1) == 0]
        one_labels = [x for x in sub_labels if ((x >> b) & 1) == 1]
        h = ancillas[depth]
        _and_with_index_bit(qc, g, index_reg[b], h, 0)  # h = g(1-A_b)
        if order == "inc":
            rec(zero_labels, h, depth + 1)
            qc.cx(g, h)  # h = g A_b
            rec(one_labels, h, depth + 1)
            qc.cx(g, h)  # back to g(1-A_b)
        elif order == "dec":
            qc.cx(g, h)  # h = g A_b
            rec(one_labels, h, depth + 1)
            qc.cx(g, h)  # back to g(1-A_b)
            rec(zero_labels, h, depth + 1)
        else:
            raise ValueError("order must be 'inc' or 'dec'")
        _uncompute_and_with_index_bit(qc, g, index_reg[b], h, 0)  # measurement/free-AND uncompute h

    rec(list(labels), ctrl, 0)


def dual_unary_iteration(
    qc: QuantumCircuit,
    *,
    index_a: Sequence[Qubit],
    index_b: Sequence[Qubit],
    labels: Sequence[int],
    ctrl_a: Qubit,
    ctrl_b: Qubit,
    ancillas_a: Sequence[Qubit],
    ancillas_b: Sequence[Qubit],
    leaf_fn: Callable[[int, Qubit, Qubit], None],
    order: Literal["inc", "dec"] = "inc",
) -> None:
    """Unary iteration that exposes e_a[j] and e_b[j] at each leaf j."""
    labels = sorted(set(labels))
    if not labels:
        return
    need = unary_depth(len(labels))
    if len(ancillas_a) < need or len(ancillas_b) < need:
        raise ValueError(f"dual_unary_iteration needs {need} ancillas per endpoint")

    def rec(sub_labels: list[int], ga: Qubit, gb: Qubit, depth: int) -> None:
        if len(sub_labels) == 1:
            leaf_fn(sub_labels[0], ga, gb)
            return
        bit = _split_bit(sub_labels)
        if bit >= len(index_a) or bit >= len(index_b):
            raise ValueError("endpoint register too small for unary label")
        zero_labels = [x for x in sub_labels if ((x >> bit) & 1) == 0]
        one_labels = [x for x in sub_labels if ((x >> bit) & 1) == 1]
        ha = ancillas_a[depth]
        hb = ancillas_b[depth]
        _and_with_index_bit(qc, ga, index_a[bit], ha, 0)
        _and_with_index_bit(qc, gb, index_b[bit], hb, 0)
        if order == "inc":
            rec(zero_labels, ha, hb, depth + 1)
            qc.cx(ga, ha)
            qc.cx(gb, hb)
            rec(one_labels, ha, hb, depth + 1)
            qc.cx(gb, hb)
            qc.cx(ga, ha)
        elif order == "dec":
            qc.cx(ga, ha)
            qc.cx(gb, hb)
            rec(one_labels, ha, hb, depth + 1)
            qc.cx(gb, hb)
            qc.cx(ga, ha)
            rec(zero_labels, ha, hb, depth + 1)
        else:
            raise ValueError("order must be 'inc' or 'dec'")
        _uncompute_and_with_index_bit(qc, gb, index_b[bit], hb, 0)
        _uncompute_and_with_index_bit(qc, ga, index_a[bit], ha, 0)

    rec(list(labels), ctrl_a, ctrl_b, 0)


def range_scan_leq(
    qc: QuantumCircuit,
    *,
    boundary_reg: Sequence[Qubit],
    k: int,
    K: int,
    ctrl: Qubit,
    range_acc: Qubit,
    ancillas: Sequence[Qubit],
    leaf_fn: Callable[[int, Qubit], None],
    order: Literal["inc", "dec"],
) -> None:
    """Scan j in [k,K] while range_acc equals ctrl & [j <= B]."""
    labels = list(range(k, K + 1))
    if order == "inc":
        qc.cx(ctrl, range_acc)

        def wrapped(j: int, ej: Qubit) -> None:
            leaf_fn(j, range_acc)
            qc.cx(ej, range_acc)  # turn off immediately after j = B

        unary_iteration(qc, index_reg=boundary_reg, labels=labels, ctrl=ctrl,
                        ancillas=ancillas, leaf_fn=wrapped, order="inc")
    elif order == "dec":
        def wrapped(j: int, ej: Qubit) -> None:
            qc.cx(ej, range_acc)  # turn on at j = B before use
            leaf_fn(j, range_acc)

        unary_iteration(qc, index_reg=boundary_reg, labels=labels, ctrl=ctrl,
                        ancillas=ancillas, leaf_fn=wrapped, order="dec")
        qc.cx(ctrl, range_acc)
    else:
        raise ValueError("order must be 'inc' or 'dec'")


def range_scan_geq(
    qc: QuantumCircuit,
    *,
    boundary_reg: Sequence[Qubit],
    k: int,
    K: int,
    ctrl: Qubit,
    range_acc: Qubit,
    ancillas: Sequence[Qubit],
    leaf_fn: Callable[[int, Qubit], None],
    order: Literal["inc", "dec"],
) -> None:
    """Scan j in [k,K] while range_acc equals ctrl & [j >= A]."""
    labels = list(range(k, K + 1))
    if order == "inc":
        def wrapped(j: int, ej: Qubit) -> None:
            qc.cx(ej, range_acc)  # turn on at j = A before use
            leaf_fn(j, range_acc)

        unary_iteration(qc, index_reg=boundary_reg, labels=labels, ctrl=ctrl,
                        ancillas=ancillas, leaf_fn=wrapped, order="inc")
        qc.cx(ctrl, range_acc)
    elif order == "dec":
        qc.cx(ctrl, range_acc)

        def wrapped(j: int, ej: Qubit) -> None:
            leaf_fn(j, range_acc)
            qc.cx(ej, range_acc)  # turn off after using j = A

        unary_iteration(qc, index_reg=boundary_reg, labels=labels, ctrl=ctrl,
                        ancillas=ancillas, leaf_fn=wrapped, order="dec")
    else:
        raise ValueError("order must be 'inc' or 'dec'")


# Controlled Cuccaro cells used inside unary-selected arithmetic blocks
def _controlled_toffoli(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, target: Qubit, pool: Sequence[Qubit]) -> None:
    mcx_vchain(qc, [ctrl, a, b], target, pool)


def controlled_maj(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, pool: Sequence[Qubit]) -> None:
    """Controlled MAJ cell from paper Figure 11(a).

    The two CNOTs are intentionally unconditional. Only the carry-updating
    Toffoli receives the location control; the matching UMA/UMA-dagger pass
    cancels the CNOT skeleton on an inactive lane.
    """
    qc.cx(c, a)
    qc.cx(c, b)
    _controlled_toffoli(qc, ctrl, a, b, c, pool)


def controlled_uma(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, pool: Sequence[Qubit]) -> None:
    """Controlled UMA cell from paper Figure 11(b)."""
    _controlled_toffoli(qc, ctrl, a, b, c, pool)
    qc.ccx(ctrl, b, a)
    qc.cx(c, b)
    qc.cx(c, a)


def controlled_maj_inv(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, pool: Sequence[Qubit]) -> None:
    """Exact inverse of Figure 11(a)."""
    _controlled_toffoli(qc, ctrl, a, b, c, pool)
    qc.cx(c, b)
    qc.cx(c, a)


def controlled_uma_inv(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, pool: Sequence[Qubit]) -> None:
    """Exact inverse of Figure 11(b)."""
    qc.cx(c, a)
    qc.cx(c, b)
    qc.ccx(ctrl, b, a)
    _controlled_toffoli(qc, ctrl, a, b, c, pool)


def _apply_cell(
    qc: QuantumCircuit,
    mode: Literal["add", "sub"],
    pass_kind: Literal["first", "second"],
    ctrl: Qubit,
    addend: Qubit,
    target: Qubit,
    carry: Qubit,
    pool: Sequence[Qubit],
) -> None:
    if mode == "add" and pass_kind == "first":
        controlled_maj(qc, ctrl, target, addend, carry, pool)
    elif mode == "add" and pass_kind == "second":
        controlled_uma(qc, ctrl, target, addend, carry, pool)
    elif mode == "sub" and pass_kind == "first":
        controlled_uma_inv(qc, ctrl, target, addend, carry, pool)
    elif mode == "sub" and pass_kind == "second":
        controlled_maj_inv(qc, ctrl, target, addend, carry, pool)
    else:
        raise ValueError("bad arithmetic cell mode/pass")


# Algorithm-3 shift and phase-update blocks
@lru_cache(maxsize=None)
def pre_shift_gate(*, work_size: int, shift_width: int, name: str = "PRE_SHIFT_NEW") -> Gate:
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(shift_width, "l_s")
    Scratch = QuantumRegister(shift_width + 4, "Scratch")
    qc = _block_circuit(Phase1, Phase2, Work2, l_s, Scratch, name=name)

    phase1_is0 = Scratch[0]
    both = Scratch[1]
    chain = list(Scratch[2:2 + max(0, shift_width - 1)])

    qc.x(Phase1[0])
    qc.cx(Phase1[0], phase1_is0)
    qc.x(Phase1[0])

    # one-position left shift of Work2 controlled by Phase1=0
    for i in range(work_size - 1):
        cswap_toffoli(qc, phase1_is0, Work2[i], Work2[i + 1])
    inc_mod2n_1ctrl(qc, phase1_is0, list(l_s), chain)

    # if Phase2=1, additionally right-shift by two and l_s -= 2.
    # The two-position shift is implemented as one controlled permutation
    # using cycle decomposition, not as two adjacent one-position shifts.
    qc.ccx(phase1_is0, Phase2[0], both)
    controlled_rotate_right_by_two(qc, both, list(Work2))
    dec_mod2n_1ctrl(qc, both, list(l_s), chain)
    dec_mod2n_1ctrl(qc, both, list(l_s), chain)
    qc.ccx(phase1_is0, Phase2[0], both)

    qc.x(Phase1[0])
    qc.cx(Phase1[0], phase1_is0)
    qc.x(Phase1[0])
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def post_shift_gate(*, work_size: int, shift_width: int, name: str = "POST_SHIFT_NEW") -> Gate:
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Work2 = QuantumRegister(work_size, "Work2")
    l_s = QuantumRegister(shift_width, "l_s")
    Scratch = QuantumRegister(shift_width + 4, "Scratch")
    qc = _block_circuit(Phase1, Phase2, Work2, l_s, Scratch, name=name)

    both = Scratch[0]
    chain = list(Scratch[1:1 + max(0, shift_width - 1)])

    for i in range(work_size - 1):
        cswap_toffoli(qc, Phase1[0], Work2[i], Work2[i + 1])
    inc_mod2n_1ctrl(qc, Phase1[0], list(l_s), chain)

    qc.ccx(Phase1[0], Phase2[0], both)
    controlled_rotate_right_by_two(qc, both, list(Work2))
    dec_mod2n_1ctrl(qc, both, list(l_s), chain)
    dec_mod2n_1ctrl(qc, both, list(l_s), chain)
    qc.ccx(Phase1[0], Phase2[0], both)
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def phase_update_gate(*, len_width: int, shift_width: int, include_shift_epoch: bool = False,
                      name: str = "PHASE_UPDATE_NEW") -> Gate:
    """Algorithm-3 phase update with exact encoded-zero predicates.

    When ``include_shift_epoch`` is true, the physical epoch qubit stores the
    complement of the high bit of the terminal-padding counter.  Consequently
    encoded ell_s=0 is ``l_s=all-ones`` together with ``ShiftEpoch=0``.  The
    epoch is included in both the compute and uncompute of the zero predicate,
    preventing a low-word wrap during padding from re-entering the arithmetic.
    """
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    l_q = QuantumRegister(len_width, "l_q")
    l_rp = QuantumRegister(len_width, "l_rp")
    l_s = QuantumRegister(shift_width, "l_s")
    ShiftEpoch = QuantumRegister(1, "ShiftEpoch") if include_shift_epoch else None
    shift_test_width = shift_width + (1 if include_shift_epoch else 0)
    eq_scratch = max(0, max(len_width, shift_test_width) - 2)
    Scratch = QuantumRegister(5 + eq_scratch, "Scratch")
    regs = [Phase1, Phase2, Sign, l_q, l_rp, l_s]
    if ShiftEpoch is not None:
        regs.append(ShiftEpoch)
    regs.append(Scratch)
    qc = _block_circuit(*regs, name=name)

    z_lq, z_lrp, z_ls, cond, tmp = Scratch[:5]
    pool = list(Scratch[5:])

    def toggle_zero(reg: Sequence[Qubit], flag: Qubit) -> None:
        mcx_vchain(qc, list(reg), flag, pool)

    def toggle_shift_zero() -> None:
        if ShiftEpoch is None:
            toggle_zero(l_s, z_ls)
        else:
            qc.x(ShiftEpoch[0])
            toggle_zero([*l_s, ShiftEpoch[0]], z_ls)
            qc.x(ShiftEpoch[0])

    toggle_zero(l_q, z_lq)
    toggle_zero(l_rp, z_lrp)
    toggle_shift_zero()

    qc.x(z_lrp)
    qc.ccx(z_lq, z_lrp, cond)
    qc.x(z_lrp)

    qc.cx(Sign[0], tmp)
    qc.cx(Phase1[0], tmp)
    qc.ccx(cond, tmp, Phase2[0])
    qc.cx(Phase1[0], tmp)
    qc.cx(Sign[0], tmp)
    qc.ccx(cond, Phase2[0], Sign[0])

    qc.x(z_lrp)
    qc.ccx(z_lq, z_lrp, cond)
    qc.x(z_lrp)

    qc.cx(z_ls, Phase1[0])
    qc.cx(z_ls, Phase2[0])

    toggle_shift_zero()
    toggle_zero(l_rp, z_lrp)
    toggle_zero(l_q, z_lq)
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def phase_update_inverse_gate(*, len_width: int, shift_width: int,
                              include_shift_epoch: bool = False,
                              name: str = "PHASE_UPDATE_NEW_INV"):
    """Explicit measurement-assisted inverse of :func:`phase_update_gate`.

    The zero predicates are reconstructed from the unchanged length metadata.
    The phase bits are then undone in exact reverse order, after which the same
    self-inverse dynamic MCX networks clear the predicate flags.  This retains
    measurement-based AND uncomputation and therefore has the same Toffoli cost
    model as the forward paper block.
    """
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    l_q = QuantumRegister(len_width, "l_q")
    l_rp = QuantumRegister(len_width, "l_rp")
    l_s = QuantumRegister(shift_width, "l_s")
    ShiftEpoch = QuantumRegister(1, "ShiftEpoch") if include_shift_epoch else None
    shift_test_width = shift_width + (1 if include_shift_epoch else 0)
    eq_scratch = max(0, max(len_width, shift_test_width) - 2)
    Scratch = QuantumRegister(5 + eq_scratch, "Scratch")
    regs = [Phase1, Phase2, Sign, l_q, l_rp, l_s]
    if ShiftEpoch is not None:
        regs.append(ShiftEpoch)
    regs.append(Scratch)
    qc = _block_circuit(*regs, name=name)

    z_lq, z_lrp, z_ls, cond, tmp = Scratch[:5]
    pool = list(Scratch[5:])

    def toggle_zero(reg: Sequence[Qubit], flag: Qubit) -> None:
        mcx_vchain(qc, list(reg), flag, pool)

    def toggle_shift_zero() -> None:
        if ShiftEpoch is None:
            toggle_zero(l_s, z_ls)
        else:
            qc.x(ShiftEpoch[0])
            toggle_zero([*l_s, ShiftEpoch[0]], z_ls)
            qc.x(ShiftEpoch[0])

    # Recreate all three zero predicates at the reverse boundary.
    toggle_zero(l_q, z_lq)
    toggle_zero(l_rp, z_lrp)
    toggle_shift_zero()

    # Undo the final z_ls-controlled phase transition first.
    qc.cx(z_ls, Phase2[0])
    qc.cx(z_ls, Phase1[0])

    # Recreate cond = z_lq AND NOT(z_lrp).
    qc.x(z_lrp)
    qc.ccx(z_lq, z_lrp, cond)
    qc.x(z_lrp)

    # Undo Sign ^= cond & Phase2, then undo the earlier Phase2 update using
    # the now-restored pre-Sign value.
    qc.ccx(cond, Phase2[0], Sign[0])
    qc.cx(Sign[0], tmp)
    qc.cx(Phase1[0], tmp)
    qc.ccx(cond, tmp, Phase2[0])
    qc.cx(Phase1[0], tmp)
    qc.cx(Sign[0], tmp)

    qc.x(z_lrp)
    qc.ccx(z_lq, z_lrp, cond)
    qc.x(z_lrp)

    toggle_shift_zero()
    toggle_zero(l_rp, z_lrp)
    toggle_zero(l_q, z_lq)
    return _finalize_block(qc)


# Location-controlled Swap / Add / Sub using unary iteration
@lru_cache(maxsize=None)
def lc_swap_unary_gate(*, k: int, K: int, len_width: int, name: str = "LC_SWAP_UNARY") -> Gate:
    """Paper Figure 9 location-controlled quotient-bit swap.

    Algorithm 3 merges the Phase-2 and Phase-3 swaps because both act on the
    same one-based Work1 position ``J = ell_t + ell_q + 1``.  The only
    phase-dependent operation is the update of ``ell_q`` after this block.
    """
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = unary_depth(M)
    scratch_size = len_width + 2 + depth

    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _block_circuit(Ctrl, Sign, Work1, l_t, l_q, Scratch, name=name)

    carry = Scratch[0]
    const_scratch = list(Scratch[: len_width + 2])
    path = list(Scratch[len_width + 2:])

    # Truth-minus-one encoding gives
    #   (ell_t-1) + (ell_q-1) + 3 = ell_t + ell_q + 1.
    qc.append(cuccaro_add_mod_2n_no_z_gate(len_width, name="ADD_lt_to_lq"), list(l_t) + list(l_q) + [carry])
    add_const_mod_2n(qc, l_q, 3, const_scratch)

    def leaf(j: int, ej: Qubit) -> None:
        cswap_toffoli(qc, ej, Sign[0], Work1[j - k])

    unary_iteration(qc, index_reg=l_q, labels=list(range(k, K + 1)), ctrl=Ctrl[0],
                    ancillas=path, leaf_fn=leaf, order="inc")

    sub_const_mod_2n(qc, l_q, 3, const_scratch)
    qc.append(cuccaro_sub_mod_2n_no_z_gate(len_width, name="SUB_lt_from_lq"), list(l_t) + list(l_q) + [carry])
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def lc_interval_addsub_unary_gate(
    *,
    n: int,
    k: int,
    K: int,
    len_width: int,
    shift_width: int,
    mode: Literal["add", "sub"],
    sign_update: bool,
    target: Literal["work1", "work2"],
    name: str,
) -> Gate:
    """Two-endpoint interval Add/Sub for r-side arithmetic.

    Endpoints are prepared exactly as PDF Eq. (13), with the Section-4.2
    truth-minus-one encoding:
        L = ell_t + ell_q + 2 -> l_q <- l_q + l_t + 4,
        R = n + 3 - ell_s     -> l_s <- n + 2 - l_s.
    The dual unary scan exposes r_j = Ctrl[R=j] and lambda_j = Ctrl[L=j].
    """
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = unary_depth(M)
    endpoint_width = max(len_width, shift_width)
    scratch_size = (endpoint_width + 2) + 2 * depth + 5

    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    l_s = QuantumRegister(shift_width, "l_s")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _block_circuit(Ctrl, Sign, Work1, Work2, l_t, l_q, l_s, Scratch, name=name)

    carry = Scratch[0]
    z = Scratch[1]
    acc = Scratch[2]
    const_scratch = list(Scratch[: endpoint_width + 2])
    anc_a = list(Scratch[endpoint_width + 2: endpoint_width + 2 + depth])
    anc_b = list(Scratch[endpoint_width + 2 + depth: endpoint_width + 2 + 2 * depth])
    pool = list(Scratch[endpoint_width + 2 + 2 * depth:])
    if len(pool) < 1:
        raise ValueError("internal pool too small")

    # Prepare raw L=(ell_t-1)+(ell_q-1)+4 and raw R=n+2-(ell_s-1).
    qc.append(cuccaro_add_mod_2n_no_z_gate(len_width, name="ADD_lt_to_lq"), list(l_t) + list(l_q) + [carry])
    add_const_mod_2n(qc, l_q, 4, const_scratch)
    const_minus_inplace(qc, l_s, n + 2, const_scratch)

    def qpair(j: int) -> tuple[Qubit, Qubit]:
        idx = j - k
        if target == "work1":
            addend = Work2[idx]
            tgt = Work1[idx]
        elif target == "work2":
            addend = Work1[idx]
            tgt = Work2[idx]
        else:
            raise ValueError("target must be 'work1' or 'work2'")
        return addend, tgt

    # First pass: j = K, K-1, ..., k.  Toggle at R before the cell and at L after.
    def leaf_first(j: int, rj: Qubit, lj: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(rj, acc)
        _apply_cell(qc, mode, "first", acc, addend, tgt, carry, pool)
        qc.cx(lj, acc)

    dual_unary_iteration(qc, index_a=l_s, index_b=l_q, labels=list(range(k, K + 1)),
                         ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
                         ancillas_b=anc_b, leaf_fn=leaf_first, order="dec")

    if sign_update:
        qc.cx(carry, Sign[0])

    # Second pass: j = k, k+1, ..., K.  Toggle at L before and at R after.
    def leaf_second(j: int, rj: Qubit, lj: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(lj, acc)
        _apply_cell(qc, mode, "second", acc, addend, tgt, carry, pool)
        qc.cx(rj, acc)

    dual_unary_iteration(qc, index_a=l_s, index_b=l_q, labels=list(range(k, K + 1)),
                         ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
                         ancillas_b=anc_b, leaf_fn=leaf_second, order="inc")

    # Restore endpoint registers.
    const_minus_inplace(qc, l_s, n + 2, const_scratch)
    sub_const_mod_2n(qc, l_q, 4, const_scratch)
    qc.append(cuccaro_sub_mod_2n_no_z_gate(len_width, name="SUB_lt_from_lq"), list(l_t) + list(l_q) + [carry])
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def lc_prefix_addsub_unary_gate(
    *,
    k: int,
    K: int,
    len_width: int,
    mode: Literal["add", "sub"],
    sign_update: bool,
    target: Literal["work1", "work2"],
    name: str,
) -> Gate:
    """One-sided Add/Sub for the t-side arithmetic.

    The physical prefix endpoint is ``R = ell_t + 1 + Phase2``. During
    Phase 3 this is the ordinary ``ell_t+1`` coefficient prefix. During
    Phase 4 the circular Work2 representation exposes one additional high
    lane, so that lane must participate in the comparison/addition. With
    the truth-minus-one encoding, the temporary endpoint is therefore
    ``l_t += 2 + Phase2``.
    """
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = unary_depth(M)
    scratch_size = (len_width + 2) + depth + 4

    Ctrl = QuantumRegister(1, "Ctrl")
    Phase2 = QuantumRegister(1, "Phase2")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _block_circuit(Ctrl, Phase2, Sign, Work1, Work2, l_t, Scratch, name=name)

    carry = Scratch[0]
    acc = Scratch[1]
    const_scratch = list(Scratch[: len_width + 2])
    path = list(Scratch[len_width + 2: len_width + 2 + depth])
    pool = list(Scratch[len_width + 2 + depth:])
    if len(pool) < 1:
        raise ValueError("internal pool too small")

    add_const_mod_2n(qc, l_t, 2, const_scratch)
    inc_mod2n_1ctrl(qc, Phase2[0], list(l_t), list(Scratch[: max(0, len_width - 1)]))

    def qpair(j: int) -> tuple[Qubit, Qubit]:
        idx = j - k
        if target == "work1":
            return Work2[idx], Work1[idx]
        if target == "work2":
            return Work1[idx], Work2[idx]
        raise ValueError("target must be 'work1' or 'work2'")

    # t and t' are little-endian at the left edge. Carry propagates from k
    # upward to R; the reverse pass scans back from R to k.
    qc.cx(Ctrl[0], acc)

    def leaf_first(j: int, ej: Qubit) -> None:
        addend, tgt = qpair(j)
        _apply_cell(qc, mode, "first", acc, addend, tgt, carry, pool)
        qc.cx(ej, acc)

    unary_iteration(qc, index_reg=l_t, labels=list(range(k, K + 1)), ctrl=Ctrl[0],
                    ancillas=path, leaf_fn=leaf_first, order="inc")

    if sign_update:
        qc.cx(carry, Sign[0])

    def leaf_second(j: int, ej: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(ej, acc)
        _apply_cell(qc, mode, "second", acc, addend, tgt, carry, pool)

    unary_iteration(qc, index_reg=l_t, labels=list(range(k, K + 1)), ctrl=Ctrl[0],
                    ancillas=path, leaf_fn=leaf_second, order="dec")
    qc.cx(Ctrl[0], acc)

    dec_mod2n_1ctrl(qc, Phase2[0], list(l_t), list(Scratch[: max(0, len_width - 1)]))
    sub_const_mod_2n(qc, l_t, 2, const_scratch)
    return _finalize_block(qc)



# Unary length-update circuits
def _upper_zero_map(
    qc: QuantumCircuit,
    *,
    ctrl: Qubit,
    boundary_B: Sequence[Qubit],
    bits: Sequence[Qubit],
    dirty: Sequence[Qubit],
    k: int,
    K: int,
    scratch: Sequence[Qubit],
) -> None:
    """UZ_u: dirty[j] ^= z_j where z_j = AND_{h=j..K} not a_h."""
    M = K - k + 1
    depth = unary_depth(M)
    need = depth + 2
    if len(scratch) < need:
        raise ValueError(f"upper-zero map needs at least {need} clean scratch qubits")
    path = list(scratch[:depth])
    range_acc = scratch[depth]
    a_tmp = scratch[depth + 1]

    def compute_a(bctrl: Qubit, u: Qubit) -> None:
        _and_with_index_bit(qc, bctrl, u, a_tmp, 1)

    def uncompute_a(bctrl: Qubit, u: Qubit) -> None:
        _uncompute_and_with_index_bit(qc, bctrl, u, a_tmp, 1)

    def use_not_a_with_gnext(a: Qubit, gnext: Qubit, target: Qubit) -> None:
        qc.x(a)
        qc.ccx(a, gnext, target)
        qc.x(a)

    def use_not_a_cnot(a: Qubit, target: Qubit) -> None:
        qc.x(a)
        qc.cx(a, target)
        qc.x(a)

    # Forward core j=k..K-1, then base j=K.
    def leaf_forward(j: int, bctrl: Qubit) -> None:
        idx = j - k
        compute_a(bctrl, bits[idx])
        if j < K:
            use_not_a_with_gnext(a_tmp, dirty[idx + 1], dirty[idx])
        else:
            # Base z_K is Ctrl & NOT(a_K); this makes every z_j zero when
            # the enclosing length-update block is inactive.
            qc.x(a_tmp)
            qc.ccx(ctrl, a_tmp, dirty[idx])
            qc.x(a_tmp)
        uncompute_a(bctrl, bits[idx])

    range_scan_leq(qc, boundary_reg=boundary_B, k=k, K=K, ctrl=ctrl,
                   range_acc=range_acc, ancillas=path, leaf_fn=leaf_forward, order="inc")

    # Reverse core j=K-1..k.  The leaf j=K is deliberately skipped.
    def leaf_reverse(j: int, bctrl: Qubit) -> None:
        if j >= K:
            return
        idx = j - k
        compute_a(bctrl, bits[idx])
        use_not_a_with_gnext(a_tmp, dirty[idx + 1], dirty[idx])
        uncompute_a(bctrl, bits[idx])

    range_scan_leq(qc, boundary_reg=boundary_B, k=k, K=K, ctrl=ctrl,
                   range_acc=range_acc, ancillas=path, leaf_fn=leaf_reverse, order="dec")


def _lower_zero_map(
    qc: QuantumCircuit,
    *,
    ctrl: Qubit,
    boundary_A: Sequence[Qubit],
    bits: Sequence[Qubit],
    dirty: Sequence[Qubit],
    k: int,
    K: int,
    scratch: Sequence[Qubit],
) -> None:
    """UZ_l mirror: dirty[j] ^= y_j where y_j = AND_{h=k..j} not a_h."""
    M = K - k + 1
    depth = unary_depth(M)
    need = depth + 2
    if len(scratch) < need:
        raise ValueError(f"lower-zero map needs at least {need} clean scratch qubits")
    path = list(scratch[:depth])
    range_acc = scratch[depth]
    a_tmp = scratch[depth + 1]

    def compute_a(bctrl: Qubit, u: Qubit) -> None:
        _and_with_index_bit(qc, bctrl, u, a_tmp, 1)

    def uncompute_a(bctrl: Qubit, u: Qubit) -> None:
        _uncompute_and_with_index_bit(qc, bctrl, u, a_tmp, 1)

    def use_not_a_with_gprev(a: Qubit, gprev: Qubit, target: Qubit) -> None:
        qc.x(a)
        qc.ccx(a, gprev, target)
        qc.x(a)

    def use_not_a_cnot(a: Qubit, target: Qubit) -> None:
        qc.x(a)
        qc.cx(a, target)
        qc.x(a)

    # Forward mirror core j=K..k+1, then base j=k.
    def leaf_forward(j: int, bctrl: Qubit) -> None:
        idx = j - k
        compute_a(bctrl, bits[idx])
        if j > k:
            use_not_a_with_gprev(a_tmp, dirty[idx - 1], dirty[idx])
        else:
            # Base y_k is Ctrl & NOT(a_k), so the dirty map is inactive
            # when Ctrl=0 without adding Ctrl to every constant write.
            qc.x(a_tmp)
            qc.ccx(ctrl, a_tmp, dirty[idx])
            qc.x(a_tmp)
        uncompute_a(bctrl, bits[idx])

    range_scan_geq(qc, boundary_reg=boundary_A, k=k, K=K, ctrl=ctrl,
                   range_acc=range_acc, ancillas=path, leaf_fn=leaf_forward, order="dec")

    # Reverse mirror core j=k+1..K.  The leaf j=k is skipped.
    def leaf_reverse(j: int, bctrl: Qubit) -> None:
        if j <= k:
            return
        idx = j - k
        compute_a(bctrl, bits[idx])
        use_not_a_with_gprev(a_tmp, dirty[idx - 1], dirty[idx])
        uncompute_a(bctrl, bits[idx])

    range_scan_geq(qc, boundary_reg=boundary_A, k=k, K=K, ctrl=ctrl,
                   range_acc=range_acc, ancillas=path, leaf_fn=leaf_reverse, order="inc")


def highest_position_xor_write(
    qc: QuantumCircuit,
    *,
    ctrl: Qubit,
    boundary_B: Sequence[Qubit],
    bits: Sequence[Qubit],
    dirty: Sequence[Qubit],
    target_len: Sequence[Qubit],
    k: int,
    K: int,
    scratch: Sequence[Qubit],
) -> None:
    """XOR the encoded highest valid nonzero position into target_len.

    Eq. (28) writes the truth length.  Because Section 4.2 stores truth-minus
    one, this routine writes ``highest_position - 1``; if no valid nonzero bit
    exists and the external Ctrl is 1, it writes ``-1 mod 2^w`` to encode truth
    length 0.  The seed is controlled by Ctrl.  The paired constant writes are
    controlled only by the borrowed dirty qubits, exactly as in Figures 12--13;
    the zero-map itself includes Ctrl, so the two dirty-controlled writes cancel
    when the enclosing block is inactive.
    """
    if len(bits) != K - k + 1 or len(dirty) != K - k + 1:
        raise ValueError("bits/dirty length does not match active window")
    M = K - k + 1
    depth = unary_depth(M)
    need = depth + 2
    if len(scratch) < need:
        raise ValueError(f"highest_position_xor_write needs {need} scratch qubits")

    mask = (1 << len(target_len)) - 1

    def apply_dirty_writes() -> None:
        for j in range(K, k, -1):
            # Enc(j) xor Enc(j-1) = (j-1) xor (j-2).
            xor_const_into_reg_controls(qc, target_len, ((j - 1) ^ (j - 2)) & mask,
                                        ctrls=[dirty[j - k]], scratch=scratch)
        # Enc(k) xor Enc(0), where Enc(0)=-1=mask.
        xor_const_into_reg_controls(qc, target_len, ((k - 1) ^ mask) & mask,
                                    ctrls=[dirty[0]], scratch=scratch)

    # Controlled Enc(K) seed.
    xor_const_into_reg_controls(qc, target_len, (K - 1) & mask, ctrls=[ctrl], scratch=scratch)
    apply_dirty_writes()

    _upper_zero_map(qc, ctrl=ctrl, boundary_B=boundary_B, bits=bits, dirty=dirty,
                    k=k, K=K, scratch=scratch)

    apply_dirty_writes()

    _upper_zero_map(qc, ctrl=ctrl, boundary_B=boundary_B, bits=bits, dirty=dirty,
                    k=k, K=K, scratch=scratch)


def right_length_xor_write(
    qc: QuantumCircuit,
    *,
    n: int,
    ctrl: Qubit,
    boundary_A: Sequence[Qubit],
    bits: Sequence[Qubit],
    dirty: Sequence[Qubit],
    target_len: Sequence[Qubit],
    k: int,
    K: int,
    scratch: Sequence[Qubit],
) -> None:
    """XOR encoded right length ``len-1`` selected by the lowest valid bit.

    If the lowest valid nonzero right-side position is m, the truth length is
    n+4-m and the stored value is n+3-m.  The no-nonzero truth length 0 is
    encoded by -1.  The zero-map includes the external Ctrl, while the paired
    borrowed-qubit constant writes use only their dirty controls; therefore the
    inactive block is exactly the identity without promoting those CNOT writes
    to Toffolis.
    """
    if len(bits) != K - k + 1 or len(dirty) != K - k + 1:
        raise ValueError("bits/dirty length does not match active window")
    M = K - k + 1
    depth = unary_depth(M)
    need = depth + 2
    if len(scratch) < need:
        raise ValueError(f"right_length_xor_write needs {need} scratch qubits")

    mask = (1 << len(target_len)) - 1

    def val(pos: int) -> int:
        # Enc(n+4-pos) = n+3-pos for one-based physical work positions.
        return (n + 3 - pos) & mask

    def apply_dirty_writes() -> None:
        for j in range(k, K):
            xor_const_into_reg_controls(qc, target_len, val(j) ^ val(j + 1),
                                        ctrls=[dirty[j - k]], scratch=scratch)
        # Enc(length at K) xor Enc(0), where Enc(0)=-1=mask.
        xor_const_into_reg_controls(qc, target_len, val(K) ^ mask,
                                    ctrls=[dirty[K - k]], scratch=scratch)

    xor_const_into_reg_controls(qc, target_len, val(k), ctrls=[ctrl], scratch=scratch)
    apply_dirty_writes()

    _lower_zero_map(qc, ctrl=ctrl, boundary_A=boundary_A, bits=bits, dirty=dirty,
                    k=k, K=K, scratch=scratch)

    apply_dirty_writes()

    _lower_zero_map(qc, ctrl=ctrl, boundary_A=boundary_A, bits=bits, dirty=dirty,
                    k=k, K=K, scratch=scratch)


@lru_cache(maxsize=None)
def len_update_lt_unary_gate(*, n: int, k: int, K: int, len_width: int, name: str = "LEN_LT_UNARY") -> Gate:
    M = K - k + 1
    depth = unary_depth(M)
    scratch_size = (len_width + 2) + depth + 2
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch, name=name)

    const_scratch = list(Scratch[: len_width + 2])
    uz_scratch = list(Scratch[len_width + 2:])

    # Raw B=n+3-ell_rp=n+2-(ell_rp-1), prepared in-place and restored.
    const_minus_inplace(qc, l_rp, n + 2, const_scratch)
    highest_position_xor_write(qc, ctrl=Ctrl[0], boundary_B=l_rp, bits=Work2, dirty=Work1,
                               target_len=l_t, k=k, K=K, scratch=uz_scratch)
    highest_position_xor_write(qc, ctrl=Ctrl[0], boundary_B=l_rp, bits=Work1, dirty=Work2,
                               target_len=l_t, k=k, K=K, scratch=uz_scratch)
    const_minus_inplace(qc, l_rp, n + 2, const_scratch)
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def len_update_lrp_unary_gate(*, n: int, k: int, K: int, len_width: int, name: str = "LEN_LRP_UNARY") -> Gate:
    M = K - k + 1
    depth = unary_depth(M)
    scratch_size = (len_width + 2) + depth + 2
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch, name=name)

    const_scratch = list(Scratch[: len_width + 2])
    uz_scratch = list(Scratch[len_width + 2:])

    # Raw A=ell_t+2=(ell_t-1)+3, prepared in-place and restored.
    add_const_mod_2n(qc, l_t, 3, const_scratch)
    # Old right length from Work1, new right length from Work2 after the full Work SWAP.
    right_length_xor_write(qc, n=n, ctrl=Ctrl[0], boundary_A=l_t, bits=Work1, dirty=Work2,
                           target_len=l_rp, k=k, K=K, scratch=uz_scratch)
    right_length_xor_write(qc, n=n, ctrl=Ctrl[0], boundary_A=l_t, bits=Work2, dirty=Work1,
                           target_len=l_rp, k=k, K=K, scratch=uz_scratch)
    sub_const_mod_2n(qc, l_t, 3, const_scratch)
    return _finalize_block(qc)


@lru_cache(maxsize=None)
def swap_work_and_len_unary_gate(
    *,
    n: int,
    len_width: int,
    k4: int,
    K4: int,
    k5: int,
    K5: int,
    name: str = "SWAP_AND_LEN_UNARY",
) -> Gate:
    work_size = n + 3
    depth4 = unary_depth(K4 - k4 + 1)
    # The right-length unary decoder needs the boundary successor label in
    # addition to the arithmetic support [k5,K5]. This does not enlarge the
    # data-active window; it completes the telescoping endpoint decode.
    K5_decode = min(K5 + 1, n + 3)
    depth5 = unary_depth(K5_decode - k5 + 1)
    scratch4 = (len_width + 2) + depth4 + 2
    scratch5 = (len_width + 2) + depth5 + 2

    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch4 = QuantumRegister(scratch4, "Scratch_lt")
    Scratch5 = QuantumRegister(scratch5, "Scratch_lrp")
    qc = _block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch4, Scratch5, name=name)

    for i in range(work_size):
        cswap_toffoli(qc, Ctrl[0], Work1[i], Work2[i])

    gate_lt = len_update_lt_unary_gate(n=n, k=k4, K=K4, len_width=len_width)
    _append_with_optional_clbits(qc, gate_lt, [Ctrl[0]] + list(Work1[k4 - 1:K4]) + list(Work2[k4 - 1:K4])
                                + list(l_t) + list(l_rp) + list(Scratch4))

    gate_lrp = len_update_lrp_unary_gate(n=n, k=k5, K=K5_decode, len_width=len_width)
    _append_with_optional_clbits(qc, gate_lrp, [Ctrl[0]] + list(Work1[k5 - 1:K5_decode]) + list(Work2[k5 - 1:K5_decode])
                                + list(l_t) + list(l_rp) + list(Scratch5))
    return _finalize_block(qc)


# Composition layer
def _ceil_safe(x: float, eps: float = 1e-12) -> int:
    return math.ceil(x - eps)


def _floor_safe(x: float, eps: float = 1e-12) -> int:
    return math.floor(x + eps)


def Nmax_steps(n: int) -> int:
    return 4 * math.ceil(C_EEA * n)


def active_windows(n: int, T: int) -> dict[str, tuple[int, int]]:  # type: ignore[override]
    """Revised Appendix-A.2 step-dependent active-window bounds.

    This function exposes the analytic bounds used for resource analysis.  The
    executable production path may deliberately choose endpoint-complete
    conservative supports via ``eea_circuit_s835_fastdual._step_windows``; the
    two concepts are kept separate so correctness tests never silently rely on
    an unverified narrow support.
    """
    if T < 1:
        raise ValueError("Algorithm-3 step index T is 1-based; need T >= 1")
    c = C_EEA
    delta = DELTA_EEA

    k1 = max(_ceil_safe((T - n - 1 - 4.0 * delta) / (4.0 * c - 1.0)), 1) + 2
    K1 = n + 3

    k2 = max(_ceil_safe((T - 3.0 * (n + 1) - 4.0 * delta) / (4.0 * c - 3.0)), 1) + 1
    K2 = min(_floor_safe(T / 2.0) + 2, n + 2)

    K3 = min(_floor_safe(T / 4.0) + 2, n + 1)

    k4 = max(_ceil_safe((T - 4.0 * (n + 1) - 4.0 * delta) / (4.0 * c - 4.0)), 1)
    K4 = min(_floor_safe(T / 4.0) + 3, n + 2)

    k5 = max(_ceil_safe((T - 4.0 * delta) / (4.0 * c)), 1) + 2
    K5 = n + 3

    # Empty analytic windows can occur on step families that are inactive at T.
    # Keep their endpoints within the physical register so construction tools
    # can inspect them without creating a malformed slice.
    out = {
        "r_addsub": (max(1, min(k1, n + 3)), K1),
        "swap": (max(1, min(k2, n + 2)), max(1, K2)),
        "t_addsub": (1, max(1, K3)),
        "len_update_lt": (max(1, min(k4, n + 3)), max(1, K4)),
        "len_update_lrp": (max(1, min(k5, n + 3)), max(1, K5)),
    }
    return out


def _r_addsub_scratch_size(n: int, len_width: int, shift_width: int, window_size: int) -> int:
    return (max(len_width, shift_width) + 2) + 2 * unary_depth(window_size) + 5


def _swap_scratch_size(len_width: int, window_size: int) -> int:
    return (len_width + 2) + unary_depth(window_size)


def _t_addsub_scratch_size(len_width: int, window_size: int) -> int:
    return (len_width + 2) + unary_depth(window_size) + 4


def _len_update_scratch_size(len_width: int, window_size: int) -> int:
    return (len_width + 2) + unary_depth(window_size) + 2


def qiskit_paper_aux_size(n: int, len_width: int, shift_width: int, T_max: Optional[int] = None,
                          include_algorithm1: bool = False) -> int:
    """Clean auxiliary pool size required by this explicit Qiskit realization.

    Paper Section 4 counts the location-controlled blocks at the Toffoli-network
    level.  Qiskit gate definitions need named qubits for the reversible-AND path
    and constant-adder work bits; this function allocates one shared pool and the
    builder reuses it across all blocks.
    """
    if T_max is None:
        T_max = Nmax_steps(n)
    max_r = max_swap = max_t = max_l4 = max_l5 = 1
    for T in range(1, T_max + 1):
        w = active_windows(n, T)
        max_r = max(max_r, w["r_addsub"][1] - w["r_addsub"][0] + 1)
        max_swap = max(max_swap, w["swap"][1] - w["swap"][0] + 1)
        max_t = max(max_t, w["t_addsub"][1] - w["t_addsub"][0] + 1)
        max_l4 = max(max_l4, w["len_update_lt"][1] - w["len_update_lt"][0] + 1)
        max_l5 = max(max_l5, w["len_update_lrp"][1] - w["len_update_lrp"][0] + 1)

    step_need = max(
        shift_width + 4,                                      # pre/post shift
        _r_addsub_scratch_size(n, len_width, shift_width, max_r) + 2,  # reserve tmp controls
        _swap_scratch_size(len_width, max_swap),
        _t_addsub_scratch_size(len_width, max_t) + 1,         # reserve tmp control
        max(len_width, shift_width) + 3,                      # all-ones zero tests in phase update
        _len_update_scratch_size(len_width, max(max_l4, max_l5)),
        len_width - 1 + 2,                                    # controlled +/-1 on l_q plus reserve
        max(len_width, shift_width) + 6,                      # literal-control helpers
    )
    if include_algorithm1:
        # Comparator, bit-length initialization, and controlled p-x use the same
        # shared pool before/after the Algorithm-3 steps.
        step_need = max(step_need, n + 2, len_width + 3, shift_width + 3)
    return step_need


def _append_with_shared_aux(qc: QuantumCircuit, gate, fixed_qubits: Sequence[Qubit],
                            Aux: QuantumRegister, reserved: int = 0) -> None:
    fixed = list(fixed_qubits)
    need = gate.num_qubits - len(fixed)
    if need < 0:
        raise ValueError(f"too many fixed qubits supplied for {gate.name}")
    if reserved + need > len(Aux):
        raise ValueError(
            f"not enough shared Aux for {gate.name}: need reserved({reserved})+scratch({need})="
            f"{reserved + need}, have {len(Aux)}"
        )
    _append_with_optional_clbits(qc, gate, fixed + list(Aux[reserved: reserved + need]))


@lru_cache(maxsize=None)
def swap_work_and_len_unary_shared_gate(
    *,
    n: int,
    len_width: int,
    k4: int,
    K4: int,
    k5: int,
    K5: int,
    name: str = "SWAP_AND_LEN_UNARY_SHARED",
) -> Gate:
    """Full Work SWAP plus the two serial length updates using one shared scratch pool."""
    work_size = n + 3
    scratch4 = _len_update_scratch_size(len_width, K4 - k4 + 1)
    K5_decode = min(K5 + 1, n + 3)
    scratch5 = _len_update_scratch_size(len_width, K5_decode - k5 + 1)
    scratch_size = max(scratch4, scratch5)

    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch, name=name)

    for i in range(work_size):
        cswap_toffoli(qc, Ctrl[0], Work1[i], Work2[i])

    gate_lt = len_update_lt_unary_gate(n=n, k=k4, K=K4, len_width=len_width)
    _append_with_optional_clbits(qc, gate_lt, [Ctrl[0]] + list(Work1[k4 - 1:K4]) + list(Work2[k4 - 1:K4])
                                + list(l_t) + list(l_rp) + list(Scratch[:scratch4]))

    gate_lrp = len_update_lrp_unary_gate(n=n, k=k5, K=K5_decode, len_width=len_width)
    _append_with_optional_clbits(qc, gate_lrp, [Ctrl[0]] + list(Work1[k5 - 1:K5_decode]) + list(Work2[k5 - 1:K5_decode])
                                + list(l_t) + list(l_rp) + list(Scratch[:scratch5]))
    return _finalize_block(qc)


def make_global_registers(*, n: int, len_width: int, shift_width: int,
                          T_max: Optional[int] = None, include_algorithm1: bool = False,
                          aux_size: Optional[int] = None):  # type: ignore[override]
    """Registers for the Qiskit paper construction.

    Work2[3:3+n] is the input x field of Algorithm 1.  The same shared Aux
    register is reused by every block; no per-block scratch pools are allocated.
    """
    work_size = n + 3
    Phase1 = QuantumRegister(1, "Phase1")
    Phase2 = QuantumRegister(1, "Phase2")
    Iter = QuantumRegister(1, "Iter")
    Sign = QuantumRegister(1, "Sign")
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    l_s = QuantumRegister(shift_width, "l_s")
    l_rp = QuantumRegister(len_width, "l_rp")
    if aux_size is None:
        aux_size = qiskit_paper_aux_size(n, len_width, shift_width, T_max, include_algorithm1)
    Aux = QuantumRegister(aux_size, "Aux")
    return Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux


def append_one_step_T(  # type: ignore[override]
    qc: QuantumCircuit,
    *,
    T: int,
    n: int,
    len_width: int,
    shift_width: int,
    Phase1: QuantumRegister,
    Phase2: QuantumRegister,
    Iter: QuantumRegister,
    Sign: QuantumRegister,
    Ctrl: QuantumRegister,
    Work1: QuantumRegister,
    Work2: QuantumRegister,
    l_t: QuantumRegister,
    l_q: QuantumRegister,
    l_s: QuantumRegister,
    l_rp: QuantumRegister,
    Aux: QuantumRegister,
) -> None:
    """Append Algorithm 3, with gates ordered as Figures 5--6 and windows from Section 4.5."""
    work_size = n + 3
    windows = active_windows(n, T)
    k1, K1 = windows["r_addsub"]
    k2, K2 = windows["swap"]
    k3, K3 = windows["t_addsub"]
    k4, K4 = windows["len_update_lt"]
    k5, K5 = windows["len_update_lrp"]

    ctrl = Ctrl[0]
    tmp = Aux[0]
    pool = list(Aux[1:])

    # Pre-shift: Phase1=0 controls Shift_1 and +1; Phase2 adds Shift_-2 and -2.
    pre = pre_shift_gate(work_size=work_size, shift_width=shift_width)
    _append_with_shared_aux(qc, pre, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s), Aux)

    # R-side subtraction: controlled by Phase1=0.
    _make_condition_into(qc, [(Phase1[0], 0)], ctrl, pool)
    rsub = lc_interval_addsub_unary_gate(n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
                                         mode="sub", sign_update=True, target="work1", name="R_SUB_UNARY")
    _append_with_shared_aux(qc, rsub,
                            [ctrl, Sign[0]] + list(Work1[k1 - 1:K1]) + list(Work2[k1 - 1:K1])
                            + list(l_t) + list(l_q) + list(l_s), Aux)
    _make_condition_into(qc, [(Phase1[0], 0)], ctrl, pool)

    # Algorithm 3: if Phase1=0 and Phase2=1 then Sign ^= 1.
    _make_condition_into(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, pool)
    qc.cx(ctrl, Sign[0])
    _make_condition_into(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, pool)

    # R-side addition: Phase1=0 and (Phase2=0 or Sign=0), i.e. not(Phase2&Sign).
    qc.ccx(Phase2[0], Sign[0], tmp)
    _make_condition_into(qc, [(Phase1[0], 0), (tmp, 0)], ctrl, pool)
    radd = lc_interval_addsub_unary_gate(n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
                                         mode="add", sign_update=False, target="work1", name="R_ADD_UNARY")
    _append_with_shared_aux(qc, radd,
                            [ctrl, Sign[0]] + list(Work1[k1 - 1:K1]) + list(Work2[k1 - 1:K1])
                            + list(l_t) + list(l_q) + list(l_s), Aux, reserved=1)
    _make_condition_into(qc, [(Phase1[0], 0), (tmp, 0)], ctrl, pool)
    qc.ccx(Phase2[0], Sign[0], tmp)

    # Phase 2 increases ell_q before selecting the newly inserted lane;
    # Phase 3 selects the existing least-significant lane and decreases ell_q
    # afterwards.  The shared Figure-9 selector therefore remains exactly
    # J=ell_t+ell_q+1.
    _make_condition_into(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, pool)
    inc_mod2n_1ctrl(qc, ctrl, list(l_q), list(Aux[: max(0, len_width - 1)]))
    _make_condition_into(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, pool)

    # Location-controlled Swap: controlled by Phase1 xor Phase2.
    qc.cx(Phase1[0], ctrl)
    qc.cx(Phase2[0], ctrl)
    lcs = lc_swap_unary_gate(k=k2, K=K2, len_width=len_width)
    _append_with_shared_aux(qc, lcs,
                            [ctrl, Sign[0]] + list(Work1[k2 - 1:K2]) + list(l_t) + list(l_q), Aux)
    qc.cx(Phase2[0], ctrl)
    qc.cx(Phase1[0], ctrl)

    _make_condition_into(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, pool)
    dec_mod2n_1ctrl(qc, ctrl, list(l_q), list(Aux[: max(0, len_width - 1)]))
    _make_condition_into(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, pool)

    # T-side subtraction: Phase1=1 and (Phase2=1 or Sign=0).
    _make_condition_into(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, pool)
    _make_condition_into(qc, [(Phase1[0], 1), (tmp, 0)], ctrl, pool)
    tsub = lc_prefix_addsub_unary_gate(k=k3, K=K3, len_width=len_width,
                                       mode="sub", sign_update=False, target="work2", name="T_SUB_UNARY")
    _append_with_shared_aux(qc, tsub,
                            [ctrl, Phase2[0], Sign[0]] + list(Work1[k3 - 1:K3]) + list(Work2[k3 - 1:K3])
                            + list(l_t), Aux, reserved=1)
    _make_condition_into(qc, [(Phase1[0], 1), (tmp, 0)], ctrl, pool)
    _make_condition_into(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, pool)

    # Sign ^= 1 if Phase1=1.
    qc.cx(Phase1[0], Sign[0])

    # T-side addition: controlled by Phase1=1 and updates Sign from the carry.
    _make_condition_into(qc, [(Phase1[0], 1)], ctrl, pool)
    tadd = lc_prefix_addsub_unary_gate(k=k3, K=K3, len_width=len_width,
                                       mode="add", sign_update=True, target="work2", name="T_ADD_UNARY")
    _append_with_shared_aux(qc, tadd,
                            [ctrl, Phase2[0], Sign[0]] + list(Work1[k3 - 1:K3]) + list(Work2[k3 - 1:K3])
                            + list(l_t), Aux)
    _make_condition_into(qc, [(Phase1[0], 1)], ctrl, pool)

    # Post-shift.
    post = post_shift_gate(work_size=work_size, shift_width=shift_width)
    _append_with_shared_aux(qc, post, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s), Aux)

    # Phase update.
    pupdate = phase_update_gate(len_width=len_width, shift_width=shift_width)
    _append_with_shared_aux(qc, pupdate,
                            [Phase1[0], Phase2[0], Sign[0]] + list(l_q) + list(l_rp) + list(l_s), Aux)

    # End-of-iteration full Work SWAP and length updates, only on 4 | T.
    # Encoded zero is all ones; compute equality-to-all-ones flags explicitly
    # instead of using a sign/MSB bit.
    if T % 4 == 0:
        z_lq = Aux[0]
        z_ls = Aux[1]
        eq_pool = list(Aux[2:])
        mcx_vchain(qc, list(l_q), z_lq, eq_pool)
        mcx_vchain(qc, list(l_s), z_ls, eq_pool)
        qc.ccx(z_lq, z_ls, ctrl)
        swlen = swap_work_and_len_unary_shared_gate(n=n, len_width=len_width, k4=k4, K4=K4, k5=k5, K5=K5)
        _append_with_shared_aux(qc, swlen,
                                [ctrl] + list(Work1) + list(Work2) + list(l_t) + list(l_rp), Aux, reserved=2)
        qc.cx(ctrl, Iter[0])
        qc.ccx(z_lq, z_ls, ctrl)
        mcx_vchain(qc, list(l_s), z_ls, eq_pool)
        mcx_vchain(qc, list(l_q), z_lq, eq_pool)



def build_full_steps_circuit(n: int, len_width: int, shift_width: int,
                             T_max: Optional[int] = None,
                             aux_size: Optional[int] = None,
                             T_start: int = 1,
                             T_end: Optional[int] = None) -> QuantumCircuit:  # type: ignore[override]
    # PATCH: the uploaded file referenced T_start/T_end without defining them.
    # The explicit step-range parameters below make the Algorithm-3 builder callable
    # and allow chunked construction of the Figure-5/Figure-6 step circuit.
    if T_max is None:
        T_max = Nmax_steps(n)

    _require_qiskit()
    if T_end is None:
        T_end = T_max
    if T_start < 1 or T_end > T_max or T_start > T_end:
        raise ValueError(f"bad step range [{T_start}, {T_end}] for T_max={T_max}")
    regs = make_global_registers(n=n, len_width=len_width, shift_width=shift_width, T_max=T_max, aux_size=aux_size)
    if MEASUREMENT_UNCOMPUTE:
        qc = QuantumCircuit(*regs, ClassicalRegister(1, "m"), name="MODINV_STEPS_UNARY_QISKIT_DYNAMIC")
    else:
        qc = QuantumCircuit(*regs, name="MODINV_STEPS_UNARY_QISKIT_PAPER")
    (Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux) = regs
    for T in range(T_start, T_end + 1):
        append_one_step_T(qc, T=T, n=n, len_width=len_width, shift_width=shift_width,
                          Phase1=Phase1, Phase2=Phase2, Iter=Iter, Sign=Sign, Ctrl=Ctrl,
                          Work1=Work1, Work2=Work2, l_t=l_t, l_q=l_q, l_s=l_s, l_rp=l_rp, Aux=Aux)
    return qc


# Algorithm-1 wrapper around the step circuit
def const_minus_le_gate(n: int, const: int, name: Optional[str] = None) -> Gate:
    """Gate for little-endian reg <- const - reg (mod 2^n)."""
    if name is None:
        name = f"CONST_MINUS_{const}_LE_{n}"
    Reg = QuantumRegister(n, "reg")
    Scratch = QuantumRegister(n + 1, "Scratch")
    qc = _block_circuit(Reg, Scratch, name=name)
    const_minus_inplace(qc, Reg, const, Scratch)
    return _finalize_block(qc)


def controlled_const_minus_le(
    qc: QuantumCircuit,
    ctrl: Qubit,
    reg_le: Sequence[Qubit],
    const: int,
    scratch: Sequence[Qubit],
) -> None:
    """Controlled little-endian affine map ``reg <- const-reg (mod 2^n)``.

    This is expanded explicitly instead of relying on ``Gate.control()``, which
    can hide an unsupported multi-controlled version of the internal ripple
    gates.  On the active branch, ``const-reg = ~reg + (const+1)``.
    """
    reg = list(reg_le); n = len(reg)
    if len(scratch) < max(0, n - 1):
        raise ValueError(f"controlled_const_minus_le needs >= {max(0,n-1)} scratch qubits")
    for q in reg:
        qc.cx(ctrl, q)
    addend = (int(const) + 1) & ((1 << n) - 1)
    for i in range(n):
        if (addend >> i) & 1:
            inc_mod2n_1ctrl(qc, ctrl, reg[i:], list(scratch[:max(0, n-i-1)]))


def xor_gt_const_little_endian(
    qc: QuantumCircuit,
    x_le: Sequence[Qubit],
    const: int,
    flag: Qubit,
    scratch: Sequence[Qubit],
) -> None:
    """Toggle flag iff little-endian x is strictly greater than a classical constant."""
    n = len(x_le)
    const %= 1 << n
    for pos in reversed(range(n)):
        if ((const >> pos) & 1) == 0:
            conditions = [(x_le[h], (const >> h) & 1) for h in range(pos + 1, n)]
            conditions.append((x_le[pos], 1))
            compute_control(qc, conditions, flag, scratch)


def xor_encoded_bit_length_big_endian(
    qc: QuantumCircuit,
    bits_be: Sequence[Qubit],
    target_len: Sequence[Qubit],
    flag: Qubit,
    scratch: Sequence[Qubit],
    ctrl: Optional[Qubit] = None,
) -> None:
    """XOR the paper-encoded bit length, i.e. bit_length(bits_be)-1.

    bits_be is ordered from most significant to least significant.  If the
    integer is zero, the encoded length is -1 mod 2^w, i.e. all ones.
    Repeating the routine uncomputes it because the writes are XORs.
    """
    n = len(bits_be)
    mask = (1 << len(target_len)) - 1

    # Nonzero cases: the first 1 from the left determines length.
    for length in range(1, n + 1):
        first_one = n - length
        conditions: list[tuple[Qubit, int]] = []
        if ctrl is not None:
            conditions.append((ctrl, 1))
        conditions.extend((bits_be[i], 0) for i in range(first_one))
        conditions.append((bits_be[first_one], 1))
        compute_control(qc, conditions, flag, scratch)
        xor_const_into_reg(qc, target_len, (length - 1) & mask, ctrl=flag)
        compute_control(qc, conditions, flag, scratch)

    # Zero case: all bits zero -> encoded length 0 is all ones.
    conditions = []
    if ctrl is not None:
        conditions.append((ctrl, 1))
    conditions.extend((q, 0) for q in bits_be)
    compute_control(qc, conditions, flag, scratch)
    xor_const_into_reg(qc, target_len, mask, ctrl=flag)
    compute_control(qc, conditions, flag, scratch)


def _set_big_endian_constant(qc: QuantumCircuit, reg_be: Sequence[Qubit], value: int) -> None:
    """Set a big-endian computational-basis constant into a zero register."""
    width = len(reg_be)
    for i, q in enumerate(reg_be):
        bit = (value >> (width - 1 - i)) & 1
        if bit:
            qc.x(q)


def build_modular_inversion_algorithm1_circuit(  # type: ignore[override]
    *,
    n: int,
    p: int,
    len_width: Optional[int] = None,
    shift_width: Optional[int] = None,
    T_max: Optional[int] = None,
) -> QuantumCircuit:
    """Full Algorithm-1 Qiskit circuit with Work2[3:3+n] as the input x.

    The circuit input convention is paper-style: Work2 is initialized by the
    caller as |000,x>; the output register starts at |0^n>.  No extra copy of x
    is allocated.
    """
    _require_qiskit()
    if not (0 < p < (1 << n)):
        raise ValueError("p must be a positive n-bit modulus, i.e. 0 < p < 2^n")
    if len_width is None:
        len_width = get_n_config(n)["len_width"]
    if shift_width is None:
        shift_width = get_n_config(n)["shift_width"]
    if T_max is None:
        T_max = Nmax_steps(n)

    step_regs = make_global_registers(n=n, len_width=len_width, shift_width=shift_width,
                                      T_max=T_max, include_algorithm1=True)
    Output = QuantumRegister(n, "out")
    qc = QuantumCircuit(*step_regs, Output, name="MODINV_ALG1_QISKIT_PAPER")

    (Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux) = step_regs
    flag = Ctrl[0]
    scratch = list(Aux)
    work_size = n + 3

    # Algorithm 1 initialization: Work1 = |100,p>, Work2 is supplied as |000,x>.
    qc.x(Work1[0])
    _set_big_endian_constant(qc, Work1[3: 3 + n], p)

    work2_rprime_le = list(reversed(list(Work2[3: 3 + n])))

    # Reversible preprocessing: if x > p/2, Iter ^= 1 and Work2.r' <- p-x.
    xor_gt_const_little_endian(qc, work2_rprime_le, p // 2, Iter[0], scratch)
    controlled_const_minus_le(qc, Iter[0], work2_rprime_le, p, scratch)

    # Encoded length-register initialization.  ell_t=1 is encoded as zero.
    enc_zero_len = (1 << len_width) - 1
    enc_zero_shift = (1 << shift_width) - 1
    xor_const_into_reg(qc, l_q, enc_zero_len)
    xor_const_into_reg(qc, l_s, enc_zero_shift)
    xor_encoded_bit_length_big_endian(qc, Work2[3: 3 + n], l_rp, flag, scratch)

    # Forward Algorithm-3 loop.
    steps = build_full_steps_circuit(n=n, len_width=len_width, shift_width=shift_width, T_max=T_max, aux_size=len(Aux)).to_gate(label="ALG3_STEPS")
    step_qubits = [q for reg in step_regs for q in reg]
    qc.append(steps, step_qubits)

    # Copy t' from the left part of Work2 to the output register.
    for bit in range(n):
        qc.cx(Work2[bit], Output[bit])

    # Algorithm 1 sign correction: if Iter=0 then Output <- p-Output.
    qc.x(Iter[0])
    controlled_const_minus_le(qc, Iter[0], list(Output), p, scratch)
    qc.x(Iter[0])

    # Reverse Algorithm-3 loop and uncompute all preprocessing/initialization.
    qc.append(steps.inverse(), step_qubits)

    xor_encoded_bit_length_big_endian(qc, Work2[3: 3 + n], l_rp, flag, scratch)
    xor_const_into_reg(qc, l_s, enc_zero_shift)
    xor_const_into_reg(qc, l_q, enc_zero_len)

    controlled_const_minus_le(qc, Iter[0], work2_rprime_le, p, scratch)
    xor_gt_const_little_endian(qc, work2_rprime_le, p // 2, Iter[0], scratch)

    _set_big_endian_constant(qc, Work1[3: 3 + n], p)
    qc.x(Work1[0])
    return qc


# Optimization: polytof tensor-decomposition optimization backend
# This section is intentionally independent from the paper-exact circuit builder
# above.  It exports the Qiskit circuit to polytof's RevKit-style .qc format,
# then optionally invokes polytof's pipeline:
#
#   .qc  --compile-->  phase tensor  --BCO-->  smaller tensor
#        --CPD/SGE/FGS--> CP rank = optimized Toffoli/CCZ count
#        --Waring/TODD--> optimized T-count, optional
#
# The output is a set of polytof artifacts and a parsed summary.
# Reconstructing the fully Qiskit circuit from
# a CP/Waring scheme is a separate synthesis step because polytof optimizes the
# non-Clifford phase polynomial up to surrounding Clifford/CNOT layers.

POLYTOF_PRIMITIVE_NAMES = {"x", "cx", "ccx"}


def _require_subprocess_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(f"Required executable {binary!r} was not found on PATH.")


def _qiskit_bit_index(qc: QuantumCircuit, q: Qubit) -> int:
    """Return the stable integer index of a Qiskit qubit in a circuit."""
    try:
        return qc.find_bit(q).index
    except Exception:
        return list(qc.qubits).index(q)


def iter_x_cx_ccx_operations(
    qc: QuantumCircuit,
    *,
    expand: Literal["recursive", "transpile"] = "transpile",
    optimization_level: int = 0,
) -> tuple[int, list[tuple[str, tuple[int, ...]]]]:
    """Flatten a Qiskit circuit to X/CX/CCX operations for polytof export.

    ``expand='transpile'`` is safest for Algorithm-1 circuits containing Qiskit
    controlled custom gates.  ``expand='recursive'`` is faster for the step
    circuit because all custom gates in this file are defined directly over
    X/CX/CCX.
    """
    _require_qiskit()
    if expand == "transpile":
        if transpile is None:
            raise RuntimeError("Qiskit transpile support is required for expand='transpile'.")
        qc_basis = transpile(qc, basis_gates=["x", "cx", "ccx"], optimization_level=optimization_level)
        ops: list[tuple[str, tuple[int, ...]]] = []
        for item in qc_basis.data:
            inst = item.operation
            qargs = item.qubits
            name = inst.name.lower()
            if name not in POLYTOF_PRIMITIVE_NAMES:
                raise ValueError(f"Transpiled circuit contains unsupported instruction {inst.name!r}.")
            ops.append((name, tuple(_qiskit_bit_index(qc_basis, q) for q in qargs)))
        return qc_basis.num_qubits, ops

    if expand != "recursive":
        raise ValueError("expand must be 'recursive' or 'transpile'")

    ops: list[tuple[str, tuple[int, ...]]] = []

    def walk(circ: QuantumCircuit, qubit_map: Sequence[int]) -> None:
        for item in circ.data:
            inst = item.operation
            qargs = item.qubits
            name = inst.name.lower()
            mapped = tuple(qubit_map[_qiskit_bit_index(circ, q)] for q in qargs)
            if name in POLYTOF_PRIMITIVE_NAMES:
                ops.append((name, mapped))
                continue
            if inst.definition is None:
                raise ValueError(f"Instruction {inst.name!r} has no definition and cannot be exported.")
            # Definition qubits are ordered exactly as the instruction's qargs.
            walk(inst.definition, mapped)

    walk(qc, list(range(qc.num_qubits)))
    return qc.num_qubits, ops


def write_polytof_qc(
    qc: QuantumCircuit,
    path: str | Path,
    *,
    expand: Literal["recursive", "transpile"] = "transpile",
    optimization_level: int = 0,
) -> dict:
    """Write ``qc`` in polytof's RevKit-style ``.qc`` format.

    Supported gate names in the emitted file are ``X``, ``cnot`` and ``tof``.
    Polytof's parser accepts this format and normalizes ``tof`` as Toffoli.
    """
    n_qubits, ops = iter_x_cx_ccx_operations(qc, expand=expand, optimization_level=optimization_level)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"q{i}" for i in range(n_qubits)]
    with path.open("w", encoding="utf-8") as f:
        f.write(".v " + " ".join(names) + "\n")
        f.write(".i " + " ".join(names) + "\n")
        f.write("BEGIN\n")
        for name, q in ops:
            if name == "x":
                f.write(f"X {names[q[0]]}\n")
            elif name == "cx":
                f.write(f"cnot {names[q[0]]} {names[q[1]]}\n")
            elif name == "ccx":
                f.write(f"tof {names[q[0]]} {names[q[1]]} {names[q[2]]}\n")
            else:  # defensive; should be unreachable
                raise ValueError(f"Unsupported primitive {name!r}")
        f.write("END\n")
    counts = Counter(name for name, _ in ops)
    return {
        "qc_path": str(path),
        "num_qubits": n_qubits,
        "x": int(counts.get("x", 0)),
        "cx": int(counts.get("cx", 0)),
        "ccx": int(counts.get("ccx", 0)),
        "total": int(sum(counts.values())),
    }


def polytof_vec_words(num_qubits: int) -> int:
    """Number of 64-bit words needed by polytof's bit-vector binaries."""
    return max(1, math.ceil(num_qubits / 64))


def _polytof_bin(polytof_root: str | Path, name: str) -> Path:
    suffix = ".exe" if shutil.which("cmd") and not Path(polytof_root, "bin", name).exists() else ""
    return Path(polytof_root) / "bin" / f"{name}{suffix}"


def _run_polytof_cmd(
    cmd: Sequence[str | Path],
    *,
    cwd: str | Path,
    timeout: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    cmd_str = [str(c) for c in cmd]
    if verbose:
        print("$", " ".join(cmd_str), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd_str, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    t1 = time.perf_counter()
    if verbose:
        if proc.stdout:
            print(proc.stdout, flush=True)
        if proc.stderr:
            print(proc.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"polytof command failed with code {proc.returncode}: {' '.join(cmd_str)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return {
        "cmd": cmd_str,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_s": t1 - t0,
    }


def ensure_polytof_binaries(
    polytof_root: str | Path,
    *,
    vec_words: int,
    build: bool = False,
    build_waring: bool = False,
) -> dict:
    """Check or build the polytof binaries needed by the pipeline."""
    # Resolve here because subprocess commands below run with cwd=root.
    # If polytof_root is a relative path such as .\polytof, passing
    # "polytof\bin\compile.exe" to g++ while cwd is already .\polytof
    # makes the linker try to create .\polytof\polytof\bin\compile.exe.
    # Absolute paths avoid that Windows/PowerShell path duplication.
    root = Path(polytof_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"polytof_root does not exist: {root}")
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    needed = {
        "compile": _polytof_bin(root, "compile"),
        "bco": _polytof_bin(root, "bco"),
        "topp": _polytof_bin(root, f"topp{vec_words}"),
    }
    if build_waring:
        needed["waring"] = _polytof_bin(root, f"waring{vec_words}")

    missing = [name for name, path in needed.items() if not path.exists()]
    if missing and not build:
        raise RuntimeError(
            "Missing polytof binaries: "
            + ", ".join(f"{name}={needed[name]}" for name in missing)
            + ". Re-run with --polytof_build or build them manually."
        )

    if build:
        _require_subprocess_binary("g++")
        compile_specs = [
            ("compile", ["g++", "-Ofast", "-std=c++20", "-march=native", "-s", "-pthread", "-I", "third_party", "-I", "src", "src/compile.cpp", "-o", str(needed["compile"])]),
            ("bco", ["g++", "-Ofast", "-std=c++20", "-march=native", "-s", "-pthread", "-I", "third_party", "-I", "src", "src/bco.cpp", "-o", str(needed["bco"])]),
            ("topp", ["g++", f"-D", f"VEC_WORDS={vec_words}", "-Ofast", "-std=c++20", "-march=native", "-s", "-pthread", "-I", "third_party", "-I", "src", "src/cpd.cpp", "-o", str(needed["topp"])]),
        ]
        if build_waring:
            compile_specs.append(("waring", ["g++", f"-D", f"VEC_WORDS={vec_words}", "-Ofast", "-std=c++20", "-march=native", "-s", "-pthread", "-I", "third_party", "-I", "src", "src/waring.cpp", "-o", str(needed["waring"])]))
        for name, cmd in compile_specs:
            if name in missing or not needed[name].exists():
                _run_polytof_cmd(cmd, cwd=root)

    return {name: str(path) for name, path in needed.items()}


def _parse_first_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _best_cpd_rank(polytof_root: str | Path, tensor_id: int) -> Optional[int]:
    cpd_dir = Path(polytof_root) / "data" / "cpd" / "topp"
    if not cpd_dir.exists():
        return None
    ranks = []
    for p in cpd_dir.glob(f"{tensor_id}-*.npy"):
        m = re.search(r"-(\d+)\.npy$", p.name)
        if m:
            ranks.append(int(m.group(1)))
    return min(ranks) if ranks else None


def _best_waring_count(polytof_root: str | Path, tensor_id: int) -> Optional[int]:
    war_dir = Path(polytof_root) / "data" / "waring"
    if not war_dir.exists():
        return None
    counts = []
    for p in war_dir.glob(f"{tensor_id}-*.npy"):
        m = re.search(r"-(\d+)\.npy$", p.name)
        if m:
            counts.append(int(m.group(1)))
    return min(counts) if counts else None


def run_polytof_optimization_on_circuit(
    qc: QuantumCircuit,
    *,
    polytof_root: str | Path,
    circuit_id: int,
    output_dir: str | Path = "outputs",
    stage: Literal["export", "compile", "bco", "cpd", "waring", "all"] = "all",
    export_expand: Literal["recursive", "transpile"] = "transpile",
    transpile_optimization_level: int = 0,
    build: bool = False,
    threads: int = 8,
    bco_beam: int = 1,
    bco_output_offset: int = 1000,
    sge: bool = True,
    fgs: bool = False,
    cpd_beam: int = 1,
    pool_size: int = 200,
    path_limit: int = 1_000_000,
    plus: bool = False,
    waring: bool = False,
    waring_num: int = 10,
    waring_beam: int = 3,
    timeout: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Export a circuit and optionally run the polytof optimization pipeline.

    ``stage='all'`` runs compile + BCO + CPD; add ``waring=True`` or use
    ``stage='waring'`` to also run the T-count optimizer.
    """
    # Use an absolute polytof root throughout the pipeline because all Polytof
    # binaries are executed with cwd=root.  This prevents relative paths like
    # .\polytof from being interpreted twice.
    root = Path(polytof_root).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root / "data" / "circuits" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    qc_path = raw_dir / f"{circuit_id:04d}.qc"
    if verbose:
        print(f"[polytof] exporting Qiskit circuit to {qc_path} using expand={export_expand!r} ...", flush=True)
    export_info = write_polytof_qc(qc, qc_path, expand=export_expand, optimization_level=transpile_optimization_level)
    export_info["local_copy"] = str(out_dir / f"{circuit_id:04d}.qc")
    shutil.copyfile(qc_path, export_info["local_copy"])
    if verbose:
        print(
            "[polytof] export done: "
            f"num_qubits={export_info.get('num_qubits')}, "
            f"x={export_info.get('x')}, cx={export_info.get('cx')}, ccx={export_info.get('ccx')}, "
            f"qc_path={export_info.get('qc_path')}",
            flush=True,
        )

    vec_words = polytof_vec_words(export_info["num_qubits"])
    result: dict[str, Any] = {
        "mode": "polytof",
        "stage": stage,
        "circuit_id": circuit_id,
        "polytof_root": str(root),
        "vec_words": vec_words,
        "export": export_info,
        "commands": [],
    }

    if stage == "export":
        return result

    do_waring = waring or stage == "waring"
    if verbose:
        print(f"[polytof] checking/building binaries for VEC_WORDS={vec_words} ...", flush=True)
    bins = ensure_polytof_binaries(root, vec_words=vec_words, build=build, build_waring=do_waring)
    result["binaries"] = bins

    compile_cmd = [bins["compile"], str(circuit_id), "--idx"]
    if verbose:
        print("[polytof] stage=compile starting ...", flush=True)
    compile_run = _run_polytof_cmd(compile_cmd, cwd=root, timeout=timeout, verbose=verbose)
    if verbose:
        print(f"[polytof] stage=compile done in {compile_run['elapsed_s']:.2f}s", flush=True)
    result["commands"].append(compile_run)
    result["compile"] = {
        "stdout": compile_run["stdout"],
        "phase_qubits": _parse_first_int(r"output:\s+(\d+)\s+qubits", compile_run["stdout"]),
        "t_count_after_merge": _parse_first_int(r"merge:\s+T=(\d+)", compile_run["stdout"]),
        "tensor_nnz": _parse_first_int(r"tensor:\s+(\d+)\s+nnz", compile_run["stdout"]),
    }
    # The compiler may introduce Hadamard-gadget ancillas before extracting the
    # phase tensor.  CPD/Waring binaries must be chosen for the tensor dimension,
    # not merely for the number of qubits in the exported .qc circuit.
    phase_qubits = result["compile"].get("phase_qubits") or export_info["num_qubits"]
    phase_vec_words = polytof_vec_words(int(phase_qubits))
    if phase_vec_words != vec_words:
        vec_words = phase_vec_words
        result["vec_words"] = vec_words
        bins = ensure_polytof_binaries(root, vec_words=vec_words, build=build, build_waring=do_waring)
        result["binaries"] = bins
    if stage == "compile":
        return result

    bco_id = circuit_id + bco_output_offset
    if stage in {"bco", "cpd", "waring", "all"}:
        bco_cmd = [bins["bco"], str(circuit_id), "-o", str(bco_id), "-b", str(bco_beam), "-t", str(threads), "--save", "--verify"]
        if verbose:
            print("[polytof] stage=bco starting ...", flush=True)
        bco_run = _run_polytof_cmd(bco_cmd, cwd=root, timeout=timeout, verbose=verbose)
        if verbose:
            print(f"[polytof] stage=bco done in {bco_run['elapsed_s']:.2f}s", flush=True)
        result["commands"].append(bco_run)
        result["bco"] = {
            "tensor_id": bco_id,
            "stdout": bco_run["stdout"],
            "nnz_initial": _parse_first_int(r"Initial:\s+(\d+)", bco_run["stdout"]),
            "nnz_final": _parse_first_int(r"Final:\s+(\d+)", bco_run["stdout"]),
            "tensor_path": str(root / "data" / "tensors" / f"{bco_id:04d}.npy"),
            "transform_path": str(root / "data" / "transform" / f"{circuit_id:04d}-{bco_id:04d}.npy"),
        }
    if stage == "bco":
        return result

    cpd_input_id = bco_id
    if stage in {"cpd", "waring", "all"}:
        cpd_cmd = [bins["topp"], str(cpd_input_id)]
        if sge:
            cpd_cmd += ["--sge", "-b", str(cpd_beam)]
        if fgs:
            cpd_cmd += ["--fgs", "-s", str(pool_size), "-f", str(path_limit)]
            if plus:
                cpd_cmd += ["--plus"]
        cpd_cmd += ["-t", str(threads), "--save", "--verify"]
        if verbose:
            print("[polytof] stage=cpd starting ...", flush=True)
        cpd_run = _run_polytof_cmd(cpd_cmd, cwd=root, timeout=timeout, verbose=verbose)
        if verbose:
            print(f"[polytof] stage=cpd done in {cpd_run['elapsed_s']:.2f}s", flush=True)
        result["commands"].append(cpd_run)
        result["cpd"] = {
            "tensor_id": cpd_input_id,
            "stdout": cpd_run["stdout"],
            "best_rank": _parse_first_int(r"Best rank:\s+(\d+)", cpd_run["stdout"]) or _best_cpd_rank(root, cpd_input_id),
            "cpd_dir": str(root / "data" / "cpd" / "topp"),
        }
    if stage == "cpd" or (stage == "all" and not do_waring):
        return result

    if do_waring:
        if "waring" not in bins:
            bins = ensure_polytof_binaries(root, vec_words=vec_words, build=build, build_waring=True)
            result["binaries"] = bins
        waring_cmd = [bins["waring"], str(cpd_input_id), "--cpd", "-n", str(waring_num), "-b", str(waring_beam), "-t", str(threads), "--save", "--verify"]
        if verbose:
            print("[polytof] stage=waring starting ...", flush=True)
        war_run = _run_polytof_cmd(waring_cmd, cwd=root, timeout=timeout, verbose=verbose)
        if verbose:
            print(f"[polytof] stage=waring done in {war_run['elapsed_s']:.2f}s", flush=True)
        result["commands"].append(war_run)
        result["waring"] = {
            "tensor_id": cpd_input_id,
            "stdout": war_run["stdout"],
            "best_t_count": _parse_first_int(r"min\s+(\d+)", war_run["stdout"]) or _best_waring_count(root, cpd_input_id),
            "waring_dir": str(root / "data" / "waring"),
        }
    return result


def run_polytof_for_n(
    n: int,
    *,
    polytof_root: str | Path,
    circuit_id: int = 9000,
    algorithm1: bool = False,
    p: Optional[int] = None,
    T_max: Optional[int] = None,
    output_dir: str | Path = "outputs",
    **kwargs: Any,
) -> dict:
    cfg = get_n_config(n)
    len_width = cfg["len_width"]
    shift_width = cfg["shift_width"]
    if T_max is None:
        T_max = cfg["T_max"]
    if algorithm1:
        if p is None:
            raise ValueError("algorithm1 polytof export requires p")
        qc = build_modular_inversion_algorithm1_circuit(n=n, p=p, len_width=len_width, shift_width=shift_width, T_max=T_max)
    else:
        qc = build_full_steps_circuit(n=n, len_width=len_width, shift_width=shift_width, T_max=T_max)
    result = run_polytof_optimization_on_circuit(
        qc,
        polytof_root=polytof_root,
        circuit_id=circuit_id,
        output_dir=output_dir,
        **kwargs,
    )
    result.update({
        "n": n,
        "len_width": len_width,
        "shift_width": shift_width,
        "T_max": T_max,
        "algorithm1": algorithm1,
        "p": p,
        "top_level_qubits_before_export": qc.num_qubits,
        "top_level_ops_before_export": len(qc.data),
    })
    return result




def clear_gate_construction_caches() -> None:
    """Clear cached step-dependent gate definitions and recursive count cache."""
    for fn_name in (
        "pre_shift_gate", "post_shift_gate", "phase_update_gate",
        "lc_swap_unary_gate", "lc_interval_addsub_unary_gate",
        "lc_prefix_addsub_unary_gate", "len_update_lt_unary_gate",
        "len_update_lrp_unary_gate", "swap_work_and_len_unary_shared_gate",
    ):
        fn = globals().get(fn_name)
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()
    _INST_COUNT_CACHE.clear()

# Recursive / formula-based resource counters
def _freeze_params(params: Sequence) -> tuple:
    out = []
    for p in params:
        if isinstance(p, (int, float, str, bool, type(None))):
            out.append(p)
        else:
            out.append(repr(p))
    return tuple(out)


def _inst_cache_key(inst: Instruction) -> tuple:
    # Include the definition identity.  Several step-dependent gates have the
    # same name and arity but different active windows/constants; caching only
    # by (name, arity, params) undercounts recursive resources.
    return (
        inst.name,
        inst.num_qubits,
        inst.num_clbits,
        _freeze_params(getattr(inst, "params", [])),
        id(getattr(inst, "definition", None)),
    )


def _iter_circuit_items(circ: QuantumCircuit):
    """Yield (operation, qubits, clbits) for Qiskit >=1.2 and fallback circuits."""
    for item in circ.data:
        if hasattr(item, "operation"):
            yield item.operation, tuple(item.qubits), tuple(item.clbits)
        else:
            yield item


def count_instruction_ops(inst: Instruction) -> Counter:
    if inst.name in PRIMITIVE_OPS:
        return Counter({inst.name: 1})
    if inst.name == "if_else" and hasattr(inst, "blocks"):
        total = Counter()
        for block in inst.blocks:
            for subinst, _qargs, _cargs in _iter_circuit_items(block):
                total += count_instruction_ops(subinst)
        return total
    key = _inst_cache_key(inst)
    if key in _INST_COUNT_CACHE:
        return _INST_COUNT_CACHE[key].copy()
    if inst.definition is None:
        raise ValueError(f"Instruction {inst.name!r} has no definition")
    total = Counter()
    for subinst, _qargs, _cargs in _iter_circuit_items(inst.definition):
        total += count_instruction_ops(subinst)
    _INST_COUNT_CACHE[key] = total.copy()
    return total


def count_circuit_ops_recursive(qc: QuantumCircuit) -> Counter:
    total = Counter()
    for inst, _qargs, _cargs in _iter_circuit_items(qc):
        total += count_instruction_ops(inst)
    return total


def count_pdf_formula_one_step(n: int, T: int) -> Counter:
    """Updated Section 5.2/Table count, using unary-iteration asymptotic block costs."""
    w = active_windows(n, T)
    k1, K1 = w["r_addsub"]
    k2, K2 = w["swap"]
    k3, K3 = w["t_addsub"]
    k4, K4 = w["len_update_lt"]
    k5, K5 = w["len_update_lrp"]

    d1 = max(0, K1 - k1)
    d2 = max(0, K2 - k2)
    d3 = max(0, K3 - k3)
    d4 = max(0, K4 - k4)
    d5 = max(0, K5 - k5)

    ccx = 14 * d1 + 2 * d2 + 12 * d3
    cx = 8 * d1 + 4 * d2 + 4 * d3
    if T % 4 == 0:
        ccx += 12 * d4 + 12 * d5
        cx += 16 * d4 + 16 * d5
    return Counter({"ccx": ccx, "cx": cx})


def count_pdf_formula_all_steps(n: int, T_max: Optional[int] = None) -> Counter:
    if T_max is None:
        T_max = Nmax_steps(n)
    total = Counter()
    for T in range(1, T_max + 1):
        total += count_pdf_formula_one_step(n, T)
    return total


def _count_full_steps_recursive_streaming(
    *,
    n: int,
    len_width: int,
    shift_width: int,
    T_max: int,
    aux_size: Optional[int] = None,
    T_start: int = 1,
    T_end: Optional[int] = None,
) -> tuple[Counter, int]:
    """Count the true recursively expanded Algorithm-3 circuit one step at a time.

    This is not the Section-5 paper formula.  Each step is built with
    ``append_one_step_T`` and recursively walked down to X/CX/CCX.  We stream the
    steps instead of materializing the entire 1476-step top-level circuit, which
    keeps n=256 counting tractable while producing the same raw recursive count
    as a full top-level circuit without cross-gate cancellation/optimization.
    """
    _require_qiskit()
    if T_end is None:
        T_end = T_max
    if T_start < 1 or T_end > T_max or T_start > T_end:
        raise ValueError(f"bad step range [{T_start}, {T_end}] for T_max={T_max}")
    regs = make_global_registers(n=n, len_width=len_width, shift_width=shift_width, T_max=T_max, aux_size=aux_size)
    (Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux) = regs
    total = Counter()
    num_qubits = sum(len(reg) for reg in regs)
    for T in range(T_start, T_end + 1):
        if MEASUREMENT_UNCOMPUTE:
            qc_step = QuantumCircuit(*regs, ClassicalRegister(1, "m"), name=f"COUNT_STEP_{T}")
        else:
            qc_step = QuantumCircuit(*regs, name=f"COUNT_STEP_{T}")
        append_one_step_T(
            qc_step,
            T=T,
            n=n,
            len_width=len_width,
            shift_width=shift_width,
            Phase1=Phase1,
            Phase2=Phase2,
            Iter=Iter,
            Sign=Sign,
            Ctrl=Ctrl,
            Work1=Work1,
            Work2=Work2,
            l_t=l_t,
            l_q=l_q,
            l_s=l_s,
            l_rp=l_rp,
            Aux=Aux,
        )
        total += count_circuit_ops_recursive(qc_step)
        if T % 300 == 0:
            clear_gate_construction_caches()
            gc.collect()
    return total, num_qubits


def _count_full_steps_recursive_subprocess_chunks(
    *,
    n: int,
    len_width: int,
    shift_width: int,
    T_max: int,
    chunk_size: int = 100,
    measurement_uncompute: bool = False,
) -> tuple[Counter, int]:
    """Count the same real streamed circuit in subprocess chunks.

    This is a memory/runtime management path only.  Each subprocess builds and
    recursively walks actual step circuits over the selected range; no paper
    formula or resource table is used.
    """
    script = Path(__file__).resolve()
    total = Counter()
    num_qubits: Optional[int] = None
    for start in range(1, T_max + 1, chunk_size):
        end = min(T_max, start + chunk_size - 1)
        cmd = [
            sys.executable, str(script),
            "--n", str(n),
            "--mode", "recursive",
            "--T_max", str(T_max),
            "--range_start", str(start),
            "--range_end", str(end),
        ]
        if measurement_uncompute:
            cmd.append("--measurement_uncompute")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        total += Counter(data["ops"])
        qn = int(data["num_qubits"])
        if num_qubits is None:
            num_qubits = qn
        elif num_qubits != qn:
            raise RuntimeError("inconsistent num_qubits across chunks")
    return total, int(num_qubits or 0)




def _count_full_steps_recursive_range(
    *,
    n: int,
    len_width: int,
    shift_width: int,
    T_max: int,
    T_start: int,
    T_end: int,
    aux_size: Optional[int] = None,
) -> tuple[Counter, int]:
    """Count true dynamically/reversibly expanded Algorithm-3 steps in [T_start,T_end]."""
    _require_qiskit()
    if T_start < 1 or T_end > T_max or T_start > T_end:
        raise ValueError(f"bad range [{T_start},{T_end}] for T_max={T_max}")
    regs = make_global_registers(n=n, len_width=len_width, shift_width=shift_width, T_max=T_max, aux_size=aux_size)
    (Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux) = regs
    total = Counter()
    num_qubits = sum(len(reg) for reg in regs)
    for T in range(T_start, T_end + 1):
        qc_step = QuantumCircuit(*regs, ClassicalRegister(1, "m"), name=f"COUNT_STEP_{T}") if MEASUREMENT_UNCOMPUTE else QuantumCircuit(*regs, name=f"COUNT_STEP_{T}")
        append_one_step_T(
            qc_step, T=T, n=n, len_width=len_width, shift_width=shift_width,
            Phase1=Phase1, Phase2=Phase2, Iter=Iter, Sign=Sign, Ctrl=Ctrl,
            Work1=Work1, Work2=Work2, l_t=l_t, l_q=l_q, l_s=l_s, l_rp=l_rp, Aux=Aux,
        )
        total += count_circuit_ops_recursive(qc_step)
        if (T - T_start + 1) % 100 == 0:
            clear_gate_construction_caches()
            gc.collect()
    return total, num_qubits

def count_full_circuit_ops(
    *,
    n: int,
    len_width: int,
    shift_width: int,
    T_max: Optional[int] = None,
    save_qasm: bool = False,
    qasm_path: Optional[str] = None,
    recursive: bool = True,
    measurement_uncompute: bool = False,
) -> dict:
    if T_max is None:
        T_max = Nmax_steps(n)

    t0 = time.perf_counter()
    streamed_recursive = False
    old_mb = MEASUREMENT_UNCOMPUTE
    set_measurement_uncompute(measurement_uncompute)
    try:
        if recursive and not save_qasm:
            if T_max >= 512:
                ops, num_qubits = _count_full_steps_recursive_subprocess_chunks(
                    n=n, len_width=len_width, shift_width=shift_width, T_max=T_max,
                    measurement_uncompute=measurement_uncompute,
                )
            else:
                ops, num_qubits = _count_full_steps_recursive_streaming(
                    n=n, len_width=len_width, shift_width=shift_width, T_max=T_max
                )
            streamed_recursive = True
        else:
            qc_full = build_full_steps_circuit(n=n, len_width=len_width, shift_width=shift_width, T_max=T_max)
            num_qubits = qc_full.num_qubits
            if save_qasm:
                if qasm2_dumps is None:
                    raise RuntimeError("Qiskit qasm2 support is required to save QASM.")
                if qasm_path is None:
                    qasm_path = f"n{n}_modinv_unary.qasm"
                with open(qasm_path, "w", encoding="utf-8") as f:
                    f.write(qasm2_dumps(qc_full))
            if recursive:
                ops = count_circuit_ops_recursive(qc_full)
            else:
                if transpile is None:
                    raise RuntimeError("Qiskit is required for transpile mode. Install with: pip install qiskit")
                basis = ["ccx", "cx", "x", "h", "z", "cz", "measure", "reset"] if measurement_uncompute else ["ccx", "cx", "x"]
                qc_basis = transpile(qc_full, basis_gates=basis, optimization_level=0)
                ops = Counter(qc_basis.count_ops())
    finally:
        set_measurement_uncompute(old_mb)
    t1 = time.perf_counter()

    return {
        "n": n,
        "len_width": len_width,
        "shift_width": shift_width,
        "T_max": T_max,
        "num_qubits": num_qubits,
        "ccx": int(ops.get("ccx", 0)),
        "cx": int(ops.get("cx", 0)),
        "x": int(ops.get("x", 0)),
        "h": int(ops.get("h", 0)),
        "measure": int(ops.get("measure", 0)),
        "reset": int(ops.get("reset", 0)),
        "z": int(ops.get("z", 0)),
        "cz": int(ops.get("cz", 0)),
        "total": int(sum(ops.values())),
        "elapsed_s": t1 - t0,
        "streamed_recursive_actual_circuit": streamed_recursive,
        "measurement_uncompute": bool(measurement_uncompute),
    }


def get_n_config(n: int) -> dict:
    if n in N_CONFIG:
        return N_CONFIG[n].copy()
    T_max = Nmax_steps(n)
    width = max(paper_len_width(n), terminal_safe_shift_width(n, T_max))
    return {"len_width": width, "shift_width": width, "T_max": T_max}


def run_for_n(
    n: int,
    *,
    mode: Literal["formula", "recursive", "transpile"] = "formula",
    T_max: Optional[int] = None,
    output_dir: str = "outputs",
    save_qasm: bool = False,
    measurement_uncompute: bool = False,
) -> dict:
    cfg = get_n_config(n)
    len_width = cfg["len_width"]
    shift_width = cfg["shift_width"]
    if T_max is None:
        T_max = cfg["T_max"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if mode == "formula":
        t0 = time.perf_counter()
        ops = count_pdf_formula_all_steps(n, T_max=T_max)
        t1 = time.perf_counter()
        return {
            "mode": mode,
            "n": n,
            "len_width": len_width,
            "shift_width": shift_width,
            "T_max": T_max,
            "ccx": int(ops.get("ccx", 0)),
            "cx": int(ops.get("cx", 0)),
            "elapsed_s": t1 - t0,
        }

    qasm_path = str(Path(output_dir) / f"n{n}_modinv_unary.qasm") if save_qasm else None
    return {"mode": mode, **count_full_circuit_ops(
        n=n, len_width=len_width, shift_width=shift_width, T_max=T_max, save_qasm=save_qasm,
        qasm_path=qasm_path, recursive=(mode == "recursive"), measurement_uncompute=measurement_uncompute,
    )}


# Preserve the generic implementations for resource-development experiments,
# but route the public API through the repaired S835 FASTDUAL scheduler.  The
# import is deliberately lazy: ``eea_circuit_s835_fastdual`` imports this module
# for primitive gates, so importing it here at module load time would create a
# cycle.
_build_full_steps_circuit_generic = build_full_steps_circuit
_build_modular_inversion_algorithm1_circuit_generic = build_modular_inversion_algorithm1_circuit

def build_full_steps_circuit(
    n: int,
    len_width: int,
    shift_width: int,
    T_max: Optional[int] = None,
    aux_size: Optional[int] = None,
) -> QuantumCircuit:
    from eea_circuit_s835_fastdual import build_full_steps_circuit as _fixed_builder
    return _fixed_builder(
        n=n, len_width=len_width, shift_width=shift_width,
        T_max=T_max, aux_size=aux_size,
    )

def build_modular_inversion_algorithm1_circuit(
    *,
    n: int,
    p: int,
    len_width: Optional[int] = None,
    shift_width: Optional[int] = None,
    T_max: Optional[int] = None,
) -> QuantumCircuit:
    from eea_circuit_s835_fastdual import build_modular_inversion_algorithm1_circuit as _fixed_builder
    return _fixed_builder(
        n=n, p=p, len_width=len_width, shift_width=shift_width, T_max=T_max,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Updated unary-iteration modular inversion circuit")
    parser.add_argument("--n", type=int, required=True, help="problem size n")
    parser.add_argument("--mode", choices=["formula", "recursive", "transpile", "polytof"], default="formula",
                        help="formula reproduces PDF Section 5.2; recursive/transpile count actual Qiskit gates; polytof exports/runs tensor-decomposition optimization")
    parser.add_argument("--T_max", type=int, default=None, help="override number of Algorithm-3 steps")
    parser.add_argument("--range_start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--range_end", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--save_qasm", action="store_true")
    parser.add_argument("--measurement-uncompute", "--measurement_uncompute", dest="measurement_uncompute", action="store_true",
                        help="build/count a real dynamic circuit where temporary unary-iteration ANDs are uncomputed by H-measure-reset plus feed-forward CZ")
    parser.add_argument("--algorithm1", action="store_true",
                        help="build the full Algorithm-1 wrapper; requires --p")
    parser.add_argument("--p", type=int, default=None,
                        help="prime modulus used by --algorithm1")

    # Optional polytof backend.  This requires a local checkout of
    # https://github.com/ZIB-IOL/polytof and, unless --polytof_stage=export,
    # compiled C++ binaries under <polytof_root>/bin.
    parser.add_argument("--polytof_root", type=str, default=None,
                        help="path to a local polytof repository checkout")
    parser.add_argument("--polytof_id", type=int, default=9000,
                        help="numeric circuit/tensor id used inside polytof/data")
    parser.add_argument("--polytof_stage", choices=["export", "compile", "bco", "cpd", "waring", "all"], default="all",
                        help="how far to run the polytof pipeline")
    parser.add_argument("--polytof_export_expand", choices=["recursive", "transpile"], default="transpile",
                        help="how to flatten the Qiskit circuit before writing .qc")
    parser.add_argument("--polytof_build", action="store_true",
                        help="compile missing polytof binaries with g++")
    parser.add_argument("--polytof_threads", type=int, default=8)
    parser.add_argument("--polytof_timeout", type=int, default=None,
                        help="timeout in seconds for each external polytof command")
    parser.add_argument("--polytof_bco_beam", type=int, default=1)
    parser.add_argument("--polytof_cpd_beam", type=int, default=1)
    parser.add_argument("--polytof_no_sge", action="store_true",
                        help="disable SGE preprocessing in CPD")
    parser.add_argument("--polytof_fgs", action="store_true",
                        help="enable flip graph search after SGE")
    parser.add_argument("--polytof_pool_size", type=int, default=200)
    parser.add_argument("--polytof_path_limit", type=int, default=1000000)
    parser.add_argument("--polytof_plus", action="store_true")
    parser.add_argument("--polytof_waring", action="store_true",
                        help="after CPD, also run Waring/FastTODD T-count optimization")
    parser.add_argument("--polytof_waring_num", type=int, default=10)
    parser.add_argument("--polytof_waring_beam", type=int, default=3)
    parser.add_argument("--polytof_json", action="store_true",
                        help="print the full polytof summary as JSON")
    args = parser.parse_args()

    if args.range_start is not None or args.range_end is not None:
        if args.range_start is None or args.range_end is None:
            raise ValueError("both --range_start and --range_end are required")
        cfg = get_n_config(args.n)
        len_width = cfg["len_width"]
        shift_width = cfg["shift_width"]
        T_max = args.T_max if args.T_max is not None else cfg["T_max"]
        old_mb = MEASUREMENT_UNCOMPUTE
        set_measurement_uncompute(args.measurement_uncompute)
        try:
            ops, num_qubits = _count_full_steps_recursive_streaming(
                n=args.n, len_width=len_width, shift_width=shift_width, T_max=T_max,
                T_start=args.range_start, T_end=args.range_end,
            )
        finally:
            set_measurement_uncompute(old_mb)
        print(json.dumps({
            "n": args.n,
            "len_width": len_width,
            "shift_width": shift_width,
            "T_max": T_max,
            "range": [args.range_start, args.range_end],
            "num_qubits": num_qubits,
            "ops": {k: int(v) for k, v in ops.items()},
            "measurement_uncompute": bool(args.measurement_uncompute),
        }))
        return

    if args.mode == "polytof":
        if args.polytof_root is None:
            raise ValueError("--mode polytof requires --polytof_root pointing to a local polytof checkout")
        if args.algorithm1 and args.p is None:
            raise ValueError("--algorithm1 requires --p")
        result = run_polytof_for_n(
            args.n,
            polytof_root=args.polytof_root,
            circuit_id=args.polytof_id,
            algorithm1=args.algorithm1,
            p=args.p,
            T_max=args.T_max,
            output_dir=args.outdir,
            stage=args.polytof_stage,
            export_expand=args.polytof_export_expand,
            build=args.polytof_build,
            threads=args.polytof_threads,
            bco_beam=args.polytof_bco_beam,
            sge=not args.polytof_no_sge,
            fgs=args.polytof_fgs,
            cpd_beam=args.polytof_cpd_beam,
            pool_size=args.polytof_pool_size,
            path_limit=args.polytof_path_limit,
            plus=args.polytof_plus,
            waring=args.polytof_waring,
            waring_num=args.polytof_waring_num,
            waring_beam=args.polytof_waring_beam,
            timeout=args.polytof_timeout,
        )
        if args.polytof_json:
            print(json.dumps(result, indent=2))
        else:
            compact = {
                "mode": "polytof",
                "n": result.get("n"),
                "T_max": result.get("T_max"),
                "algorithm1": result.get("algorithm1"),
                "qc_path": result.get("export", {}).get("qc_path"),
                "num_qubits": result.get("export", {}).get("num_qubits"),
                "export_ccx": result.get("export", {}).get("ccx"),
                "compile_tensor_nnz": result.get("compile", {}).get("tensor_nnz"),
                "bco_nnz_final": result.get("bco", {}).get("nnz_final"),
                "cpd_best_rank": result.get("cpd", {}).get("best_rank"),
                "waring_best_t_count": result.get("waring", {}).get("best_t_count"),
                "polytof_root": result.get("polytof_root"),
                "vec_words": result.get("vec_words"),
            }
            print(compact)
        return

    if args.algorithm1:
        if args.p is None:
            raise ValueError("--algorithm1 requires --p")
        cfg = get_n_config(args.n)
        len_width = cfg["len_width"]
        shift_width = cfg["shift_width"]
        T_max = args.T_max if args.T_max is not None else cfg["T_max"]
        circuit = build_modular_inversion_algorithm1_circuit(
            n=args.n, p=args.p, len_width=len_width, shift_width=shift_width, T_max=T_max
        )
        print({
            "mode": "algorithm1",
            "n": args.n,
            "p": args.p,
            "len_width": len_width,
            "shift_width": shift_width,
            "T_max": T_max,
            "num_qubits": circuit.num_qubits,
            "num_top_level_ops": len(circuit.data),
        })
        if args.save_qasm:
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            qasm_path = Path(args.outdir) / f"n{args.n}_p{args.p}_modinv_algorithm1.qasm"
            with open(qasm_path, "w", encoding="utf-8") as f:
                f.write(qasm2_dumps(circuit))
            print({"qasm": str(qasm_path)})
        return

    result = run_for_n(args.n, mode=args.mode, T_max=args.T_max, output_dir=args.outdir, save_qasm=args.save_qasm, measurement_uncompute=args.measurement_uncompute)
    print(result)


if __name__ == "__main__":
    main()
