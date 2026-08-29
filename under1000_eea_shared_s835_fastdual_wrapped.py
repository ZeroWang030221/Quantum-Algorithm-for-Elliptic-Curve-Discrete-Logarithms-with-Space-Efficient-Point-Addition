from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Qubit

import eea_circuit_s835_fastdual as eea
from quadratic_lazy_instruction import LazyDefinedInstruction
from quadratic_gidney_arithmetic import (
    append_gidney_compare_ge_const,
    append_gidney_add_const_mod2n,
)
from under1000_modular_arithmetic_base import _append_cx_multi

SECP256K1_P = (1 << 256) - (1 << 32) - 977


def eea_microstep_count(p: int, x: int) -> int:
    """Return the exact four-phase Algorithm-3 microstep count.

    Algorithm 1 runs the EEA on min(x, p-x).  An EEA quotient q contributes
    four phases of ``bit_length(q)`` microsteps.
    """
    p = int(p); x = int(x)
    if not (0 < x < p):
        raise ValueError("need 0 < x < p")
    r0, r1 = p, min(x, p - x)
    weighted = 0
    while r1:
        q, r2 = divmod(r0, r1)
        weighted += int(q).bit_length()
        r0, r1 = r1, r2
    return 4 * weighted


@lru_cache(maxsize=None)
def certified_uniform_steps_for_modulus(p: int, *, exhaustive_bit_limit: int = 16) -> int:
    """A deterministic schedule for a concrete modulus.

    For small regression moduli we exhaust all nonzero inputs, which gives 56
    for p=419.  For cryptographic moduli we retain the paper's published
    size-only schedule; callers may still provide an independently certified
    ``T_max`` explicitly.
    """
    p = int(p)
    if p <= 2:
        return 0
    n = p.bit_length()
    paper_steps = int(eea.Nmax_steps(n))
    if n <= int(exhaustive_bit_limit):
        return max(paper_steps, max(eea_microstep_count(p, x) for x in range(1, p)))
    return paper_steps


@dataclass(frozen=True)
class SharedEEALayout:
    n: int
    len_width: int
    shift_width: int
    T_max: int
    work2_tail: int = 3
    work1_tail: int = 3
    persistent_controls: int = 4  # Phase1, Phase2, Iter, Sign. Ctrl lives in Aux[0].
    step_aux: int = 20            # includes temporary Ctrl in Aux[0].

    @property
    def length_registers(self) -> int:
        return 3 * self.len_width + self.shift_width

    @property
    def s_qubits(self) -> int:
        return self.work2_tail + self.work1_tail + self.persistent_controls + self.length_registers + self.step_aux

    @property
    def forward_gate_qubits(self) -> int:
        return 2 * self.n + self.s_qubits

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "controls": self.persistent_controls,
            "length_registers": self.length_registers,
            "s_qubits": self.s_qubits,
            "forward_gate_qubits": self.forward_gate_qubits,
            "point_addition_quantum_qubits_with_control": 1 + 3 * self.n + self.s_qubits,
        })
        return d


def shared_eea_layout(n: int, *, p: int = SECP256K1_P, T_max: Optional[int] = None) -> SharedEEALayout:
    cfg = eea.get_n_config(n)
    lw = int(cfg["len_width"])
    certified_steps = (
        certified_uniform_steps_for_modulus(int(p))
        if int(p).bit_length() == int(n)
        else int(cfg["T_max"])
    )
    if T_max is None:
        T_max = certified_steps
    else:
        T_max = max(int(T_max), int(certified_steps))
    # Retain the paper-width low shift word whenever one borrowed terminal
    # epoch bit suffices.  The epoch is Aux[1] during the fixed-horizon padding
    # and is reversibly folded back into l_s at the public EEA boundary.
    sw = max(int(cfg["shift_width"]), int(eea.terminal_safe_shift_width(n, T_max)))
    aux = int(eea.qiskit_paper_aux_size(n, lw, sw, T_max, include_algorithm1=False))
    return SharedEEALayout(n=int(n), len_width=lw, shift_width=sw, T_max=int(T_max), step_aux=aux)


def shared_eea_s_qubits(n: int, *, p: int = SECP256K1_P, T_max: Optional[int] = None) -> int:
    return shared_eea_layout(n, p=p, T_max=T_max).s_qubits


def split_shared_s(S: Sequence[Qubit], n: int, *, p: int = SECP256K1_P, T_max: Optional[int] = None) -> dict[str, list[Qubit]]:
    layout = shared_eea_layout(n, p=p, T_max=T_max)
    S = list(S)
    if len(S) < layout.s_qubits:
        raise ValueError(f"shared S too small: need {layout.s_qubits}, got {len(S)}")
    off = 0; out: dict[str, list[Qubit]] = {}
    out["work2_tail"] = S[off:off+3]; off += 3
    out["work1_tail"] = S[off:off+3]; off += 3
    for name in ["Phase1", "Phase2", "Iter", "Sign"]:
        out[name] = S[off:off+1]; off += 1
    lw = layout.len_width; sw = layout.shift_width
    out["l_t"] = S[off:off+lw]; off += lw
    out["l_q"] = S[off:off+lw]; off += lw
    out["l_s"] = S[off:off+sw]; off += sw
    out["l_rp"] = S[off:off+lw]; off += lw
    out["Aux"] = S[off:off+layout.step_aux]; off += layout.step_aux
    out["unused"] = S[off:]
    return out


def _apply_source_to_target_permutation(qc: QuantumCircuit, wires: Sequence[Qubit], target_of_source: Sequence[int]) -> None:
    wires = list(wires); m = len(wires)
    if sorted(target_of_source) != list(range(m)):
        raise ValueError("target_of_source is not a permutation")
    desired_at_pos = [None] * m
    for source, target in enumerate(target_of_source):
        desired_at_pos[target] = source
    current_at_pos = list(range(m))
    for pos in range(m):
        want = desired_at_pos[pos]
        if current_at_pos[pos] == want:
            continue
        j = current_at_pos.index(want)
        qc.swap(wires[pos], wires[j])
        current_at_pos[pos], current_at_pos[j] = current_at_pos[j], current_at_pos[pos]


def _work2_layout_permutation(n: int) -> list[int]:
    target = [0] * (n + 3)
    for i in range(n):
        target[i] = 3 + (n - 1 - i)
    target[n] = 0; target[n + 1] = 1; target[n + 2] = 2
    return target


def _prepare_work2_from_little_endian_x(qc: QuantumCircuit, X: Sequence[Qubit], tail: Sequence[Qubit]) -> list[Qubit]:
    """Permute little-endian X into the paper's Work2=[000,x]_big_endian layout."""
    X = list(X); tail = list(tail); n = len(X)
    work2 = X + tail
    _apply_source_to_target_permutation(qc, work2, _work2_layout_permutation(n))
    return work2


def _restore_little_endian_x_from_work2(qc: QuantumCircuit, X: Sequence[Qubit], tail: Sequence[Qubit]) -> list[Qubit]:
    """Apply the inverse of ``_prepare_work2_from_little_endian_x``."""
    X = list(X); tail = list(tail); n = len(X)
    work2 = X + tail
    forward = _work2_layout_permutation(n)
    inverse = [0] * len(forward)
    for source, target in enumerate(forward):
        inverse[target] = source
    _apply_source_to_target_permutation(qc, work2, inverse)
    return work2


def _set_big_endian_constant(qc: QuantumCircuit, reg_be: Sequence[Qubit], value: int) -> None:
    width = len(reg_be)
    for i, q in enumerate(reg_be):
        if (int(value) >> (width - 1 - i)) & 1:
            qc.x(q)


def _toggle_work1_constant(qc: QuantumCircuit, Work1: Sequence[Qubit], p: int) -> None:
    Work1 = list(Work1)
    qc.x(Work1[0])
    _set_big_endian_constant(qc, Work1[3:], p)


def _toggle_terminal_work1_constant(qc: QuantumCircuit, Work1: Sequence[Qubit], p: int, n: int) -> None:
    """XOR the input-independent terminal Work1=(t=p,r=1) state.

    At termination ``ell_t=n``.  Work1 therefore stores the n little-endian
    bits of p, the appended zero lane, and the two-bit big-endian value 01.
    Applying this involution clears/restores the large n-qubit workspace.
    """
    Work1 = list(Work1)
    if len(Work1) != n + 3:
        raise ValueError("bad Work1 width")
    for i in range(n):
        if (int(p) >> i) & 1:
            qc.x(Work1[i])
    qc.x(Work1[n + 2])


def _apply_controlled_source_to_target_permutation(
    qc: QuantumCircuit, ctrl: Qubit, wires: Sequence[Qubit], target_of_source: Sequence[int]
) -> None:
    """Controlled counterpart of ``_apply_source_to_target_permutation``."""
    wires = list(wires); m = len(wires)
    if sorted(target_of_source) != list(range(m)):
        raise ValueError("target_of_source is not a permutation")
    desired_at_pos = [None] * m
    for source, target in enumerate(target_of_source):
        desired_at_pos[target] = source
    current_at_pos = list(range(m))
    for pos in range(m):
        want = desired_at_pos[pos]
        if current_at_pos[pos] == want:
            continue
        j = current_at_pos.index(want)
        eea.cswap_toffoli(qc, ctrl, wires[pos], wires[j])
        current_at_pos[pos], current_at_pos[j] = current_at_pos[j], current_at_pos[pos]


def _canonical_rotate_work2(
    qc: QuantumCircuit, Work2: Sequence[Qubit], l_s: Sequence[Qubit],
    scratch: Sequence[Qubit], *, shift_epoch: Optional[Qubit] = None,
    inverse: bool = False,
) -> None:
    """Canonicalize/decanonicalize the circular Work2 representation."""
    Work2=list(Work2);counter=list(l_s);scratch=list(scratch)
    if shift_epoch is not None:
        qc.x(shift_epoch);counter.append(shift_epoch)
    if len(scratch)<len(counter)+1:
        raise ValueError("canonical Work2 rotation needs counter_width+1 clean scratch qubits")
    cs=scratch[:len(counter)+1]
    eea.add_const_mod_2n(qc,counter,1,cs)
    bit_order=range(len(counter)-1,-1,-1) if inverse else range(len(counter))
    m=len(Work2)
    for b in bit_order:
        d=(1<<b)%m
        if d==0:continue
        target=[((i-d)%m if inverse else (i+d)%m) for i in range(m)]
        _apply_controlled_source_to_target_permutation(qc,counter[b],Work2,target)
    eea.sub_const_mod_2n(qc,counter,1,cs)
    if shift_epoch is not None: qc.x(shift_epoch)


def _xor_const_into_reg(qc: QuantumCircuit, reg: Sequence[Qubit], value: int) -> None:
    mask = (1 << len(reg)) - 1
    value &= mask
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            qc.x(q)


def _append_controlled_const_minus_mod2n_gidney(
    qc: QuantumCircuit,
    ctrl: Qubit,
    reg_le: Sequence[Qubit],
    const: int,
    dirty: Sequence[Qubit],
    clean3: Sequence[Qubit],
    cbits: Sequence,
) -> None:
    """If ctrl=1, reg <- const - reg (mod 2^n), restoring dirty/clean.

    For ctrl=1, bitwise complement + add 1 + add const gives const-reg modulo
    2^n. For ctrl=0 every suboperation is identity. Constant additions use the
    Gidney measurement-vented backend, so this wrapper stays linear in n.
    """
    reg = list(reg_le)
    for q in reg:
        qc.cx(ctrl, q)
    append_gidney_add_const_mod2n(qc, reg, 1, dirty[:max(0, len(reg)-1)], clean3, cbits, ctrl=ctrl)
    append_gidney_add_const_mod2n(qc, reg, const, dirty[:max(0, len(reg)-1)], clean3, cbits, ctrl=ctrl)


def _xor_encoded_bit_length_big_endian(qc: QuantumCircuit, bits_be: Sequence[Qubit], target_len: Sequence[Qubit], flag: Qubit) -> None:
    """XOR len(bits_be)-1 into target_len under the paper's encoded-length convention.

    This is a real reversible circuit. It scans for the first one in the
    big-endian string. The all-zero case writes encoded zero length, i.e. all
    ones. The temporary flag is restored after each case.
    """
    bits = list(bits_be); n = len(bits); mask = (1 << len(target_len)) - 1
    for first_one in range(n):
        encoded = (n - first_one - 1) & mask
        for j in range(first_one):
            qc.x(bits[j])
        _append_cx_multi(qc, bits[:first_one+1], flag)
        for i, q in enumerate(target_len):
            if (encoded >> i) & 1:
                qc.cx(flag, q)
        _append_cx_multi(qc, bits[:first_one+1], flag)
        for j in reversed(range(first_one)):
            qc.x(bits[j])
    # zero input -> encoded length 0 is -1 = all ones
    for q in bits:
        qc.x(q)
    _append_cx_multi(qc, bits, flag)
    for q in target_len:
        qc.cx(flag, q)
    _append_cx_multi(qc, bits, flag)
    for q in reversed(bits):
        qc.x(q)


@lru_cache(maxsize=None)
def _algorithm3_step_fastdual_gate(
    n: int, len_width: int, shift_width: int, T_max: int, aux_size: int, T: int,
    *, inverse: bool = False,
) -> Instruction:
    """One latest-paper Algorithm-3 step or its explicit dynamic inverse.

    Both directions retain the paper's measurement-assisted unary-AND
    uncomputation.  The inverse is constructed block-by-block by
    ``append_one_step_T_inverse``; it is neither a renamed forward definition
    nor a coherent ``q.inverse()`` surrogate.  Consequently the circuit used by
    Figure 15 and the circuit used by the resource counter are identical.
    """
    work_size = n + 3
    num_qubits = 4 + 2 * work_size + 3 * len_width + shift_width + aux_size

    def _builder() -> QuantumCircuit:
        old_mode = bool(eea.MEASUREMENT_UNCOMPUTE)
        eea.set_measurement_uncompute(True)
        try:
            Phase1 = QuantumRegister(1, "Phase1")
            Phase2 = QuantumRegister(1, "Phase2")
            Iter = QuantumRegister(1, "Iter")
            Sign = QuantumRegister(1, "Sign")
            Work1 = QuantumRegister(work_size, "Work1")
            Work2 = QuantumRegister(work_size, "Work2")
            l_t = QuantumRegister(len_width, "l_t")
            l_q = QuantumRegister(len_width, "l_q")
            l_s = QuantumRegister(shift_width, "l_s")
            l_rp = QuantumRegister(len_width, "l_rp")
            Aux = QuantumRegister(aux_size, "Aux")
            m_step = ClassicalRegister(1, "m_step")
            q = QuantumCircuit(
                Phase1, Phase2, Iter, Sign, Work1, Work2,
                l_t, l_q, l_s, l_rp, Aux, m_step,
                name=f"ALG3_STEP_FASTDUAL_WRAPPED_DEF_T{T}_{n}{'_INV' if inverse else ''}",
            )
            kwargs = dict(
                T=T, n=n, len_width=len_width, shift_width=shift_width,
                Phase1=Phase1, Phase2=Phase2, Iter=Iter, Sign=Sign,
                Work1=Work1, Work2=Work2, l_t=l_t, l_q=l_q, l_s=l_s,
                l_rp=l_rp, Aux=Aux, T_max=T_max,
            )
            if inverse:
                eea.append_one_step_T_inverse(q, **kwargs)
            else:
                eea.append_one_step_T(q, **kwargs)
            return q
        finally:
            eea.set_measurement_uncompute(old_mode)

    suffix = "_INV" if inverse else ""
    return LazyDefinedInstruction(
        f"ALG3_STEP_FASTDUAL_WRAPPED_T{T}_{n}{suffix}",
        num_qubits, 1, _builder,
    )


@lru_cache(maxsize=None)
def forward_eea_shared_definition(n: int, p: int = SECP256K1_P, T_max: Optional[int] = None) -> QuantumCircuit:
    layout = shared_eea_layout(n, p=p, T_max=T_max)
    X = QuantumRegister(n, "X_le")
    A = QuantumRegister(n, "A_large_workspace")
    S = QuantumRegister(layout.s_qubits, "S_shared")
    m = ClassicalRegister(max(1, n), "m_eea_wrapper")
    qc = QuantumCircuit(X, A, S, m, name=f"EEA_SHARED_ALG3_FASTDUAL_WRAPPED_DEF_{n}")
    parts = split_shared_s(S, n, p=p, T_max=T_max)

    Work2 = _prepare_work2_from_little_endian_x(qc, X, parts["work2_tail"])
    Work1 = parts["work1_tail"] + list(A)
    _toggle_work1_constant(qc, Work1, p)

    work2_rprime_le = list(reversed(Work2[3:3+n]))
    dirty = Work1[:n]
    clean3 = [parts["Aux"][2], parts["Aux"][3], parts["Aux"][4]]
    cbits = list(m)[:n]

    # Algorithm 1 preprocessing: if x > p/2, set Iter and run EEA on p-x.
    append_gidney_compare_ge_const(qc, work2_rprime_le, p // 2 + 1, parts["Iter"][0], dirty, clean3, cbits)
    _append_controlled_const_minus_mod2n_gidney(qc, parts["Iter"][0], work2_rprime_le, p, dirty, clean3, cbits)

    lw = layout.len_width; sw = layout.shift_width
    _xor_const_into_reg(qc, parts["l_q"], (1 << lw) - 1)       # ell_q = 0 encoded as -1
    _xor_const_into_reg(qc, parts["l_s"], (1 << sw) - 1)       # ell_s = 0 encoded as -1
    _xor_const_into_reg(qc, parts["l_t"], 0)                   # ell_t = 1 encoded as 0
    _xor_encoded_bit_length_big_endian(qc, Work2[3:3+n], parts["l_rp"], parts["Aux"][0])

    step_qubits = [parts["Phase1"][0], parts["Phase2"][0], parts["Iter"][0], parts["Sign"][0],
                  *Work1, *Work2, *parts["l_t"], *parts["l_q"], *parts["l_s"], *parts["l_rp"], *parts["Aux"]]
    for T in range(1, layout.T_max + 1):
        qc.append(_algorithm3_step_fastdual_gate(n, lw, sw, layout.T_max, layout.step_aux, T), step_qubits, [m[0]])

    # Terminal padding rotates Work2 while preserving l_s.  Canonicalize the
    # circular representation before exposing t' on the X wires.
    _canonical_rotate_work2(qc, Work2, parts["l_s"], parts["Aux"][2:], shift_epoch=parts["Aux"][1])
    qc.x(parts["Aux"][0])
    eea.compress_terminal_shift_epoch(qc, shift_epoch=parts["Aux"][1], l_s=parts["l_s"], outer_ctrl=parts["Aux"][0], scratch=parts["Aux"][2:])
    qc.x(parts["Aux"][0])

    # Algorithm 1 postprocessing: if Iter=0, convert t' to the positive inverse p-t'.
    qc.x(parts["Iter"][0])
    _append_controlled_const_minus_mod2n_gidney(qc, parts["Iter"][0], Work2[:n], p, dirty, clean3, cbits)
    qc.x(parts["Iter"][0])

    # The remaining Work1 state is the input-independent terminal (p,1) state.
    # Clear it exactly, including the n-qubit large workspace A.
    _toggle_terminal_work1_constant(qc, Work1, p, n)
    return qc


@lru_cache(maxsize=None)
def inverse_eea_shared_definition(n: int, p: int = SECP256K1_P, T_max: Optional[int] = None) -> QuantumCircuit:
    """Literal executable reverse of ``forward_eea_shared_definition``.

    This is not a forward circuit with an inverse-looking label.  It reverses
    Algorithm 1 in the exact opposite order, appending the explicit
    measurement-assisted inverse of every Algorithm-3 step and then undoing
    all wrapper preparation.
    """
    layout = shared_eea_layout(n, p=p, T_max=T_max)
    X = QuantumRegister(n, "X_le")
    A = QuantumRegister(n, "A_large_workspace")
    S = QuantumRegister(layout.s_qubits, "S_shared")
    m = ClassicalRegister(max(1, n), "m_eea_wrapper")
    qc = QuantumCircuit(X, A, S, m, name=f"EEA_SHARED_ALG3_FASTDUAL_WRAPPED_DEF_{n}_dg")
    parts = split_shared_s(S, n, p=p, T_max=T_max)

    Work2 = list(X) + list(parts["work2_tail"])
    Work1 = list(parts["work1_tail"]) + list(A)
    dirty = Work1[:n]
    clean3 = [parts["Aux"][2], parts["Aux"][3], parts["Aux"][4]]
    cbits = list(m)[:n]

    # Forward execution cleared the fixed terminal Work1=(p,1) state last.
    _toggle_terminal_work1_constant(qc, Work1, p, n)

    # Undo Algorithm-1 postprocessing (an involution controlled on Iter=0).
    qc.x(parts["Iter"][0])
    _append_controlled_const_minus_mod2n_gidney(qc, parts["Iter"][0], Work2[:n], p, dirty, clean3, cbits)
    qc.x(parts["Iter"][0])

    # Restore the circularly shifted terminal representation expected by the
    # reverse Algorithm-3 schedule.
    qc.x(parts["Aux"][0])
    eea.compress_terminal_shift_epoch(qc, shift_epoch=parts["Aux"][1], l_s=parts["l_s"], outer_ctrl=parts["Aux"][0], scratch=parts["Aux"][2:])
    qc.x(parts["Aux"][0])
    _canonical_rotate_work2(qc, Work2, parts["l_s"], parts["Aux"][2:], shift_epoch=parts["Aux"][1], inverse=True)

    lw = layout.len_width; sw = layout.shift_width
    step_qubits = [parts["Phase1"][0], parts["Phase2"][0], parts["Iter"][0], parts["Sign"][0],
                  *Work1, *Work2, *parts["l_t"], *parts["l_q"], *parts["l_s"], *parts["l_rp"], *parts["Aux"]]
    for T in range(layout.T_max, 0, -1):
        qc.append(_algorithm3_step_fastdual_gate(n, lw, sw, layout.T_max, layout.step_aux, T, inverse=True), step_qubits, [m[0]])

    # Undo the length initialization while Work2 still contains the preprocessed
    # EEA divisor, then undo p-x and the Iter comparison.
    _xor_encoded_bit_length_big_endian(qc, Work2[3:3+n], parts["l_rp"], parts["Aux"][0])
    _xor_const_into_reg(qc, parts["l_q"], (1 << lw) - 1)
    _xor_const_into_reg(qc, parts["l_s"], (1 << sw) - 1)

    work2_rprime_le = list(reversed(Work2[3:3+n]))
    _append_controlled_const_minus_mod2n_gidney(qc, parts["Iter"][0], work2_rprime_le, p, dirty, clean3, cbits)
    append_gidney_compare_ge_const(qc, work2_rprime_le, p // 2 + 1, parts["Iter"][0], dirty, clean3, cbits)

    _toggle_work1_constant(qc, Work1, p)
    _restore_little_endian_x_from_work2(qc, X, parts["work2_tail"])
    return qc


def eea_forward_shared_instruction(n: int, p: int = SECP256K1_P, *, T_max: Optional[int] = None, lazy_definition: bool = True) -> Instruction:
    layout = shared_eea_layout(n, p=p, T_max=T_max)
    builder = lambda: forward_eea_shared_definition(n, p, T_max)
    if lazy_definition:
        return LazyDefinedInstruction(f"EEA_FORWARD_SHARED_ALG3_FASTDUAL_WRAPPED_{n}", 2*n + layout.s_qubits, max(1, n), builder)
    q = builder()
    return q.to_instruction(label=f"EEA_FORWARD_SHARED_ALG3_FASTDUAL_WRAPPED_{n}")


def eea_inverse_shared_instruction(n: int, p: int = SECP256K1_P, *, T_max: Optional[int] = None, lazy_definition: bool = True) -> Instruction:
    layout = shared_eea_layout(n, p=p, T_max=T_max)
    builder = lambda: inverse_eea_shared_definition(n, p, T_max)
    if lazy_definition:
        return LazyDefinedInstruction(f"EEA_INVERSE_SHARED_ALG3_FASTDUAL_WRAPPED_{n}", 2*n + layout.s_qubits, max(1, n), builder)
    q = builder()
    return q.to_instruction(label=f"EEA_INVERSE_SHARED_ALG3_FASTDUAL_WRAPPED_{n}")


# Compatibility aliases used by some older scripts.
eea_forward_shared_gate = eea_forward_shared_instruction
eea_inverse_shared_gate = eea_inverse_shared_instruction


def width_report(n: int = 256, *, T_max: Optional[int] = None) -> dict:
    layout = shared_eea_layout(n, p=SECP256K1_P, T_max=T_max)
    return {"shared_eea_layout": layout.as_dict(), "point_addition_quantum_qubits": 1 + 3*n + layout.s_qubits}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    print(json.dumps(width_report(args.n), indent=2, sort_keys=True))
