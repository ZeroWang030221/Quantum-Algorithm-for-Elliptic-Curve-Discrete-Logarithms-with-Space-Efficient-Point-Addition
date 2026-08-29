from functools import lru_cache
import math
from typing import Literal, Optional, Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit

import eea_circuit_updated as _e

C_EEA = _e.C_EEA
N_CONFIG = _e.N_CONFIG
paper_len_width = _e.paper_len_width
paper_shift_width = _e.paper_shift_width
terminal_safe_shift_width = _e.terminal_safe_shift_width
Nmax_steps = _e.Nmax_steps
active_windows = _e.active_windows
get_n_config = getattr(_e, "get_n_config")
count_circuit_ops_recursive = getattr(_e, "count_circuit_ops_recursive", None)

def clear_gate_construction_caches() -> None:
    """Clear all measurement-mode-dependent helper definitions."""
    _e.clear_gate_construction_caches()
    for name in (
        "lc_swap_unary_gate", "lc_interval_addsub_unary_gate",
        "lc_prefix_addsub_unary_gate", "lc_prefix_addsub_prepared_boundary_gate",
        "swap_work_and_len_unary_shared_gate",
        "swap_work_and_len_unary_shared_inverse_gate",
    ):
        fn = globals().get(name)
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()
    low = globals().get("_low")
    if low is not None:
        for name in (
            "lc_swap_unary_gate", "lc_interval_addsub_unary_gate",
            "lc_prefix_addsub_unary_gate", "len_update_lt_unary_gate",
            "len_update_lrp_unary_gate", "swap_work_and_len_unary_shared_gate",
        ):
            clear = getattr(getattr(low, name, None), "cache_clear", None)
            if callable(clear):
                clear()

def set_measurement_uncompute(enabled: bool) -> None:
    changed = bool(enabled) != bool(_e.MEASUREMENT_UNCOMPUTE)
    _e.set_measurement_uncompute(enabled)
    if changed:
        clear_gate_construction_caches()


def __getattr__(name: str):
    return getattr(_e, name)


def _tight_unary_depth_for_labels(labels: Sequence[int]) -> int:
    labels = sorted(set(labels))
    if len(labels) <= 1:
        return 0
    bit = _e._split_bit(labels)
    z = [x for x in labels if ((x >> bit) & 1) == 0]
    o = [x for x in labels if ((x >> bit) & 1) == 1]
    return 1 + max(_tight_unary_depth_for_labels(z), _tight_unary_depth_for_labels(o))


def unary_iteration_tight(qc: QuantumCircuit, *, index_reg: Sequence[Qubit], labels: Sequence[int],
                          ctrl: Qubit, ancillas: Sequence[Qubit], leaf_fn, order: Literal["inc", "dec"] = "inc") -> None:
    labels = sorted(set(labels))
    if not labels:
        return
    need = _tight_unary_depth_for_labels(labels)
    if len(ancillas) < need:
        raise ValueError(f"tight unary iteration needs {need} ancillas, got {len(ancillas)}")
    def rec(sub_labels, g, depth):
        if len(sub_labels) == 1:
            leaf_fn(sub_labels[0], g); return
        b = _e._split_bit(sub_labels)
        z = [x for x in sub_labels if ((x >> b) & 1) == 0]
        o = [x for x in sub_labels if ((x >> b) & 1) == 1]
        h = ancillas[depth]
        _e._and_with_index_bit(qc, g, index_reg[b], h, 0)
        if order == "inc":
            rec(z, h, depth+1)
            qc.cx(g, h)
            rec(o, h, depth+1)
            qc.cx(g, h)
        else:
            qc.cx(g, h)
            rec(o, h, depth+1)
            qc.cx(g, h)
            rec(z, h, depth+1)
        _e._uncompute_and_with_index_bit(qc, g, index_reg[b], h, 0)
    rec(labels, ctrl, 0)


def dual_unary_iteration_tight(qc: QuantumCircuit, *, index_a: Sequence[Qubit], index_b: Sequence[Qubit], labels: Sequence[int],
                               ctrl_a: Qubit, ctrl_b: Qubit, ancillas_a: Sequence[Qubit], ancillas_b: Sequence[Qubit],
                               leaf_fn, order: Literal["inc", "dec"] = "inc") -> None:
    labels = sorted(set(labels))
    if not labels:
        return
    need = _tight_unary_depth_for_labels(labels)
    if len(ancillas_a) < need or len(ancillas_b) < need:
        raise ValueError(f"tight dual unary iteration needs {need} ancillas per endpoint")
    def rec(sub_labels, ga, gb, depth):
        if len(sub_labels) == 1:
            leaf_fn(sub_labels[0], ga, gb); return
        bit = _e._split_bit(sub_labels)
        z = [x for x in sub_labels if ((x >> bit) & 1) == 0]
        o = [x for x in sub_labels if ((x >> bit) & 1) == 1]
        ha = ancillas_a[depth]; hb = ancillas_b[depth]
        _e._and_with_index_bit(qc, ga, index_a[bit], ha, 0)
        _e._and_with_index_bit(qc, gb, index_b[bit], hb, 0)
        if order == "inc":
            rec(z, ha, hb, depth+1)
            qc.cx(ga, ha); qc.cx(gb, hb)
            rec(o, ha, hb, depth+1)
            qc.cx(gb, hb); qc.cx(ga, ha)
        else:
            qc.cx(ga, ha); qc.cx(gb, hb)
            rec(o, ha, hb, depth+1)
            qc.cx(gb, hb); qc.cx(ga, ha)
            rec(z, ha, hb, depth+1)
        _e._uncompute_and_with_index_bit(qc, gb, index_b[bit], hb, 0)
        _e._uncompute_and_with_index_bit(qc, ga, index_a[bit], ha, 0)
    rec(labels, ctrl_a, ctrl_b, 0)


def _toggle_eq_const_under_ctrl_direct(qc: QuantumCircuit, *, endpoint: Sequence[Qubit], const: int, ctrl: Qubit, acc: Qubit, scratch: Sequence[Qubit]) -> None:
    # scratch supplies a temporary eq flag followed by mcx scratch.
    eq = scratch[0]
    pool = list(scratch[1:])
    _e.compute_eq_const(qc, endpoint, const, eq, pool)
    qc.ccx(ctrl, eq, acc)
    _e.compute_eq_const(qc, endpoint, const, eq, pool)


def _const_scratch(Scratch, width: int, carry: Qubit) -> list[Qubit]:
    # add_const_mod_2n expects width constant bits followed by one clean carry.
    return list(Scratch[:width]) + [carry]


def _dirty_c3x(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, target: Qubit, dirty: Qubit) -> None:
    """Exact C^3X using one dirty ancilla and four Toffolis.

    The dirty qubit is restored even when it initially contains an unknown value.
    """
    qc.ccx(a, b, dirty)
    qc.ccx(c, dirty, target)
    qc.ccx(a, b, dirty)
    qc.ccx(c, dirty, target)


def _controlled_toffoli_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, target: Qubit, dirty: Qubit) -> None:
    _dirty_c3x(qc, ctrl, a, b, target, dirty)


def controlled_maj_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    """Dirty-ancilla realization of paper Figure 11(a)."""
    qc.cx(c, a)
    qc.cx(c, b)
    _controlled_toffoli_dirty(qc, ctrl, a, b, c, dirty)


def controlled_uma_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    """Dirty-ancilla realization of paper Figure 11(b)."""
    _controlled_toffoli_dirty(qc, ctrl, a, b, c, dirty)
    qc.ccx(ctrl, b, a)
    qc.cx(c, b)
    qc.cx(c, a)


def controlled_maj_inv_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    _controlled_toffoli_dirty(qc, ctrl, a, b, c, dirty)
    qc.cx(c, b)
    qc.cx(c, a)


def controlled_uma_inv_dirty(qc: QuantumCircuit, ctrl: Qubit, a: Qubit, b: Qubit, c: Qubit, dirty: Qubit) -> None:
    qc.cx(c, a)
    qc.cx(c, b)
    qc.ccx(ctrl, b, a)
    _controlled_toffoli_dirty(qc, ctrl, a, b, c, dirty)


def _apply_cell_dirty(qc: QuantumCircuit, mode: Literal["add", "sub"], pass_kind: Literal["first", "second"],
                      ctrl: Qubit, addend: Qubit, target: Qubit, carry: Qubit, dirty: Qubit) -> None:
    if mode == "add" and pass_kind == "first":
        controlled_maj_dirty(qc, ctrl, target, addend, carry, dirty)
    elif mode == "add" and pass_kind == "second":
        controlled_uma_dirty(qc, ctrl, target, addend, carry, dirty)
    elif mode == "sub" and pass_kind == "first":
        controlled_uma_inv_dirty(qc, ctrl, target, addend, carry, dirty)
    elif mode == "sub" and pass_kind == "second":
        controlled_maj_inv_dirty(qc, ctrl, target, addend, carry, dirty)
    else:
        raise ValueError("bad arithmetic cell mode/pass")


def _apply_cell_inverse(
    qc: QuantumCircuit, mode: Literal["add", "sub"],
    pass_kind: Literal["first", "second"], ctrl: Qubit, addend: Qubit,
    target: Qubit, carry: Qubit, pool: Sequence[Qubit],
) -> None:
    """Literal inverse of one Figure-11 cell using the same paper controls."""
    opposite = "sub" if mode == "add" else "add"
    reverse_pass = "second" if pass_kind == "first" else "first"
    _e._apply_cell(qc, opposite, reverse_pass, ctrl, addend, target, carry, pool)


@lru_cache(maxsize=None)
def lc_swap_unary_gate(*, k: int, K: int, len_width: int, name: str = "LC_SWAP_S835_FAST") -> Gate:
    """Paper Figure 9 / Algorithm 3 merged quotient-bit swap.

    Both the Phase-2 insertion and the Phase-3 removal act on the same
    one-based Work1 position

        J = ell_t + ell_q + 1.

    The phase-dependent difference is only the subsequent update of ell_q.
    With the paper's truth-minus-one length encoding, temporarily replacing
    l_q by l_t + l_q + 3 computes exactly J for both branches.
    """
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = _e.unary_depth(M)
    base = max(len_width, depth)
    scratch_size = base + 1
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Sign, Work1, l_t, l_q, Scratch, name=name)
    carry = Scratch[base]
    cs = list(Scratch[:len_width]) + [carry]
    qc.append(_e.cuccaro_add_mod_2n_no_z_gate(len_width, name="ADD_lt_to_lq"), list(l_t) + list(l_q) + [carry])
    _e.add_const_mod_2n(qc, l_q, 3, cs)
    path = list(Scratch[:depth])

    def leaf(j: int, ej: Qubit) -> None:
        _e.cswap_toffoli(qc, ej, Sign[0], Work1[j - k])

    unary_iteration_tight(
        qc,
        index_reg=l_q,
        labels=list(range(k, K + 1)),
        ctrl=Ctrl[0],
        ancillas=path,
        leaf_fn=leaf,
        order="inc",
    )
    _e.sub_const_mod_2n(qc, l_q, 3, cs)
    qc.append(_e.cuccaro_sub_mod_2n_no_z_gate(len_width, name="SUB_lt_from_lq"), list(l_t) + list(l_q) + [carry])
    return _e._finalize_block(qc)


@lru_cache(maxsize=None)
def lc_interval_addsub_unary_gate(*, n: int, k: int, K: int, len_width: int, shift_width: int,
                                  mode: Literal["add", "sub"], sign_update: bool,
                                  target: Literal["work1", "work2"], name: str,
                                  inverse: bool = False) -> Gate:
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    endpoint_width = max(len_width, shift_width)
    # If the interval has one more label than a power of two (the n=256 worst case),
    # handle the top label separately and run the unary scans over the remaining power-of-two interval.
    labels_all_abs = list(range(k, K + 1))
    rel_count = len(labels_all_abs)
    # A [0,2^d] interval has one more leaf than a depth-d binary tree.
    # Handle the top label explicitly and use the pruned depth-d tree for
    # labels 0,...,2^d-1.  The leaf-zero controls are masked by the true
    # top bit below, so endpoint 2^d never aliases endpoint 0.
    labels_main = list(range(rel_count))
    top_special = False
    if rel_count > 1 and ((rel_count - 1) & (rel_count - 2)) == 0:
        labels_main = list(range(rel_count - 1))
        top_special = True
    top_rel = rel_count - 1
    depth = _tight_unary_depth_for_labels(labels_main)
    # Layout note:
    #   anc_a/anc_b occupy the first 2*depth wires and are used only by
    #   the unary endpoint scans.  Endpoint affine transforms need
    #   endpoint_width scratch wires plus a carry.  For late steps the unary
    #   depth can be smaller than endpoint_width; placing carry immediately
    #   after the unary paths would then alias it with the constant-adder
    #   scratch.  We therefore place carry/acc/cell_pool after the larger of
    #   the unary-scratch region and the endpoint-transform scratch region.
    base = max(2 * depth, endpoint_width)
    scratch_size = base + 3
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_q = QuantumRegister(len_width, "l_q")
    l_s = QuantumRegister(shift_width, "l_s")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Sign, Work1, Work2, l_t, l_q, l_s, Scratch, name=name)
    anc_a = list(Scratch[:depth])
    anc_b = list(Scratch[depth:2*depth])
    carry = Scratch[base]
    acc = Scratch[base + 1]
    cell_pool = [Scratch[base + 2]]
    # Top-special equality controls need a clean v-chain scratch pool.  At the
    # moment they are used, all wires before 'base' are clean.
    eq_scratch = [Scratch[base + 2]] + list(Scratch[:base])
    cs = _const_scratch(Scratch, endpoint_width, carry)
    # Prepare L=(ell_t-1)+(ell_q-1)+4 and R=n+2-(ell_s-1).
    qc.append(_e.cuccaro_add_mod_2n_no_z_gate(len_width, name="ADD_lt_to_lq"), list(l_t) + list(l_q) + [carry])
    _e.add_const_mod_2n(qc, l_q, 4, cs[:len_width] + [carry])
    _e.const_minus_inplace(qc, l_s, n + 2, cs[:shift_width] + [carry])
    # Convert absolute endpoints to relative offsets in [0, K-k].
    _e.sub_const_mod_2n(qc, l_q, k, cs[:len_width] + [carry])
    _e.sub_const_mod_2n(qc, l_s, k, cs[:shift_width] + [carry])
    def qpair(j: int) -> tuple[Qubit, Qubit]:
        j_abs = k + j
        idx = j_abs - k
        if target == "work1":
            return Work2[idx], Work1[idx]
        if target == "work2":
            return Work1[idx], Work2[idx]
        raise ValueError("bad target")
    def toggle_endpoint_leaf(endpoint: Sequence[Qubit], j: int, ej: Qubit) -> None:
        # In the pruned [0,2^d-1] tree the low d bits of endpoint 2^d
        # equal those of endpoint 0.  Mask leaf zero with NOT(top_bit), while
        # the explicit equality operation handles endpoint 2^d.
        if top_special and j == 0:
            top_bit = top_rel.bit_length() - 1
            qc.x(endpoint[top_bit])
            qc.ccx(ej, endpoint[top_bit], acc)
            qc.x(endpoint[top_bit])
        else:
            qc.cx(ej, acc)

    def apply_cell(which: Literal["first", "second"], j: int, inv: bool) -> None:
        addend, tgt = qpair(j)
        if inv:
            _apply_cell_inverse(qc, mode, which, acc, addend, tgt, carry, cell_pool)
        else:
            _e._apply_cell(qc, mode, which, acc, addend, tgt, carry, cell_pool)

    if not inverse:
        def leaf_first(j: int, rj: Qubit, lj: Qubit) -> None:
            toggle_endpoint_leaf(l_s, j, rj)
            apply_cell("first", j, False)
            toggle_endpoint_leaf(l_q, j, lj)

        if top_special:
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_s, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
            apply_cell("first", top_rel, False)
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_q, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
        dual_unary_iteration_tight(
            qc, index_a=l_s, index_b=l_q, labels=labels_main,
            ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
            ancillas_b=anc_b, leaf_fn=leaf_first, order="dec",
        )
        if sign_update:
            qc.cx(carry, Sign[0])

        def leaf_second(j: int, rj: Qubit, lj: Qubit) -> None:
            toggle_endpoint_leaf(l_q, j, lj)
            apply_cell("second", j, False)
            toggle_endpoint_leaf(l_s, j, rj)

        dual_unary_iteration_tight(
            qc, index_a=l_s, index_b=l_q, labels=labels_main,
            ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
            ancillas_b=anc_b, leaf_fn=leaf_second, order="inc",
        )
        if top_special:
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_q, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
            apply_cell("second", top_rel, False)
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_s, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
    else:
        # Reverse the second (low-to-high) pass, then the carry write, then the
        # first (high-to-low) pass.  Each unary tree is recomputed and
        # measurement-uncomputed in the same way as the forward paper block.
        if top_special:
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_s, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
            apply_cell("second", top_rel, True)
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_q, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)

        def leaf_second_inv(j: int, rj: Qubit, lj: Qubit) -> None:
            toggle_endpoint_leaf(l_s, j, rj)
            apply_cell("second", j, True)
            toggle_endpoint_leaf(l_q, j, lj)

        dual_unary_iteration_tight(
            qc, index_a=l_s, index_b=l_q, labels=labels_main,
            ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
            ancillas_b=anc_b, leaf_fn=leaf_second_inv, order="dec",
        )
        if sign_update:
            qc.cx(carry, Sign[0])

        def leaf_first_inv(j: int, rj: Qubit, lj: Qubit) -> None:
            toggle_endpoint_leaf(l_q, j, lj)
            apply_cell("first", j, True)
            toggle_endpoint_leaf(l_s, j, rj)

        dual_unary_iteration_tight(
            qc, index_a=l_s, index_b=l_q, labels=labels_main,
            ctrl_a=Ctrl[0], ctrl_b=Ctrl[0], ancillas_a=anc_a,
            ancillas_b=anc_b, leaf_fn=leaf_first_inv, order="inc",
        )
        if top_special:
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_q, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
            apply_cell("first", top_rel, True)
            _toggle_eq_const_under_ctrl_direct(qc, endpoint=l_s, const=top_rel, ctrl=Ctrl[0], acc=acc, scratch=eq_scratch)
    _e.add_const_mod_2n(qc, l_s, k, cs[:shift_width] + [carry])
    _e.add_const_mod_2n(qc, l_q, k, cs[:len_width] + [carry])
    _e.const_minus_inplace(qc, l_s, n + 2, cs[:shift_width] + [carry])
    _e.sub_const_mod_2n(qc, l_q, 4, cs[:len_width] + [carry])
    qc.append(_e.cuccaro_sub_mod_2n_no_z_gate(len_width, name="SUB_lt_from_lq"), list(l_t) + list(l_q) + [carry])
    return _e._finalize_block(qc)


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
    masked_sign_update: bool = False,
) -> Gate:
    """Fast-dual t-side prefix arithmetic with the Phase-4 high lane.

    When ``masked_sign_update`` is true, an additional clean ``SignMask`` input
    is present and the selected physical ripple carry updates ``Sign`` only
    under that mask.  This is required for the packed Work2 representation:
    the carry of the restoring addition is a logical negative flag only when
    every still-live upper-tail bit of ``t'`` is zero.
    """
    if k > K:
        raise ValueError("need k <= K")
    M = K - k + 1
    depth = _e.unary_depth(M)
    base = max(depth, len_width)
    scratch_size = base + 3
    Ctrl = QuantumRegister(1, "Ctrl")
    Extend = QuantumRegister(1, "Extend")
    Sign = QuantumRegister(1, "Sign")
    SignMask = QuantumRegister(1, "SignMask") if masked_sign_update else None
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    regs = [Ctrl, Extend, Sign]
    if SignMask is not None:
        regs.append(SignMask)
    regs.extend([Work1, Work2, l_t, Scratch])
    qc = _e._block_circuit(*regs, name=name)
    path = list(Scratch[:depth])
    carry = Scratch[base]
    acc = Scratch[base + 1]
    cell_pool = [Scratch[base + 2]]
    cs = list(Scratch[:len_width]) + [carry]
    _e.add_const_mod_2n(qc, l_t, 2, cs)
    _e.inc_mod2n_1ctrl(qc, Extend[0], list(l_t), list(Scratch[:max(0, len_width - 1)]))

    def qpair(j: int) -> tuple[Qubit, Qubit]:
        idx = j - k
        if target == "work1":
            return Work2[idx], Work1[idx]
        if target == "work2":
            return Work1[idx], Work2[idx]
        raise ValueError("bad target")

    qc.cx(Ctrl[0], acc)

    def leaf_first(j: int, ej: Qubit) -> None:
        addend, tgt = qpair(j)
        _e._apply_cell(qc, mode, "first", acc, addend, tgt, carry, cell_pool)
        qc.cx(ej, acc)

    unary_iteration_tight(
        qc,
        index_reg=l_t,
        labels=list(range(k, K + 1)),
        ctrl=Ctrl[0],
        ancillas=path,
        leaf_fn=leaf_first,
        order="inc",
    )
    if sign_update:
        if SignMask is None:
            qc.cx(carry, Sign[0])
        else:
            qc.ccx(carry, SignMask[0], Sign[0])

    def leaf_second(j: int, ej: Qubit) -> None:
        addend, tgt = qpair(j)
        qc.cx(ej, acc)
        _e._apply_cell(qc, mode, "second", acc, addend, tgt, carry, cell_pool)

    unary_iteration_tight(
        qc,
        index_reg=l_t,
        labels=list(range(k, K + 1)),
        ctrl=Ctrl[0],
        ancillas=path,
        leaf_fn=leaf_second,
        order="dec",
    )
    qc.cx(Ctrl[0], acc)
    _e.dec_mod2n_1ctrl(qc, Extend[0], list(l_t), list(Scratch[:max(0, len_width - 1)]))
    _e.sub_const_mod_2n(qc, l_t, 2, cs)
    return _e._finalize_block(qc)



@lru_cache(maxsize=None)
def lc_prefix_addsub_prepared_boundary_gate(
    *, k: int, K: int, len_width: int, mode: Literal["add", "sub"],
    sign_update: bool, target: Literal["work1", "work2"], name: str,
    inverse: bool = False,
) -> Gate:
    """T-side prefix arithmetic with an already prepared absolute endpoint."""
    if k > K: raise ValueError("need k <= K")
    M=K-k+1; depth=_e.unary_depth(M); base=max(depth,len_width)
    Ctrl=QuantumRegister(1,"Ctrl"); Sign=QuantumRegister(1,"Sign")
    Work1=QuantumRegister(M,"Work1"); Work2=QuantumRegister(M,"Work2")
    Boundary=QuantumRegister(len_width,"Boundary"); Scratch=QuantumRegister(base+3,"Scratch")
    qc=_e._block_circuit(Ctrl,Sign,Work1,Work2,Boundary,Scratch,name=name)
    path=list(Scratch[:depth]); carry=Scratch[base]; acc=Scratch[base+1]; pool=[Scratch[base+2]]
    def pair(j):
        i=j-k
        return (Work2[i],Work1[i]) if target=="work1" else (Work1[i],Work2[i])
    def apply_cell(which: Literal["first", "second"], j: int, inv: bool) -> None:
        a, b = pair(j)
        if inv:
            _apply_cell_inverse(qc, mode, which, acc, a, b, carry, pool)
        else:
            _e._apply_cell(qc, mode, which, acc, a, b, carry, pool)

    labels = list(range(k, K + 1))
    if not inverse:
        qc.cx(Ctrl[0], acc)
        def first(j, ej):
            apply_cell("first", j, False)
            qc.cx(ej, acc)
        unary_iteration_tight(qc, index_reg=Boundary, labels=labels, ctrl=Ctrl[0], ancillas=path, leaf_fn=first, order="inc")
        if sign_update:
            qc.cx(carry, Sign[0])
        def second(j, ej):
            qc.cx(ej, acc)
            apply_cell("second", j, False)
        unary_iteration_tight(qc, index_reg=Boundary, labels=labels, ctrl=Ctrl[0], ancillas=path, leaf_fn=second, order="dec")
        qc.cx(Ctrl[0], acc)
    else:
        qc.cx(Ctrl[0], acc)
        def second_inv(j, ej):
            apply_cell("second", j, True)
            qc.cx(ej, acc)
        unary_iteration_tight(qc, index_reg=Boundary, labels=labels, ctrl=Ctrl[0], ancillas=path, leaf_fn=second_inv, order="inc")
        if sign_update:
            qc.cx(carry, Sign[0])
        def first_inv(j, ej):
            qc.cx(ej, acc)
            apply_cell("first", j, True)
        unary_iteration_tight(qc, index_reg=Boundary, labels=labels, ctrl=Ctrl[0], ancillas=path, leaf_fn=first_inv, order="dec")
        qc.cx(Ctrl[0], acc)
    return _e._finalize_block(qc)


def _prepare_latest_paper_t_boundary(qc: QuantumCircuit, *, phase2: Qubit,
    l_t: Sequence[Qubit], l_rp: Sequence[Qubit], l_s: Sequence[Qubit],
    n: int, scratch: Sequence[Qubit]) -> None:
    """Prepare R3=ell_t+1 and R4=n+3-ell_rp-ell_s, then select R4 in Phase 4."""
    lt=list(l_t); rp=list(l_rp); ls=list(l_s); work=list(scratch); w=len(lt)
    if len(rp)!=w or len(ls)<w: raise ValueError("incompatible T-boundary widths")
    ls=ls[:w]
    if len(work)<w+1: raise ValueError("insufficient T-boundary scratch")
    carry=work[w]; arith=work[:w]+[carry]
    # raw truth-minus-one ell_t -> absolute label ell_t+1
    _e.add_const_mod_2n(qc,lt,2,arith)
    # raw ell_rp -> absolute label n+3-ell_rp-ell_s = n+1-rp_raw-ls_raw
    _e.const_minus_inplace(qc,rp,n+1,arith)
    qc.append(_e.cuccaro_sub_mod_2n_no_z_gate(w,name="SUB_ls_T_BOUNDARY"),ls+rp+[carry])
    for a,b in zip(lt,rp): _e.cswap_toffoli(qc,phase2,a,b)


def _restore_latest_paper_t_boundary(qc: QuantumCircuit, *, phase2: Qubit,
    l_t: Sequence[Qubit], l_rp: Sequence[Qubit], l_s: Sequence[Qubit],
    n: int, scratch: Sequence[Qubit]) -> None:
    lt=list(l_t); rp=list(l_rp); ls=list(l_s); work=list(scratch); w=len(lt); ls=ls[:w]
    carry=work[w]; arith=work[:w]+[carry]
    for a,b in zip(lt,rp): _e.cswap_toffoli(qc,phase2,a,b)
    _e.const_minus_inplace(qc,rp,n+1,arith)
    qc.append(_e.cuccaro_sub_mod_2n_no_z_gate(w,name="SUB_ls_RESTORE_T_BOUNDARY"),ls+rp+[carry])
    _e.sub_const_mod_2n(qc,lt,2,arith)

# Reuse the low-aux length update; it is already the paper dirty-work construction with live-range shared scratch.
import eea_circuit_s835_lowaux as _low
# R-side interval arithmetic is the FASTDUAL pruned-unary construction
# defined above.  Do not replace it with the serial per-label equality scan: 
# the latter is semantically useful as a diagnostic fallback but does not
# implement the paper's Figure-10 gate complexity.
len_update_lt_unary_gate = _low.len_update_lt_unary_gate
len_update_lrp_unary_gate = _low.len_update_lrp_unary_gate

@lru_cache(maxsize=None)
def swap_work_and_len_unary_shared_gate(*, n: int, len_width: int, k4: int, K4: int,
                                        k5: int, K5: int, name: str = "SWAP_AND_LEN_S835_FAST") -> Gate:
    work_size = n + 3
    depth4 = _e.unary_depth(K4 - k4 + 1)
    K5_decode = min(K5 + 1, n + 3)
    depth5 = _e.unary_depth(K5_decode - k5 + 1)
    scratch4 = max(len_width + 1, depth4 + 2)
    scratch5 = max(len_width + 1, depth5 + 2)
    scratch_size = max(scratch4, scratch5)
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch, name=name)
    for i in range(work_size):
        _e.cswap_toffoli(qc, Ctrl[0], Work1[i], Work2[i])
    gate_lt = len_update_lt_unary_gate(n=n, k=k4, K=K4, len_width=len_width)
    _e._append_with_optional_clbits(qc, gate_lt, [Ctrl[0]] + list(Work1[k4 - 1:K4]) + list(Work2[k4 - 1:K4])
                                    + list(l_t) + list(l_rp) + list(Scratch[:scratch4]))
    gate_lrp = len_update_lrp_unary_gate(n=n, k=k5, K=K5_decode, len_width=len_width)
    _e._append_with_optional_clbits(qc, gate_lrp, [Ctrl[0]] + list(Work1[k5 - 1:K5_decode]) + list(Work2[k5 - 1:K5_decode])
                                    + list(l_t) + list(l_rp) + list(Scratch[:scratch5]))
    return _e._finalize_block(qc)


@lru_cache(maxsize=None)
def swap_work_and_len_unary_shared_inverse_gate(*, n: int, len_width: int, k4: int, K4: int,
                                                k5: int, K5: int,
                                                name: str = "SWAP_AND_LEN_S835_FAST_INV") -> Gate:
    """Literal dynamic inverse of the paper end-of-iteration block.

    Each length update is an XOR-write involution.  The inverse therefore
    applies the two updates in reverse order and then reverses the controlled
    full Work-register swap.  Measurement-assisted AND uncomputation inside
    the length decoders is retained, so this inverse has the same low-Toffoli
    implementation as the forward paper circuit.
    """
    work_size = n + 3
    depth4 = _e.unary_depth(K4 - k4 + 1)
    K5_decode = min(K5 + 1, n + 3)
    depth5 = _e.unary_depth(K5_decode - k5 + 1)
    scratch4 = max(len_width + 1, depth4 + 2)
    scratch5 = max(len_width + 1, depth5 + 2)
    scratch_size = max(scratch4, scratch5)
    Ctrl = QuantumRegister(1, "Ctrl")
    Work1 = QuantumRegister(work_size, "Work1")
    Work2 = QuantumRegister(work_size, "Work2")
    l_t = QuantumRegister(len_width, "l_t")
    l_rp = QuantumRegister(len_width, "l_rp")
    Scratch = QuantumRegister(scratch_size, "Scratch")
    qc = _e._block_circuit(Ctrl, Work1, Work2, l_t, l_rp, Scratch, name=name)

    gate_lrp = len_update_lrp_unary_gate(n=n, k=k5, K=K5_decode, len_width=len_width)
    _e._append_with_optional_clbits(
        qc, gate_lrp, [Ctrl[0]] + list(Work1[k5 - 1:K5_decode])
        + list(Work2[k5 - 1:K5_decode]) + list(l_t) + list(l_rp)
        + list(Scratch[:scratch5]),
    )
    gate_lt = len_update_lt_unary_gate(n=n, k=k4, K=K4, len_width=len_width)
    _e._append_with_optional_clbits(
        qc, gate_lt, [Ctrl[0]] + list(Work1[k4 - 1:K4])
        + list(Work2[k4 - 1:K4]) + list(l_t) + list(l_rp)
        + list(Scratch[:scratch4]),
    )
    for i in reversed(range(work_size)):
        _e.cswap_toffoli(qc, Ctrl[0], Work1[i], Work2[i])
    return _e._finalize_block(qc)


def _fastdual_interval_scratch_size(n: int, k: int, K: int, len_width: int, shift_width: int) -> int:
    """Scratch size used by ``lc_interval_addsub_unary_gate``.

    This helper mirrors the scratch layout in ``lc_interval_addsub_unary_gate``.
    It is intentionally kept next to ``qiskit_paper_aux_size`` because the
    default Aux size used by the checkpointed counter must scale with this
    value.  For n=256 the worst case is 19 scratch qubits plus the temporary
    Ctrl bit, i.e. Aux=20.  For n=512 the unary path depth increases by one
    on each of the two endpoint scans, so the worst-case scratch is 21 and
    Aux must be 22.
    """
    if k > K:
        return 0
    endpoint_width = max(len_width, shift_width)
    rel_count = K - k + 1
    labels_main = list(range(rel_count))
    depth = _tight_unary_depth_for_labels(labels_main) if labels_main else 0
    base = max(2 * depth, endpoint_width)
    return base + 3


def _fastdual_prefix_scratch_size(k: int, K: int, len_width: int) -> int:
    if k > K:
        return 0
    depth = _e.unary_depth(K - k + 1)
    return max(depth, len_width) + 3


def _fastdual_interval_scratch_size(label_count: int, endpoint_width: int) -> int:
    """Scratch qubits used by lc_interval_addsub_unary_gate.

    The FASTDUAL interval Add/Sub block handles a one-more-than-a-power-of-two
    interval by pulling the top label out as a special endpoint.  Its two endpoint
    unary paths therefore have depth based on ``main_count`` rather than directly
    on ``label_count``.  The scratch layout in lc_interval_addsub_unary_gate is

        base = max(2*depth, endpoint_width)
        Scratch[base], Scratch[base+1], Scratch[base+2]

    so the number of scratch qubits needed by the block is ``base + 3``.
    This is 19 for n=256 but grows to 21 for n=384/512; the previous hard-coded
    lower bound of 19 caused the n=512 qubit-arity mismatch.
    """
    if label_count <= 1:
        depth = 0
    else:
        main_count = label_count
        if label_count > 1 and ((label_count - 1) & (label_count - 2)) == 0:
            main_count = label_count - 1
        depth = 0 if main_count <= 1 else (main_count - 1).bit_length()
    return max(2 * depth, endpoint_width) + 3


def _step_windows(n: int, T: int) -> dict[str, tuple[int, int]]:
    """Latest-paper Appendix-A.2 windows used by the emitted circuit."""
    return _e.active_windows(n, T)


def qiskit_paper_aux_size(n: int, len_width: int, shift_width: int, T_max: Optional[int] = None,
                          include_algorithm1: bool = False) -> int:
    """Shared paper auxiliary pool, including the temporary Ctrl wire.

    The dominant R block uses two pruned unary paths.  For a 257-label window
    the endpoint 256 is handled explicitly, leaving two depth-8 paths and
    exactly 19 scratch wires.  Together with Aux[0]=Ctrl this is the paper's
    20-qubit pool at n=256.  Terminal padding shares the same pool by spilling
    its epoch into the known-zero quotient metadata during R arithmetic.
    """
    if T_max is None:
        T_max = _e.Nmax_steps(n)
    max_r = max_swap = max_t = max_l4 = max_l5 = 1
    for T in range(1, T_max + 1):
        w = _step_windows(n, T)
        max_r = max(max_r, w["r_addsub"][1] - w["r_addsub"][0] + 1)
        max_swap = max(max_swap, w["swap"][1] - w["swap"][0] + 1)
        max_t = max(max_t, w["t_addsub"][1] - w["t_addsub"][0] + 1)
        max_l4 = max(max_l4, w["len_update_lt"][1] - w["len_update_lt"][0] + 1)
        max_l5 = max(max_l5, w["len_update_lrp"][1] - w["len_update_lrp"][0] + 1)
    endpoint_width = max(len_width, shift_width)
    r_need = _fastdual_interval_scratch_size(max_r, endpoint_width)
    swap_need = max(len_width + 1, _e.unary_depth(max_swap) + 1)
    t_need = _fastdual_prefix_scratch_size(1, max_t, len_width) + 1
    len_need = max(len_width + 1, _e.unary_depth(max(max_l4, max_l5)) + 2)
    ordinary_need = max(
        shift_width + 4, r_need, swap_need, t_need,
        max(len_width, shift_width) + 3, len_need, len_width + 1,
    )
    # Aux[0] is Ctrl.  The remaining wires are shared by all serial blocks.
    return max(1 + ordinary_need, 20)

def make_global_registers_noctrl(*, n: int, len_width: int, shift_width: int,
                                 T_max: Optional[int] = None, include_algorithm1: bool = False,
                                 aux_size: Optional[int] = None):
    work_size = n + 3
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
    if aux_size is None:
        aux_size = qiskit_paper_aux_size(n, len_width, shift_width, T_max, include_algorithm1)
    Aux = QuantumRegister(aux_size, "Aux")
    return Phase1, Phase2, Iter, Sign, Work1, Work2, l_t, l_q, l_s, l_rp, Aux


def _make_condition(qc: QuantumCircuit, conditions, out: Qubit, scratch: Sequence[Qubit]) -> None:
    _e.compute_control(qc, conditions, out, scratch)


def _store_terminal_epoch_in_lq(
    qc: QuantumCircuit, *, terminal: Qubit, shift_epoch: Qubit, l_q: Sequence[Qubit]
) -> None:
    """Temporarily clear ShiftEpoch without allocating another qubit.

    On every terminal branch Algorithm 3 has logical ell_q=0, represented by
    the all-ones truth-minus-one word.  Controlled on ``terminal`` we first
    turn l_q[0] into a clean zero and then swap the epoch into that bit.  Thus
    ShiftEpoch is zero while the Figure-10 FASTDUAL block uses the complete
    paper auxiliary pool.  The operation is reversed immediately after the R
    arithmetic, before l_q is used again.
    """
    if not l_q:
        raise ValueError("terminal epoch spill needs a quotient-length bit")
    qc.cx(terminal, l_q[0])
    _e.cswap_toffoli(qc, terminal, shift_epoch, l_q[0])


def _restore_terminal_epoch_from_lq(
    qc: QuantumCircuit, *, terminal: Qubit, shift_epoch: Qubit, l_q: Sequence[Qubit]
) -> None:
    if not l_q:
        raise ValueError("terminal epoch spill needs a quotient-length bit")
    _e.cswap_toffoli(qc, terminal, shift_epoch, l_q[0])
    qc.cx(terminal, l_q[0])


def _toggle_r_control_nonterminal(
    qc: QuantumCircuit, *, conditions, ctrl: Qubit, l_rp: Sequence[Qubit],
    scratch: Sequence[Qubit],
) -> None:
    """Toggle an R-block control and exclude terminal branches.

    Encoded ell_r'=0 is the all-ones word.  The equality flag is computed,
    used to add the negative condition [ell_r' != 0], and uncomputed *before*
    the arithmetic gate.  Consequently every qubit in ``scratch`` is clean
    while the pruned dual-unary Figure-10 block is active.  Calling this helper
    a second time after the block clears ``ctrl``.
    """
    scratch = list(scratch)
    if len(scratch) < 2:
        raise ValueError("nonterminal R control needs clean scratch")
    z_rp = scratch[0]
    pool = scratch[1:]
    _e.mcx_vchain(qc, list(l_rp), z_rp, pool)
    _make_condition(qc, list(conditions) + [(z_rp, 0)], ctrl, pool)
    _e.mcx_vchain(qc, list(l_rp), z_rp, pool)


def _append_terminal_padding_rotation(
    qc: QuantumCircuit, *, terminal: Qubit, shift_epoch: Qubit,
    Work2: Sequence[Qubit], l_s: Sequence[Qubit], scratch: Sequence[Qubit],
    n: int, T_max: int,
) -> None:
    """Apply the exact terminal padding permutation with a borrowed epoch bit.

    The low word ``l_s`` uses truth-minus-one encoding.  During terminal
    padding its modular increment is extended by ``shift_epoch``: the physical
    epoch stores the complement of the logical high counter bit.  Thus the
    initial zero code is ``epoch=0, l_s=all-ones`` and the first padding
    rotation maps it to ``epoch=1, l_s=0``.  This distinguishes a low-word wrap
    from the genuine ell_s=0 state without adding a logical qubit.
    """
    Work2=list(Work2); l_s=list(l_s); scratch=list(scratch)
    max_padding=max(0,int(T_max)-4*int(n))
    if max_padding >= (1 << (len(l_s)+1)):
        raise ValueError("borrowed shift-epoch bit is too narrow for the fixed schedule")
    need=max(1, len(l_s)-1) + 1 + max(0, len(l_s)-2)
    if len(scratch) < max(len(l_s), 2):
        raise ValueError("terminal shift-epoch update needs additional clean scratch")

    for i in range(len(Work2)-1):
        _e.cswap_toffoli(qc, terminal, Work2[i], Work2[i+1])
    chain=scratch[:max(0,len(l_s)-1)]
    _e.inc_mod2n_1ctrl(qc, terminal, l_s, chain)

    wrapped=scratch[0]
    pool=scratch[1:]
    for q in l_s: qc.x(q)
    _e.mcx_vchain(qc, l_s, wrapped, pool)
    for q in reversed(l_s): qc.x(q)
    qc.ccx(terminal, wrapped, shift_epoch)
    for q in l_s: qc.x(q)
    _e.mcx_vchain(qc, l_s, wrapped, pool)
    for q in reversed(l_s): qc.x(q)


def _append_terminal_padding_rotation_inverse(
    qc: QuantumCircuit, *, terminal: Qubit, shift_epoch: Qubit,
    Work2: Sequence[Qubit], l_s: Sequence[Qubit], scratch: Sequence[Qubit],
    n: int, T_max: int,
) -> None:
    """Reverse one exact terminal-padding step.

    The forward operation rotates Work2 left, increments the low shift word,
    and toggles the borrowed epoch on low-word wrap.  At the reverse boundary
    the current low word already contains the incremented value, so the wrap
    predicate is recomputed first, followed by a controlled decrement and the
    inverse (right) rotation.
    """
    Work2 = list(Work2); l_s = list(l_s); scratch = list(scratch)
    max_padding = max(0, int(T_max) - 4 * int(n))
    if max_padding >= (1 << (len(l_s) + 1)):
        raise ValueError("borrowed shift-epoch bit is too narrow for the fixed schedule")
    if len(scratch) < max(len(l_s), 2):
        raise ValueError("terminal shift-epoch inverse needs additional clean scratch")

    wrapped = scratch[0]
    pool = scratch[1:]
    for q in l_s: qc.x(q)
    _e.mcx_vchain(qc, l_s, wrapped, pool)
    for q in reversed(l_s): qc.x(q)
    qc.ccx(terminal, wrapped, shift_epoch)
    for q in l_s: qc.x(q)
    _e.mcx_vchain(qc, l_s, wrapped, pool)
    for q in reversed(l_s): qc.x(q)

    chain = scratch[:max(0, len(l_s) - 1)]
    _e.dec_mod2n_1ctrl(qc, terminal, l_s, chain)
    for i in range(len(Work2) - 2, -1, -1):
        _e.cswap_toffoli(qc, terminal, Work2[i], Work2[i + 1])


def append_one_step_T(qc: QuantumCircuit, *, T: int, n: int, len_width: int, shift_width: int,
                      Phase1, Phase2, Iter, Sign, Work1, Work2, l_t, l_q, l_s, l_rp, Aux,
                      T_max: Optional[int] = None) -> None:
    work_size = n + 3
    windows = _step_windows(n, T)
    k1, K1 = windows["r_addsub"]
    k2, K2 = windows["swap"]
    k3, K3 = windows["t_addsub"]
    k4, K4 = windows["len_update_lt"]
    k5, K5 = windows["len_update_lrp"]
    ctrl = Aux[0]
    shift_epoch = Aux[1]
    scratch = list(Aux[2:])
    if len(scratch) < 2:
        raise ValueError("Algorithm-3 step needs at least two scratch qubits")
    terminal = scratch[0]
    block_scratch = scratch[1:]
    # terminal ^= [Phase1=0 and encoded ell_r'=0].  Terminal branches perform
    # only the explicit padding rotation below; the ordinary pre-shift is
    # temporarily disabled so that the low-word wrap can update ShiftEpoch.
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)
    _append_terminal_padding_rotation(
        qc, terminal=terminal, shift_epoch=shift_epoch,
        Work2=Work2, l_s=l_s, scratch=block_scratch, n=n,
        T_max=int(T_max if T_max is not None else _e.Nmax_steps(n)),
    )
    qc.cx(terminal, Phase1[0])
    pre = _e.pre_shift_gate(work_size=work_size, shift_width=shift_width)
    _e._append_with_optional_clbits(qc, pre, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s) + block_scratch[:pre.num_qubits-(2+work_size+shift_width)])
    qc.cx(terminal, Phase1[0])
    # Release the borrowed epoch wire for the paper Figure-10 decoder.
    # Terminal branches have ell_q=0 (all-ones encoding), so the epoch can be
    # spilled reversibly into l_q[0].  We then uncompute the terminal flag; all
    # nineteen Aux[1:] wires are clean during each FASTDUAL interval block.
    _store_terminal_epoch_in_lq(
        qc, terminal=terminal, shift_epoch=shift_epoch, l_q=l_q
    )
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)

    r_gate_scratch = list(Aux[1:])

    # R subtraction: Phase1=0 and ell_r' != 0.
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0)], ctrl=ctrl, l_rp=l_rp, scratch=scratch
    )
    rsub = lc_interval_addsub_unary_gate(
        n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
        mode="sub", sign_update=True, target="work1", name="R_SUB_S835_FAST",
    )
    rfixed = 2 + 2 * (K1 - k1 + 1) + len_width + len_width + shift_width
    rneed = rsub.num_qubits - rfixed
    if rneed > len(r_gate_scratch):
        raise ValueError(f"R subtraction needs {rneed} paper scratch qubits, have {len(r_gate_scratch)}")
    _e._append_with_optional_clbits(
        qc, rsub, [ctrl, Sign[0]] + list(Work1[k1-1:K1]) + list(Work2[k1-1:K1])
        + list(l_t) + list(l_q) + list(l_s) + r_gate_scratch[:rneed],
    )
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0)], ctrl=ctrl, l_rp=l_rp, scratch=scratch
    )

    # if Phase1=0 and Phase2=1 then Sign ^= 1.  Terminal branches are
    # excluded by the same ell_r' != 0 predicate.
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (Phase2[0], 1)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch,
    )
    qc.cx(ctrl, Sign[0])
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (Phase2[0], 1)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch,
    )

    # R restoring addition: Phase1=0, ell_r'!=0, and not(Phase2 & Sign).
    tmp = scratch[0]
    qc.ccx(Phase2[0], Sign[0], tmp)
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (tmp, 0)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch[1:],
    )
    qc.ccx(Phase2[0], Sign[0], tmp)
    radd = lc_interval_addsub_unary_gate(
        n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
        mode="add", sign_update=False, target="work1", name="R_ADD_S835_FAST",
    )
    rneed = radd.num_qubits - rfixed
    if rneed > len(r_gate_scratch):
        raise ValueError(f"R addition needs {rneed} paper scratch qubits, have {len(r_gate_scratch)}")
    _e._append_with_optional_clbits(
        qc, radd, [ctrl, Sign[0]] + list(Work1[k1-1:K1]) + list(Work2[k1-1:K1])
        + list(l_t) + list(l_q) + list(l_s) + r_gate_scratch[:rneed],
    )
    qc.ccx(Phase2[0], Sign[0], tmp)
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (tmp, 0)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch[1:],
    )
    qc.ccx(Phase2[0], Sign[0], tmp)

    # Restore the terminal epoch before quotient/phase metadata are used.
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)
    _restore_terminal_epoch_from_lq(
        qc, terminal=terminal, shift_epoch=shift_epoch, l_q=l_q
    )
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)

    # In Phase 2 the quotient length is increased before selecting the newly
    # inserted least-significant quotient lane (as in the unoptimized
    # Algorithm 2 ordering).  In Phase 3 the existing least-significant lane is
    # selected first and the length is decreased afterwards.  This keeps the
    # merged Figure-9 selector itself exactly J=ell_t+ell_q+1 in both cases.
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)
    _e.inc_mod2n_1ctrl(qc, ctrl, list(l_q), scratch[:max(0, len_width - 1)])
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)

    # Swap: ctrl = Phase1 xor Phase2.
    qc.cx(Phase1[0], ctrl); qc.cx(Phase2[0], ctrl)
    lcs = lc_swap_unary_gate(k=k2, K=K2, len_width=len_width)
    _e._append_with_optional_clbits(qc, lcs, [ctrl, Sign[0]] + list(Work1[k2-1:K2]) + list(l_t) + list(l_q)
                                    + scratch[:lcs.num_qubits-(2+(K2-k2+1)+len_width+len_width)])
    qc.cx(Phase2[0], ctrl); qc.cx(Phase1[0], ctrl)

    # Phase-3 removal updates ell_q after the swap.
    _make_condition(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, scratch)
    _e.dec_mod2n_1ctrl(qc, ctrl, list(l_q), scratch[:max(0, len_width - 1)])
    _make_condition(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, scratch)
    # Latest-paper Arithmetic block 3: prepare the phase-dependent full endpoint.
    tmp=scratch[0]; twork=scratch[1:]
    _make_condition(qc,[(Phase2[0],0),(Sign[0],1)],tmp,twork)
    _make_condition(qc,[(Phase1[0],1),(tmp,0)],ctrl,twork)
    _make_condition(qc,[(Phase2[0],0),(Sign[0],1)],tmp,twork)
    _prepare_latest_paper_t_boundary(qc,phase2=Phase2[0],l_t=l_t,l_rp=l_rp,l_s=l_s,n=n,scratch=twork)
    tsub=lc_prefix_addsub_prepared_boundary_gate(k=k3,K=K3,len_width=len_width,mode="sub",sign_update=False,target="work2",name="T_SUB_LATEST_PAPER")
    fixed=2+2*(K3-k3+1)+len_width
    _e._append_with_optional_clbits(qc,tsub,[ctrl,Sign[0]]+list(Work1[k3-1:K3])+list(Work2[k3-1:K3])+list(l_t)+twork[:tsub.num_qubits-fixed])
    _make_condition(qc,[(Phase2[0],0),(Sign[0],1)],tmp,twork)
    _make_condition(qc,[(Phase1[0],1),(tmp,0)],ctrl,twork)
    _make_condition(qc,[(Phase2[0],0),(Sign[0],1)],tmp,twork)
    qc.cx(Phase1[0],Sign[0])
    _make_condition(qc,[(Phase1[0],1)],ctrl,twork)
    tadd=lc_prefix_addsub_prepared_boundary_gate(k=k3,K=K3,len_width=len_width,mode="add",sign_update=True,target="work2",name="T_ADD_LATEST_PAPER")
    fixed=2+2*(K3-k3+1)+len_width
    _e._append_with_optional_clbits(qc,tadd,[ctrl,Sign[0]]+list(Work1[k3-1:K3])+list(Work2[k3-1:K3])+list(l_t)+twork[:tadd.num_qubits-fixed])
    _make_condition(qc,[(Phase1[0],1)],ctrl,twork)
    _restore_latest_paper_t_boundary(qc,phase2=Phase2[0],l_t=l_t,l_rp=l_rp,l_s=l_s,n=n,scratch=twork)
    # Post-shift
    post = _e.post_shift_gate(work_size=work_size, shift_width=shift_width)
    _e._append_with_optional_clbits(qc, post, [Phase1[0], Phase2[0]] + list(Work2) + list(l_s) + scratch[:post.num_qubits-(2+work_size+shift_width)])
    # Phase update
    pupdate = _e.phase_update_gate(len_width=len_width, shift_width=shift_width, include_shift_epoch=True)
    fixed = 3 + len_width + len_width + shift_width + 1
    _e._append_with_optional_clbits(qc, pupdate,
        [Phase1[0], Phase2[0], Sign[0]] + list(l_q) + list(l_rp) + list(l_s) + [shift_epoch]
        + scratch[:pupdate.num_qubits-fixed])
    # End iteration every four steps.
    if T % 4 == 0:
        z_lq = scratch[0]; z_ls = scratch[1]; eq_pool = scratch[2:]
        _e.mcx_vchain(qc, list(l_q), z_lq, eq_pool)
        qc.x(shift_epoch)
        _e.mcx_vchain(qc, [*l_s, shift_epoch], z_ls, eq_pool)
        qc.x(shift_epoch)
        qc.ccx(z_lq, z_ls, ctrl)
        swlen = swap_work_and_len_unary_shared_gate(n=n, len_width=len_width, k4=k4, K4=K4, k5=k5, K5=K5)
        need = swlen.num_qubits - (1+2*work_size+2*len_width)
        _e._append_with_optional_clbits(qc, swlen, [ctrl] + list(Work1) + list(Work2) + list(l_t) + list(l_rp) + scratch[2:2+need])
        qc.cx(ctrl, Iter[0])
        qc.ccx(z_lq, z_ls, ctrl)
        qc.x(shift_epoch)
        _e.mcx_vchain(qc, [*l_s, shift_epoch], z_ls, eq_pool)
        qc.x(shift_epoch)
        _e.mcx_vchain(qc, list(l_q), z_lq, eq_pool)


def append_one_step_T_inverse(
    qc: QuantumCircuit, *, T: int, n: int, len_width: int, shift_width: int,
    Phase1, Phase2, Iter, Sign, Work1, Work2, l_t, l_q, l_s, l_rp, Aux,
    T_max: Optional[int] = None,
) -> None:
    """Append the exact dynamic reverse of one latest-paper Algorithm-3 step.

    This reverses the emitted forward construction block-by-block while keeping
    measurement-assisted unary-AND cleanup in both directions.  It is therefore
    the low-Toffoli inverse required by Figure 15, not a renamed forward block
    and not a coherent ``q.inverse()`` substitute.
    """
    work_size = n + 3
    windows = _step_windows(n, T)
    k1, K1 = windows["r_addsub"]
    k2, K2 = windows["swap"]
    k3, K3 = windows["t_addsub"]
    k4, K4 = windows["len_update_lt"]
    k5, K5 = windows["len_update_lrp"]
    ctrl = Aux[0]
    shift_epoch = Aux[1]
    scratch = list(Aux[2:])
    if len(scratch) < 2:
        raise ValueError("Algorithm-3 inverse needs at least two scratch qubits")
    terminal = scratch[0]
    block_scratch = scratch[1:]

    # H^{-1}: undo the end-of-iteration Work swap/length writes and Iter toggle.
    if T % 4 == 0:
        z_lq = scratch[0]; z_ls = scratch[1]; eq_pool = scratch[2:]
        _e.mcx_vchain(qc, list(l_q), z_lq, eq_pool)
        qc.x(shift_epoch)
        _e.mcx_vchain(qc, [*l_s, shift_epoch], z_ls, eq_pool)
        qc.x(shift_epoch)
        qc.ccx(z_lq, z_ls, ctrl)
        qc.cx(ctrl, Iter[0])
        swinv = swap_work_and_len_unary_shared_inverse_gate(
            n=n, len_width=len_width, k4=k4, K4=K4, k5=k5, K5=K5,
        )
        need = swinv.num_qubits - (1 + 2 * work_size + 2 * len_width)
        _e._append_with_optional_clbits(
            qc, swinv, [ctrl] + list(Work1) + list(Work2) + list(l_t)
            + list(l_rp) + scratch[2:2 + need],
        )
        qc.ccx(z_lq, z_ls, ctrl)
        qc.x(shift_epoch)
        _e.mcx_vchain(qc, [*l_s, shift_epoch], z_ls, eq_pool)
        qc.x(shift_epoch)
        _e.mcx_vchain(qc, list(l_q), z_lq, eq_pool)

    # G^{-1}, F^{-1}: phase update and post-shift are pure reversible blocks.
    pupdate = _e.phase_update_inverse_gate(
        len_width=len_width, shift_width=shift_width, include_shift_epoch=True,
    )
    fixed = 3 + len_width + len_width + shift_width + 1
    _e._append_with_optional_clbits(
        qc, pupdate,
        [Phase1[0], Phase2[0], Sign[0]] + list(l_q) + list(l_rp)
        + list(l_s) + [shift_epoch] + scratch[:pupdate.num_qubits - fixed],
    )
    post = _e.post_shift_gate(work_size=work_size, shift_width=shift_width)
    _e._append_with_optional_clbits(
        qc, post.inverse(),
        [Phase1[0], Phase2[0]] + list(Work2) + list(l_s)
        + scratch[:post.num_qubits - (2 + work_size + shift_width)],
    )

    # E^{-1}: recreate the latest-paper phase-dependent endpoint, undo T-add,
    # the explicit Sign toggle, and then T-sub, before restoring both lengths.
    twork = scratch[1:]
    _prepare_latest_paper_t_boundary(
        qc, phase2=Phase2[0], l_t=l_t, l_rp=l_rp, l_s=l_s,
        n=n, scratch=twork,
    )
    _make_condition(qc, [(Phase1[0], 1)], ctrl, twork)
    tadd_inv = lc_prefix_addsub_prepared_boundary_gate(
        k=k3, K=K3, len_width=len_width, mode="add", sign_update=True,
        target="work2", name="T_ADD_LATEST_PAPER_INV", inverse=True,
    )
    tfixed = 2 + 2 * (K3 - k3 + 1) + len_width
    _e._append_with_optional_clbits(
        qc, tadd_inv, [ctrl, Sign[0]] + list(Work1[k3 - 1:K3])
        + list(Work2[k3 - 1:K3]) + list(l_t)
        + twork[:tadd_inv.num_qubits - tfixed],
    )
    _make_condition(qc, [(Phase1[0], 1)], ctrl, twork)
    qc.cx(Phase1[0], Sign[0])

    tmp = scratch[0]
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, twork)
    _make_condition(qc, [(Phase1[0], 1), (tmp, 0)], ctrl, twork)
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, twork)
    tsub_inv = lc_prefix_addsub_prepared_boundary_gate(
        k=k3, K=K3, len_width=len_width, mode="sub", sign_update=False,
        target="work2", name="T_SUB_LATEST_PAPER_INV", inverse=True,
    )
    _e._append_with_optional_clbits(
        qc, tsub_inv, [ctrl, Sign[0]] + list(Work1[k3 - 1:K3])
        + list(Work2[k3 - 1:K3]) + list(l_t)
        + twork[:tsub_inv.num_qubits - tfixed],
    )
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, twork)
    _make_condition(qc, [(Phase1[0], 1), (tmp, 0)], ctrl, twork)
    _make_condition(qc, [(Phase2[0], 0), (Sign[0], 1)], tmp, twork)
    _restore_latest_paper_t_boundary(
        qc, phase2=Phase2[0], l_t=l_t, l_rp=l_rp, l_s=l_s,
        n=n, scratch=twork,
    )

    # D^{-1}: undo Phase-3 decrement, the selected quotient-bit swap, and the
    # Phase-2 pre-increment in exact reverse order.
    _make_condition(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, scratch)
    _e.inc_mod2n_1ctrl(qc, ctrl, list(l_q), scratch[:max(0, len_width - 1)])
    _make_condition(qc, [(Phase1[0], 1), (Phase2[0], 0)], ctrl, scratch)
    qc.cx(Phase1[0], ctrl); qc.cx(Phase2[0], ctrl)
    lcs = lc_swap_unary_gate(k=k2, K=K2, len_width=len_width)
    _e._append_with_optional_clbits(
        qc, lcs, [ctrl, Sign[0]] + list(Work1[k2 - 1:K2])
        + list(l_t) + list(l_q)
        + scratch[:lcs.num_qubits - (2 + (K2 - k2 + 1) + 2 * len_width)],
    )
    qc.cx(Phase2[0], ctrl); qc.cx(Phase1[0], ctrl)
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)
    _e.dec_mod2n_1ctrl(qc, ctrl, list(l_q), scratch[:max(0, len_width - 1)])
    _make_condition(qc, [(Phase1[0], 0), (Phase2[0], 1)], ctrl, scratch)

    # C^{-1}: spill the terminal epoch back into the known-zero quotient word so
    # the complete Aux[1:] pool is clean while reversing both Figure-10 blocks.
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)
    _store_terminal_epoch_in_lq(
        qc, terminal=terminal, shift_epoch=shift_epoch, l_q=l_q,
    )
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)

    r_gate_scratch = list(Aux[1:])
    rfixed = 2 + 2 * (K1 - k1 + 1) + 2 * len_width + shift_width

    # B3^{-1}: restoring addition does not change Sign, so its original
    # state-dependent control is available directly at the reverse boundary.
    tmp = scratch[0]
    qc.ccx(Phase2[0], Sign[0], tmp)
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (tmp, 0)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch[1:],
    )
    qc.ccx(Phase2[0], Sign[0], tmp)
    radd_inv = lc_interval_addsub_unary_gate(
        n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
        mode="add", sign_update=False, target="work1",
        name="R_ADD_S835_FAST_INV", inverse=True,
    )
    rneed = radd_inv.num_qubits - rfixed
    _e._append_with_optional_clbits(
        qc, radd_inv, [ctrl, Sign[0]] + list(Work1[k1 - 1:K1])
        + list(Work2[k1 - 1:K1]) + list(l_t) + list(l_q) + list(l_s)
        + r_gate_scratch[:rneed],
    )
    qc.ccx(Phase2[0], Sign[0], tmp)
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (tmp, 0)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch[1:],
    )
    qc.ccx(Phase2[0], Sign[0], tmp)

    # B2^{-1}: the explicit Phase-2 Sign toggle is an involution.
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (Phase2[0], 1)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch,
    )
    qc.cx(ctrl, Sign[0])
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0), (Phase2[0], 1)],
        ctrl=ctrl, l_rp=l_rp, scratch=scratch,
    )

    # B1^{-1}: undo the subtraction and its carry-to-Sign update.
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0)], ctrl=ctrl,
        l_rp=l_rp, scratch=scratch,
    )
    rsub_inv = lc_interval_addsub_unary_gate(
        n=n, k=k1, K=K1, len_width=len_width, shift_width=shift_width,
        mode="sub", sign_update=True, target="work1",
        name="R_SUB_S835_FAST_INV", inverse=True,
    )
    rneed = rsub_inv.num_qubits - rfixed
    _e._append_with_optional_clbits(
        qc, rsub_inv, [ctrl, Sign[0]] + list(Work1[k1 - 1:K1])
        + list(Work2[k1 - 1:K1]) + list(l_t) + list(l_q) + list(l_s)
        + r_gate_scratch[:rneed],
    )
    _toggle_r_control_nonterminal(
        qc, conditions=[(Phase1[0], 0)], ctrl=ctrl,
        l_rp=l_rp, scratch=scratch,
    )

    # A^{-1}: restore the epoch from l_q, undo the ordinary pre-shift only on
    # nonterminal branches, and finally reverse the terminal-only rotation.
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)
    _restore_terminal_epoch_from_lq(
        qc, terminal=terminal, shift_epoch=shift_epoch, l_q=l_q,
    )
    qc.cx(terminal, Phase1[0])
    pre = _e.pre_shift_gate(work_size=work_size, shift_width=shift_width)
    _e._append_with_optional_clbits(
        qc, pre.inverse(), [Phase1[0], Phase2[0]] + list(Work2) + list(l_s)
        + block_scratch[:pre.num_qubits - (2 + work_size + shift_width)],
    )
    qc.cx(terminal, Phase1[0])
    _append_terminal_padding_rotation_inverse(
        qc, terminal=terminal, shift_epoch=shift_epoch,
        Work2=Work2, l_s=l_s, scratch=block_scratch, n=n,
        T_max=int(T_max if T_max is not None else _e.Nmax_steps(n)),
    )
    _make_condition(qc, [(Phase1[0], 0)] + [(q, 1) for q in l_rp], terminal, block_scratch)


def _eea_microstep_count_for_modulus(p: int, x: int) -> int:
    r0, r1 = int(p), min(int(x), int(p) - int(x))
    weighted = 0
    while r1:
        q, r2 = divmod(r0, r1)
        weighted += q.bit_length()
        r0, r1 = r1, r2
    return 4 * weighted


def certified_T_max_for_modulus(p: int, *, exhaustive_bit_limit: int = 16) -> int:
    p = int(p); n = p.bit_length()
    paper = int(_e.Nmax_steps(n))
    if n <= exhaustive_bit_limit:
        return max(paper, max(_eea_microstep_count_for_modulus(p, x) for x in range(1, p)))
    return paper


def _apply_controlled_permutation(qc: QuantumCircuit, ctrl: Qubit, wires: Sequence[Qubit], target_of_source: Sequence[int]) -> None:
    wires = list(wires); m = len(wires)
    if sorted(target_of_source) != list(range(m)):
        raise ValueError("target_of_source is not a permutation")
    desired = [None] * m
    for source, target in enumerate(target_of_source):
        desired[target] = source
    current = list(range(m))
    for pos in range(m):
        want = desired[pos]
        if current[pos] == want:
            continue
        j = current.index(want)
        _e.cswap_toffoli(qc, ctrl, wires[pos], wires[j])
        current[pos], current[j] = current[j], current[pos]


def canonical_rotate_work2(
    qc: QuantumCircuit,
    Work2: Sequence[Qubit],
    l_s: Sequence[Qubit],
    scratch: Sequence[Qubit],
    *,
    shift_epoch: Optional[Qubit] = None,
    inverse: bool = False,
    outer_ctrl: Optional[Qubit] = None,
) -> None:
    """Canonicalize/decanonicalize Work2 using the extended shift counter."""
    Work2=list(Work2); low=list(l_s); scratch=list(scratch)
    counter=list(low)
    if shift_epoch is not None:
        qc.x(shift_epoch)
        counter.append(shift_epoch)
    need=len(counter)+(2 if outer_ctrl is not None else 1)
    if len(scratch)<need:
        raise ValueError(f"need at least {need} clean scratch qubits")
    cs=scratch[:len(counter)+1]
    joint=scratch[len(counter)+1] if outer_ctrl is not None else None
    _e.add_const_mod_2n(qc,counter,1,cs)
    m=len(Work2)
    order=range(len(counter)-1,-1,-1) if inverse else range(len(counter))
    for b in order:
        d=(1<<b)%m
        if d==0: continue
        target=[((i-d)%m if inverse else (i+d)%m) for i in range(m)]
        control=counter[b]
        if outer_ctrl is not None:
            assert joint is not None
            qc.ccx(outer_ctrl,counter[b],joint);control=joint
        _apply_controlled_permutation(qc,control,Work2,target)
        if outer_ctrl is not None: qc.ccx(outer_ctrl,counter[b],joint)
    _e.sub_const_mod_2n(qc,counter,1,cs)
    if shift_epoch is not None:
        qc.x(shift_epoch)


def _controlled_adjacent_basis_transposition(
    qc: QuantumCircuit, wires: Sequence[Qubit], state: int, target_bit: int,
    *, outer_ctrl: Optional[Qubit], scratch: Sequence[Qubit],
) -> None:
    wires=list(wires); controls=[]; flipped=[]
    for i,q in enumerate(wires):
        if i==target_bit: continue
        if ((state>>i)&1)==0:
            qc.x(q);flipped.append(q)
        controls.append(q)
    if outer_ctrl is not None: controls.append(outer_ctrl)
    _e.mcx_vchain(qc,controls,wires[target_bit],list(scratch))
    for q in reversed(flipped): qc.x(q)


def _controlled_basis_transposition(
    qc: QuantumCircuit, wires: Sequence[Qubit], a: int, b: int,
    *, outer_ctrl: Optional[Qubit], scratch: Sequence[Qubit],
) -> None:
    """Swap two computational-basis states and leave every other state fixed."""
    wires=list(wires)
    diff=[i for i in range(len(wires)) if ((a^b)>>i)&1]
    if not diff:return
    path=[a];cur=a
    for bit in diff:
        cur ^= 1<<bit;path.append(cur)
    for i in range(len(path)-1):
        _controlled_adjacent_basis_transposition(qc,wires,path[i],diff[i],outer_ctrl=outer_ctrl,scratch=scratch)
    for i in range(len(path)-3,-1,-1):
        _controlled_adjacent_basis_transposition(qc,wires,path[i],diff[i],outer_ctrl=outer_ctrl,scratch=scratch)


def compress_terminal_shift_epoch(
    qc: QuantumCircuit, *, shift_epoch: Qubit, l_s: Sequence[Qubit],
    outer_ctrl: Optional[Qubit] = None, scratch: Sequence[Qubit] = (),
) -> None:
    """Reversibly fold the borrowed terminal epoch into the low pointer.

    Exact EEA microstep counts are multiples of four.  At a fixed horizon the
    reachable epoch-one terminal endpoints therefore have the two low bits of
    ``l_s`` equal to ``11``.  The disjoint basis transposition

        (epoch=1, low[1:0]=11) <-> (epoch=0, low[1:0]=00)

    clears the borrowed epoch on every reachable forward endpoint while
    retaining the padding count in ``l_s``.  The same operation is its own
    inverse and restores the epoch before the literal inverse traversal.
    """
    l_s=list(l_s)
    if len(l_s)<2: raise ValueError("terminal epoch packing needs at least two low bits")
    wires=[l_s[0],l_s[1],shift_epoch]
    _controlled_basis_transposition(qc,wires,0b111,0b000,outer_ctrl=outer_ctrl,scratch=list(scratch))


def build_full_steps_circuit(
    n: int, len_width: int, shift_width: int, T_max: Optional[int] = None,
    aux_size: Optional[int] = None, T_start: int = 1, T_end: Optional[int] = None,
) -> QuantumCircuit:
    """Literal Algorithm-3 schedule using the repaired paper blocks.

    The separate public Ctrl register is retained for compatibility and remains
    clean; the step-local Ctrl is Aux[0], as in the 835-qubit implementation.
    """
    if T_max is None:
        T_max = _e.Nmax_steps(n)
    if T_end is None:
        T_end = T_max
    if not (1 <= T_start <= T_end <= T_max):
        raise ValueError("bad step range")
    if aux_size is None:
        aux_size = qiskit_paper_aux_size(n, len_width, shift_width, T_max)
    regs = _e.make_global_registers(n=n, len_width=len_width, shift_width=shift_width,
                                    T_max=T_max, aux_size=aux_size)
    qc = QuantumCircuit(*regs, name="MODINV_STEPS_S835_FASTDUAL_PAPER")
    Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux = regs
    for T in range(T_start, T_end + 1):
        append_one_step_T(qc, T=T, n=n, len_width=len_width, shift_width=shift_width,
                          Phase1=Phase1, Phase2=Phase2, Iter=Iter, Sign=Sign,
                          Work1=Work1, Work2=Work2, l_t=l_t, l_q=l_q, l_s=l_s, l_rp=l_rp,
                          Aux=Aux, T_max=T_max)

    # Algorithm 1 exposes the logical t' prefix after a fixed schedule.  During
    # terminal padding Work2 is only circularly shifted, with l_s recording the
    # accumulated shift.  Canonicalize exactly on terminal branches so the
    # externally visible Work2[0:n] implements the paper's "Keep the t' part"
    # instruction.  l_s itself is retained as part of Gamma(x), so this map is
    # fully reversible and the literal circuit dagger restores the circular
    # representation before reversing the Algorithm-3 steps.
    if T_start == 1 and T_end == T_max:
        _e.mcx_vchain(qc, list(l_rp), Ctrl[0], list(Aux[2:]))
        canonical_rotate_work2(qc, Work2, l_s, Aux[2:], shift_epoch=Aux[1], outer_ctrl=Ctrl[0])
        compress_terminal_shift_epoch(qc, shift_epoch=Aux[1], l_s=l_s, outer_ctrl=Ctrl[0], scratch=Aux[2:])
        _e.mcx_vchain(qc, list(l_rp), Ctrl[0], list(Aux[2:]))
    return qc


def build_modular_inversion_algorithm1_circuit(
    *, n: int, p: int, len_width: Optional[int] = None, shift_width: Optional[int] = None,
    T_max: Optional[int] = None,
) -> QuantumCircuit:
    """Paper Algorithm 1 with a literal forward schedule and literal dagger."""
    if not (0 < int(p) < (1 << int(n))):
        raise ValueError("p must satisfy 0 < p < 2^n")
    if len_width is None:
        len_width = int(_e.get_n_config(n)["len_width"])
    if shift_width is None:
        shift_width = int(_e.get_n_config(n)["shift_width"])
    certified_steps = certified_T_max_for_modulus(int(p))
    if T_max is None:
        T_max = certified_steps
    else:
        # An explicit caller bound may be the paper's size-only value.  Promote
        # it when a concrete small-modulus exhaustive certificate is larger
        # (p=419: 52 -> 56), so Algorithm 1 cannot silently truncate.
        T_max = max(int(T_max), certified_steps)
    # Prevent terminal padding from wrapping the shift pointer while retaining
    # the paper widths of ell_t, ell_q, and ell_r'.  The T-side endpoint helpers
    # explicitly support this one-bit-wider shift register.
    required_width = int(_e.terminal_safe_shift_width(n, T_max))
    len_width = int(len_width)
    shift_width = max(int(shift_width), required_width)
    alg1_aux_size = qiskit_paper_aux_size(n, len_width, shift_width, T_max, include_algorithm1=True)
    regs = _e.make_global_registers(n=n, len_width=len_width, shift_width=shift_width,
                                    T_max=T_max, aux_size=alg1_aux_size, include_algorithm1=True)
    Output = QuantumRegister(n, "out")
    qc = QuantumCircuit(*regs, Output, name="MODINV_ALG1_S835_FASTDUAL_PAPER")
    Phase1, Phase2, Iter, Sign, Ctrl, Work1, Work2, l_t, l_q, l_s, l_rp, Aux = regs
    scratch = list(Aux); flag = Ctrl[0]

    qc.x(Work1[0]); _e._set_big_endian_constant(qc, Work1[3:3+n], int(p))
    rprime_le = list(reversed(list(Work2[3:3+n])))
    _e.xor_gt_const_little_endian(qc, rprime_le, int(p)//2, Iter[0], scratch)
    _e.controlled_const_minus_le(qc, Iter[0], rprime_le, int(p), scratch)
    _e.xor_const_into_reg(qc, l_q, (1 << len_width) - 1)
    _e.xor_const_into_reg(qc, l_s, (1 << shift_width) - 1)
    _e.xor_encoded_bit_length_big_endian(qc, Work2[3:3+n], l_rp, flag, scratch)

    steps = build_full_steps_circuit(n, len_width, shift_width, T_max=T_max, aux_size=len(Aux))
    step_qubits = [q for reg in regs for q in reg]
    step_gate = steps.to_gate(label="ALG3_STEPS_S835_FASTDUAL")
    qc.append(step_gate, step_qubits)

    for i in range(n):
        qc.cx(Work2[i], Output[i])
    qc.x(Iter[0]); _e.controlled_const_minus_le(qc, Iter[0], list(Output), int(p), scratch); qc.x(Iter[0])

    # The full step gate already includes the terminal canonicalization.  Its
    # literal dagger first restores the circular Work2 layout and then reverses
    # the Algorithm-3 schedule.
    qc.append(step_gate.inverse(), step_qubits)
    _e.xor_encoded_bit_length_big_endian(qc, Work2[3:3+n], l_rp, flag, scratch)
    _e.xor_const_into_reg(qc, l_s, (1 << shift_width) - 1)
    _e.xor_const_into_reg(qc, l_q, (1 << len_width) - 1)
    _e.controlled_const_minus_le(qc, Iter[0], rprime_le, int(p), scratch)
    _e.xor_gt_const_little_endian(qc, rprime_le, int(p)//2, Iter[0], scratch)
    _e._set_big_endian_constant(qc, Work1[3:3+n], int(p)); qc.x(Work1[0])
    return qc


def build_step_circuit(n:int, T:int, *, T_max:Optional[int]=None, aux_size:Optional[int]=None, measurement_uncompute:bool=True):
    cfg=get_n_config(n); T_max=int(T_max or cfg['T_max']); req=int(_e.terminal_safe_shift_width(n, T_max)); lw=int(cfg['len_width']); sw=max(int(cfg['shift_width']), req)
    if aux_size is None: aux_size=qiskit_paper_aux_size(n,lw,sw,T_max)
    set_measurement_uncompute(measurement_uncompute)
    regs=make_global_registers_noctrl(n=n,len_width=lw,shift_width=sw,T_max=T_max,aux_size=aux_size)
    qc=QuantumCircuit(*regs, name=f"S835_FASTDUAL_STEP_T{T}_{n}")
    Phase1,Phase2,Iter,Sign,Work1,Work2,l_t,l_q,l_s,l_rp,Aux=regs
    append_one_step_T(qc,T=T,n=n,len_width=lw,shift_width=sw,Phase1=Phase1,Phase2=Phase2,Iter=Iter,Sign=Sign,Work1=Work1,Work2=Work2,l_t=l_t,l_q=l_q,l_s=l_s,l_rp=l_rp,Aux=Aux,T_max=T_max)
    return qc

if __name__ == '__main__':
    import argparse,json
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=256); ap.add_argument('--T',type=int,default=1); ap.add_argument('--count',action='store_true'); args=ap.parse_args()
    cfg=get_n_config(args.n); lw=int(cfg['len_width']); sw=int(cfg['shift_width']); Tm=int(cfg['T_max'])
    out={'n':args.n,'len_width':lw,'shift_width':sw,'T_max':Tm,'aux_size':qiskit_paper_aux_size(args.n,lw,sw,Tm)}
    qc=build_step_circuit(args.n,args.T,T_max=Tm)
    out['step_qubits']=qc.num_qubits; out['top_ops']={str(k):int(v) for k,v in qc.count_ops().items()}
    if args.count:
        out['ops']={str(k):int(v) for k,v in _e.count_circuit_ops_recursive(qc).items()}
    print(json.dumps(out,indent=2,sort_keys=True))
