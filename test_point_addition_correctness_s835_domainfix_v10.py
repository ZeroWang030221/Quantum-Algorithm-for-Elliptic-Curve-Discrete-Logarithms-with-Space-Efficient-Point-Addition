"""Correctness/scaling checks for the S835 FASTDUAL corrected-inverse point-addition circuit.

This file contains two layers of tests.

1. Pure Python algebraic tests of the exact Fig.14/Fig.15 schedule.  These do not
   need Qiskit and are intended to prove that the schedule implemented by the
   circuit has the same classical action as affine point addition on all tested
   non-exceptional inputs.

2. Optional Qiskit/Aer smoke tests.  These build the real Qiskit circuit, check
   the live-qubit layout, check for definition-less opaque gates on small n, and
   can simulate a small dynamic-circuit instance if qiskit-aer is installed.

The n=256 circuit is far too large for full state simulation, so n=256/512 are
checked by construction/width and by the compiled counters; functional execution
is only attempted on very small fields.
"""

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

# secp256k1 constants used by the project.
SECP256K1_P = (1 << 256) - (1 << 32) - 977
SECP256K1_A = 0
SECP256K1_B = 7
SECP256K1_GX = 55066263022277343669578718895168534326250603453777594175500187360389116729240
SECP256K1_GY = 32670510020758816978083085130507043184471273380659243275938904335757337482424

Point = Optional[tuple[int, int]]  # None denotes the point at infinity.


def inv_mod(x: int, p: int) -> int:
    x %= p
    if x == 0:
        raise ZeroDivisionError("inverse of zero modulo p")
    return pow(x, -1, p)


def is_on_curve(P: Point, *, p: int, a: int, b: int) -> bool:
    if P is None:
        return True
    x, y = P
    return (y * y - (x * x * x + a * x + b)) % p == 0


def affine_add(P: Point, Q: Point, *, p: int, a: int, b: int) -> Point:
    """Standard affine Weierstrass addition, used as the test oracle."""
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if (x1 - x2) % p == 0 and (y1 + y2) % p == 0:
        return None
    if (x1 - x2) % p == 0 and (y1 - y2) % p == 0:
        if y1 % p == 0:
            return None
        lam = ((3 * x1 * x1 + a) * inv_mod(2 * y1, p)) % p
    else:
        lam = ((y2 - y1) * inv_mod(x2 - x1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    assert is_on_curve((x3, y3), p=p, a=a, b=b)
    return (x3, y3)


def affine_neg(P: Point, *, p: int) -> Point:
    if P is None:
        return None
    x, y = P
    return (x, (-y) % p)


def scalar_mul(k: int, P: Point, *, p: int, a: int, b: int) -> Point:
    R: Point = None
    Q = P
    while k:
        if k & 1:
            R = affine_add(R, Q, p=p, a=a, b=b)
        Q = affine_add(Q, Q, p=p, a=a, b=b)
        k >>= 1
    return R


def enumerate_points(p: int, a: int, b: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for x in range(p):
        rhs = (x * x * x + a * x + b) % p
        for y in range(p):
            if (y * y - rhs) % p == 0:
                pts.append((x, y))
    return pts


def fig15_idiv_reference(x: int, y: int, p: int) -> tuple[int, int, int]:
    """Classical action of Fig.15 in-place division: (X,Y,A=0)->(X,Y/X,A=0)."""
    return x % p, (y * inv_mod(x, p)) % p, 0


def fig15_imul_reference(x: int, y: int, p: int) -> tuple[int, int, int]:
    """Classical action of the analogous in-place multiplication: (X,Y,A=0)->(X,XY,A=0)."""
    return x % p, (x * y) % p, 0


def square_minus_reference(ctrl: int, X: int, Y: int, p: int) -> tuple[int, int, int]:
    """Classical action of Squ; CSub; Squ^dagger: X <- X - ctrl*Y^2, A returns to 0."""
    if ctrl not in (0, 1):
        raise ValueError("ctrl must be 0 or 1")
    return (X - ctrl * Y * Y) % p, Y % p, 0


def fig14_schedule_reference(
    ctrl: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    p: int,
) -> tuple[int, int, int]:
    """Classical action of the exact Fig.14 schedule used by the Qiskit builder.

    Schedule:
        -x2; -ctrl*y2; IDiv; SquMinus; +ctrl*3x2; IMul; Neg; +x2; -ctrl*y2.

    The denominator x1-x2 must be nonzero because the low-width controlled-add
    circuit uses the same IDiv path even when ctrl=0; for ctrl=0 the surrounding
    arithmetic cancels and the final output is (x1,y1).
    """
    if ctrl not in (0, 1):
        raise ValueError("ctrl must be 0 or 1")
    X = (x1 - x2) % p
    Y = (y1 - ctrl * y2) % p
    A = 0

    X, Y, A = fig15_idiv_reference(X, Y, p)
    X, Y, A = square_minus_reference(ctrl, X, Y, p)
    X = (X + ctrl * 3 * x2) % p
    X, Y, A = fig15_imul_reference(X, Y, p)
    if ctrl:
        X = (-X) % p
    X = (X + x2) % p
    Y = (Y - ctrl * y2) % p
    return X, Y, A


def expected_controlled_add(ctrl: int, P1: tuple[int, int], P2: tuple[int, int], *, p: int, a: int, b: int) -> tuple[int, int]:
    if ctrl == 0:
        return P1
    out = affine_add(P1, P2, p=p, a=a, b=b)
    if out is None:
        raise ValueError("exceptional point-addition case produced infinity")
    return out


def assert_fig14_matches_affine(ctrl: int, P1: tuple[int, int], P2: tuple[int, int], *, p: int, a: int, b: int) -> None:
    assert is_on_curve(P1, p=p, a=a, b=b), f"P1 not on curve: {P1}"
    assert is_on_curve(P2, p=p, a=a, b=b), f"P2 not on curve: {P2}"
    x1, y1 = P1
    x2, y2 = P2
    if (x1 - x2) % p == 0:
        raise ValueError("Fig.14 non-exceptional test requires x1 != x2")
    X, Y, A = fig14_schedule_reference(ctrl, x1, y1, x2, y2, p)
    exp = expected_controlled_add(ctrl, P1, P2, p=p, a=a, b=b)
    assert A == 0, f"workspace A not returned to zero: A={A}"
    assert (X, Y) == exp, f"Fig.14 mismatch: got {(X,Y)}, expected {exp}, ctrl={ctrl}, P1={P1}, P2={P2}"


def run_reference_tests(seed: int = 1) -> dict:
    """Run pure-Python algebraic tests over small curves and secp256k1."""
    rng = random.Random(seed)
    cases = 0

    # A small toy curve with many non-exceptional point pairs.
    toy_p, toy_a, toy_b = 97, 2, 3
    pts = enumerate_points(toy_p, toy_a, toy_b)
    P2 = pts[5]
    for _ in range(100):
        P1 = rng.choice(pts)
        if P1[0] == P2[0]:
            continue
        for ctrl in (0, 1):
            # Skip cases that would be exceptional when ctrl=1.
            if ctrl and affine_add(P1, P2, p=toy_p, a=toy_a, b=toy_b) is None:
                continue
            assert_fig14_matches_affine(ctrl, P1, P2, p=toy_p, a=toy_a, b=toy_b)
            cases += 1

    # secp256k1 generator tests.  Use P1=[k]G and P2=G, with k>1 so x1 != x2.
    G = (SECP256K1_GX, SECP256K1_GY)
    assert is_on_curve(G, p=SECP256K1_P, a=SECP256K1_A, b=SECP256K1_B)
    for k in range(2, 18):
        P1 = scalar_mul(k, G, p=SECP256K1_P, a=SECP256K1_A, b=SECP256K1_B)
        assert P1 is not None
        for ctrl in (0, 1):
            assert_fig14_matches_affine(ctrl, P1, G, p=SECP256K1_P, a=SECP256K1_A, b=SECP256K1_B)
            cases += 1

    # Direct Fig.15 and square-minus algebra tests.
    for p in (97, 251, 65537):
        for _ in range(50):
            x = rng.randrange(1, p)
            y = rng.randrange(0, p)
            X, Y, A = fig15_idiv_reference(x, y, p)
            assert X == x % p and A == 0 and (X * Y - y) % p == 0
            X2, Y2, A2 = fig15_imul_reference(x, y, p)
            assert X2 == x % p and A2 == 0 and Y2 == (x * y) % p
            for ctrl in (0, 1):
                Xm, Ym, Am = square_minus_reference(ctrl, x, y, p)
                assert Ym == y % p and Am == 0 and Xm == (x - ctrl * y * y) % p
                cases += 1

    return {"reference_tests_passed": True, "cases_checked": cases, "seed": seed}


# --- Optional Qiskit/Aer checks -------------------------------------------------

def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except Exception:
        return False


def dry_s835_width(n: int) -> dict:
    """Compute the expected S835 FASTDUAL width without importing Qiskit.

    This mirrors the current FASTDUAL layout: work tails 3+3, four persistent
    controls, four length/shift registers, and step_aux=20 for the tested
    cryptographic sizes.  For n where the step-aux formula grows, the real
    builder is authoritative; use --qiskit-build-widths in a Qiskit environment.
    """
    lw = max(math.floor(math.log2(n)) + 1, math.ceil(math.log2(n + 4)))
    sw = lw
    step_aux = 20
    S = 3 + 3 + 4 + 3 * lw + sw + step_aux
    return {
        "n": n,
        "len_width": lw,
        "shift_width": sw,
        "assumed_step_aux": step_aux,
        "S": S,
        "quantum_qubits": 1 + 3 * n + S,
    }


def build_widths_qiskit(n_values: Sequence[int], p_values: Optional[dict[int, int]] = None) -> list[dict]:
    """Build top-level Qiskit circuits and return width/top-level-op reports."""
    if not qiskit_available():
        raise RuntimeError("Qiskit is not installed; cannot build circuits in this environment")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_report, SECP256K1_P

    rows: list[dict] = []
    for n in n_values:
        p = (p_values or {}).get(n, SECP256K1_P if n == 256 else (1 << n) - 189)
        try:
            rows.append(build_report(n=n, p=p))
        except Exception as ex:  # keep going so scaling failures are visible.
            rows.append({"n": n, "build_failed": True, "error": repr(ex)})
    return rows


def audit_small_no_opaque(n: int = 4, p: int = 13) -> dict:
    """Run the small-n no-opaque audit script if Qiskit is available."""
    if not qiskit_available():
        return {"skipped": True, "reason": "Qiskit is not installed"}
    import subprocess
    out = Path(f"point_addition_s835_fastdual_n{n}_audit_tmp.json")
    cmd = [sys.executable, str(Path(__file__).with_name("audit_s835_fastdual_no_opaque_small.py")), "--n", str(n), "--p", str(p), "--out", str(out)]
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())





def _operation_name(inst) -> str:
    return getattr(inst, "name", str(inst))


def _is_known_aer_op(name: str, backend_basis: set[str]) -> bool:
    # Aer supports a long list of basis operations depending on the installed
    # version and method.  Keep explicit dynamic/structural operations here
    # because some versions omit them from configuration().basis_gates even
    # though they are accepted by assemble_circuit.
    always_ok = {
        "barrier", "delay", "measure", "reset", "if_else", "while_loop",
        "for_loop", "switch_case", "break_loop", "continue_loop", "store",
    }
    return name in backend_basis or name in always_ok


def unknown_instruction_names(qc, backend_basis: set[str]) -> list[str]:
    names: set[str] = set()
    for item in qc.data:
        op = item.operation if hasattr(item, "operation") else item[0]
        name = _operation_name(op)
        if not _is_known_aer_op(name, backend_basis):
            names.add(name)
    return sorted(names)


def _custom_ops_with_definitions(qc, backend_basis: set[str]) -> list[str]:
    """Return unknown instruction names that have definitions and should be expanded."""
    out: set[str] = set()
    for item in qc.data:
        op = item.operation if hasattr(item, "operation") else item[0]
        name = _operation_name(op)
        if _is_known_aer_op(name, backend_basis):
            continue
        if _force_instruction_definition(op) is not None:
            out.add(name)
    return sorted(out)






# During --aer-small the corrected inverse EEA instruction name contains n but
# not the toy modulus p.  If an old Qiskit version leaves the instruction
# unmaterialized, the manual expander uses this global to rebuild the definition
# with the same modulus used to build the test circuit.
_AER_SMALL_FORCED_P = None
_AER_SMALL_FORCED_TMAX = None

def _force_instruction_definition(op):
    """Return a real QuantumCircuit definition for a lazy/custom instruction.

    Some Qiskit/Aer versions do not automatically materialize definitions of
    subclassed Instruction objects during transpilation, which leads to errors
    such as "HighLevelSynthesis is unable to synthesize ...".  This helper
    explicitly asks LazyDefinedInstruction objects to build their definitions,
    and it also recognizes the corrected true-inverse EEA instruction by name.
    """
    try:
        definition = op.definition
    except Exception:
        definition = None
    if definition is not None:
        return definition

    builder = getattr(op, "_lazy_builder", None)
    if builder is not None:
        try:
            definition = builder()
            try:
                op.definition = definition
            except Exception:
                pass
            return definition
        except Exception:
            pass

    name = _operation_name(op)
    prefix = "EEA_INVERSE_SHARED_ALG3_FASTDUAL_TRUE_"
    if name.startswith(prefix):
        try:
            n = int(name[len(prefix):].split("_")[0])
            from under1000_eea_shared_s835_fastdual_wrapped_corrected import inverse_eea_shared_definition
            forced_p = _AER_SMALL_FORCED_P
            forced_tmax = _AER_SMALL_FORCED_TMAX
            if forced_p is None:
                definition = inverse_eea_shared_definition(n)
            else:
                definition = inverse_eea_shared_definition(n, int(forced_p), forced_tmax)
            try:
                op.definition = definition
            except Exception:
                pass
            return definition
        except Exception as ex:
            # Keep this as an attribute for diagnostics in the final JSON/error.
            try:
                op._force_definition_error = repr(ex)
            except Exception:
                pass
            return None
    return None


def _safe_basis_for_transpile(backend_basis: set[str]) -> list[str]:
    """Return a conservative basis_gates list accepted by qiskit.transpile.

    Aer configuration().basis_gates may contain simulator pseudo-instructions
    such as save_expval_var.  Passing these through basis_gates causes a
    ValueError.  The fallback transpile only needs the actual circuit gates.
    """
    allowed = {
        "id", "x", "sx", "rz", "rx", "ry", "h", "z", "s", "sdg",
        "cx", "cz", "ccx", "swap", "measure", "reset", "barrier",
    }
    return sorted((backend_basis & allowed) | {"x", "cx", "ccx", "h", "z", "cz", "measure", "reset", "barrier"})

def _copy_operation_with_mapped_condition(op, cmap):
    """Return op with any classical condition remapped to the output circuit bits.

    Qiskit stores classically controlled gates as an operation plus an internal
    ``condition`` field.  When we manually copy/expand circuits, the operation
    must not keep pointing at clbits from the source circuit; otherwise Aer or
    later compose/decompose passes can see a condition on a bit that is not in
    the circuit.  Most project operations use one-bit conditions, but this helper
    also handles register conditions when the destination circuit preserves the
    register object.
    """
    cond = getattr(op, "condition", None)
    if cond is None:
        return op
    try:
        op2 = op.copy()
    except Exception:
        op2 = op
    try:
        classical, val = cond
    except Exception:
        return op2
    try:
        mapped = cmap.get(classical, classical)
    except Exception:
        mapped = classical
    try:
        op2.condition = (mapped, val)
    except Exception:
        pass
    return op2


def _empty_like_for_manual_expand(qc, name: str):
    """Create an empty circuit preserving all qubits/clbits of ``qc``.

    Some Qiskit definitions use anonymous/loose bits in addition to named
    registers.  Reconstructing an empty circuit from only ``qc.qregs`` and
    ``qc.cregs`` drops these loose bits, causing errors such as
    ``Bit '<Qubit uid=...>' is not in the circuit`` during manual expansion.
    ``copy_empty_like`` preserves them when available.  A numeric fallback is
    provided for older Qiskit versions; it loses register names, but preserves
    bit counts and is only used if ``copy_empty_like`` is unavailable.
    """
    try:
        return qc.copy_empty_like(name=name)
    except Exception:
        from qiskit import QuantumCircuit
        try:
            out = QuantumCircuit(*qc.qregs, *qc.cregs, name=name)
            if len(out.qubits) == len(qc.qubits) and len(out.clbits) == len(qc.clbits):
                return out
        except Exception:
            pass
        return QuantumCircuit(len(qc.qubits), len(qc.clbits), name=name)


def _bit_maps_for_manual_expand(src, dst):
    """Map bits from a source circuit to an empty-like destination circuit."""
    qmap = {src.qubits[i]: dst.qubits[i] for i in range(len(src.qubits))}
    cmap = {src.clbits[i]: dst.clbits[i] for i in range(len(src.clbits))}
    return qmap, cmap
def _manual_expand_project_custom(qc, backend_basis: set[str], *, max_depth: int = 64, _depth: int = 0):
    """Manually expand project-defined composite Instructions.

    We intentionally do not call QuantumCircuit.decompose here.  Some Qiskit
    versions fail when decomposing dynamic Instructions whose definitions carry
    internal classical measurement bits, raising errors like

        DAGCircuitError: 'bit mapping invalid: expected 50, got 51'

    The project uses measurement-based adders/EEA steps where a custom
    Instruction may contain internal measurement bits.  In the full circuit only
    the logical input/output wires are visible, while these internal classical
    bits are local to the definition.  Qiskit's generic DAG decompose expects the
    node clbit count and the definition clbit count to match exactly, and can
    therefore fail.  This helper expands definitions explicitly and allocates
    local classical bits for any definition-internal measurements.

    This remains a real circuit expansion: definitions are composed into the
    parent circuit.  No arithmetic formula or classical oracle is substituted.
    """
    if _depth > max_depth:
        raise RuntimeError("custom-instruction expansion exceeded max_depth")
    from qiskit import ClassicalRegister, QuantumCircuit

    # Preserve *all* bits from the original definition, including anonymous/loose
    # qubits that are not contained in qc.qregs.  We then remap every operation's
    # qargs/cargs from qc's bits to out's corresponding bits by index.
    out = _empty_like_for_manual_expand(qc, (getattr(qc, "name", "qc") or "qc") + "_aer_expanded")
    qmap, cmap = _bit_maps_for_manual_expand(qc, out)
    auto_creg_id = 0
    expansions: list[dict] = []

    for item in qc.data:
        if hasattr(item, "operation"):
            op, qargs, cargs = item.operation, list(item.qubits), list(item.clbits)
        else:  # older qiskit tuple style
            op, qargs, cargs = item[0], list(item[1]), list(item[2])
        name = _operation_name(op)

        mapped_qargs = [qmap[q] for q in qargs]
        mapped_cargs = [cmap[c] for c in cargs]

        if _is_known_aer_op(name, backend_basis):
            out.append(_copy_operation_with_mapped_condition(op, cmap), mapped_qargs, mapped_cargs)
            continue

        definition = _force_instruction_definition(op)
        if definition is None:
            # Leave the operation in place for now.  prepare_for_aer will report
            # it explicitly before any Aer run if it remains unknown.
            out.append(_copy_operation_with_mapped_condition(op, cmap), mapped_qargs, mapped_cargs)
            continue

        expanded_def, child_meta = _manual_expand_project_custom(definition, backend_basis, max_depth=max_depth, _depth=_depth + 1)
        if expanded_def.num_qubits != len(qargs):
            raise RuntimeError(
                f"Cannot expand {name}: definition has {expanded_def.num_qubits} qubits "
                f"but node has {len(qargs)} qargs"
            )

        # Map externally supplied clbits first.  If the definition contains
        # additional local measurement bits, allocate fresh clbits in the parent.
        full_cargs = list(mapped_cargs)
        missing = expanded_def.num_clbits - len(full_cargs)
        if missing > 0:
            local = ClassicalRegister(missing, f"_auto_m{_depth}_{auto_creg_id}")
            auto_creg_id += 1
            out.add_register(local)
            full_cargs.extend(list(local))
        elif missing < 0:
            # Qiskit allows nodes to carry more clbits than their definition uses;
            # compose only needs the definition's clbits.
            full_cargs = full_cargs[: expanded_def.num_clbits]

        out.compose(expanded_def, qubits=mapped_qargs, clbits=full_cargs, inplace=True)
        expansions.append({
            "name": name,
            "definition_qubits": int(expanded_def.num_qubits),
            "definition_clbits": int(expanded_def.num_clbits),
            "node_qargs": int(len(qargs)),
            "node_cargs": int(len(cargs)),
            "auto_clbits_added": int(max(0, missing)),
            "child_expansions": child_meta.get("num_expansions", 0),
        })

    return out, {
        "num_expansions": len(expansions) + sum(e.get("child_expansions", 0) for e in expansions),
        "expansion_sample": expansions[:20],
    }


def prepare_for_aer(qc, sim, *, max_decompose_passes: int = 24):
    """Expand project-specific custom instructions before running Aer.

    This version avoids Qiskit's generic ``QuantumCircuit.decompose`` because it
    can fail on dynamic Instructions with internal classical measurement bits.
    Instead, it manually composes instruction definitions and allocates local
    classical bits for hidden measurement/venting registers.
    """
    from qiskit import transpile
    try:
        backend_basis = set(sim.configuration().basis_gates or [])
    except Exception:
        backend_basis = set()

    backend_basis.update({
        "id", "x", "sx", "rz", "rx", "ry", "h", "z", "s", "sdg",
        "cx", "cz", "ccx", "swap", "measure", "reset", "barrier",
        "if_else",
    })

    initial_unknown = unknown_instruction_names(qc, backend_basis)
    cur, expand_meta = _manual_expand_project_custom(qc, backend_basis, max_depth=max_decompose_passes)
    after_manual_unknown = unknown_instruction_names(cur, backend_basis)

    # At this point there should be no project custom instructions left.  If an
    # unknown instruction remains but has a definition, expand one more time for
    # compatibility with older Qiskit versions that create new composite names
    # during compose.
    extra_passes = []
    for pass_id in range(4):
        expandable = _custom_ops_with_definitions(cur, backend_basis)
        if not expandable:
            break
        cur, meta = _manual_expand_project_custom(cur, backend_basis, max_depth=max_decompose_passes)
        extra_passes.append({"pass": pass_id, "expanded_names": expandable[:20], "meta": meta})
    after_extra_unknown = unknown_instruction_names(cur, backend_basis)

    final_unknown_before_transpile = unknown_instruction_names(cur, backend_basis)
    if final_unknown_before_transpile:
        raise RuntimeError(
            "Aer-unknown instructions remain after manual expansion, before transpile: "
            f"{final_unknown_before_transpile[:20]}"
        )

    try:
        tqc = transpile(cur, backend=sim, optimization_level=0)
    except Exception:
        tqc = transpile(cur, basis_gates=_safe_basis_for_transpile(backend_basis), optimization_level=0)

    unknown_after = unknown_instruction_names(tqc, backend_basis)
    if unknown_after:
        raise RuntimeError(
            "Aer-unknown instructions remain after manual expansion/transpile: "
            f"{unknown_after[:20]}"
        )

    report = {
        "initial_ops": len(qc.data),
        "initial_unknown_sample": initial_unknown[:20],
        "expanded_ops_before_transpile": len(cur.data),
        "unknown_after_manual_expansion": after_manual_unknown[:20],
        "unknown_after_extra_expansion": after_extra_unknown[:20],
        "unknown_before_transpile": final_unknown_before_transpile[:20],
        "extra_expansion_passes": extra_passes,
        "manual_expansion_meta": expand_meta,
        "transpiled_ops": len(tqc.data),
        "unknown_after_transpile": unknown_after,
        "top_ops_after_transpile": {str(k): int(v) for k, v in tqc.count_ops().items()},
    }
    return tqc, report

def aer_small_point_addition_test(
    n: int = 4,
    p: int = 13,
    a: int = 0,
    b: int = 7,
    shots: int = 8,
    aer_method: str = "matrix_product_state",
) -> dict:
    """Attempt a full dynamic-circuit simulation on a small field using qiskit-aer.

    This is optional because qiskit-aer is not installed in all environments and
    n=256 is not intended to be simulated as a full state/vector circuit.
    """
    if not qiskit_available():
        return {"skipped": True, "reason": "Qiskit is not installed"}
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, transpile
        from qiskit.result import marginal_counts
        from qiskit_aer import AerSimulator
    except Exception as ex:
        return {"skipped": True, "reason": f"qiskit-aer or qiskit result tools unavailable: {ex!r}"}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from point_addition_fig14_s835_fastdual_wrapped_corrected_quadratic import build_point_addition_fig14_quadratic

    global _AER_SMALL_FORCED_P, _AER_SMALL_FORCED_TMAX
    _AER_SMALL_FORCED_P = int(p)
    _AER_SMALL_FORCED_TMAX = None

    pts = enumerate_points(p, a, b)
    if len(pts) < 4:
        return {"skipped": True, "reason": "not enough points on the toy curve"}
    P2 = pts[1]

    # The Fig.14 schedule uses two Fig.15 subroutines whose current circuit
    # implementation is EEA-backed.  Therefore the denominator of IDiv,
    # x1-x2, must be nonzero, and the multiplicand used by IMul, x2-x3,
    # must also be nonzero.  The latter condition is not automatically implied
    # by non-exceptional affine addition on small toy curves.  For example on
    # y^2=x^3+7 over F_13, the first naive choice P1=(8,5), P2=(7,8) gives
    # P1+P2=(7,5), so x2-x3=0 and the EEA-backed IMul is outside its domain.
    P1 = None
    expected = None
    rejected = []
    for cand in pts[2:]:
        if cand[0] == P2[0]:
            rejected.append({"P1": cand, "reason": "x1_minus_x2_zero"})
            continue
        R = affine_add(cand, P2, p=p, a=a, b=b)
        if R is None:
            rejected.append({"P1": cand, "reason": "exceptional_sum"})
            continue
        if (P2[0] - R[0]) % p == 0:
            rejected.append({"P1": cand, "R": R, "reason": "internal_imul_multiplier_x2_minus_x3_zero"})
            continue
        P1 = cand
        expected = R
        break
    if P1 is None or expected is None:
        return {"skipped": True, "reason": "no toy point satisfies Fig15 nonzero-domain conditions", "P2": P2, "rejected": rejected}
    x1, y1 = P1
    x2, y2 = P2

    body = build_point_addition_fig14_quadratic(n=n, p=p, x2=x2, y2=y2)
    qc = QuantumCircuit(*body.qregs, *body.cregs, name=f"aer_test_n{n}")
    qregs = {r.name: r for r in qc.qregs}
    X = qregs["X_x1_to_x3"]
    Y = qregs["Y_y1_to_y3"]
    A = qregs["A_shared_work"]
    S = qregs["S_shared_eea_arith"]
    ctrl = qregs["ctrl"]
    qc.x(ctrl[0])
    for i in range(n):
        if (x1 >> i) & 1:
            qc.x(X[i])
        if (y1 >> i) & 1:
            qc.x(Y[i])
    qc.compose(body, qubits=qc.qubits, clbits=qc.clbits, inplace=True)

    final = ClassicalRegister(3 * n + len(S), "final_XYA_S")
    qc.add_register(final)
    off = 0
    for reg in (X, Y, A, S):
        for q in reg:
            qc.measure(q, final[off]); off += 1
    final_indices = [qc.find_bit(c).index for c in final]

    # Do not use Aer's default automatic method here.  Even for n=4 the S835
    # dynamic point-addition circuit has dozens of qubits; Aer's automatic
    # selection can choose the dense statevector simulator, which immediately
    # requires exponential memory.  The matrix-product-state backend is the
    # intended small-circuit smoke-test backend for this highly structured
    # basis-state arithmetic circuit.
    sim = AerSimulator(method=aer_method)
    try:
        sim.set_options(
            matrix_product_state_truncation_threshold=0.0,
            matrix_product_state_max_bond_dimension=256,
            max_parallel_threads=1,
        )
    except Exception:
        # Older Aer versions may not expose all MPS options.
        pass

    # Aer cannot execute project-specific custom instructions such as
    # QUAD_X_SUB_X2 directly.  They must first be expanded into their Qiskit
    # definitions and then transpiled to Aer-supported primitive operations.
    # This keeps the test as a real circuit test: the definitions are expanded,
    # not replaced by algebraic formulas.
    tqc, aer_prep_report = prepare_for_aer(qc, sim)

    # Recompute final-register classical indices after transpilation because the
    # transpiler is allowed to rebuild Circuit/Clbit objects while preserving
    # register names and order.
    final_t = next((r for r in tqc.cregs if r.name == final.name), None)
    if final_t is None:
        # Some older Qiskit versions lose register names when manually expanded
        # through anonymous-bit fallback circuits.  Classical bit order is still
        # preserved by construction, so reuse the indices computed before
        # transpilation as a conservative fallback.
        final_indices_t = final_indices
    else:
        final_indices_t = [tqc.find_bit(c).index for c in final_t]

    result = sim.run(tqc, shots=shots).result()
    if not result.success:
        return {
            "aer_small_passed": False,
            "n": n,
            "p": p,
            "aer_method": aer_method,
            "shots": shots,
            "failure": "Aer returned an unsuccessful Result",
            "status": getattr(result, "status", None),
            "aer_preparation": aer_prep_report,
        }
    counts = result.get_counts()
    marg = marginal_counts(counts, indices=final_indices_t)

    exp_x, exp_y = expected
    expected_a = 0
    expected_s = 0
    accepted = set()
    for key in marg:
        compact = key.replace(" ", "")
        # Try both displayed bit orders; Qiskit count strings are conventionally
        # printed most-significant classical bit first.
        candidates = [compact, compact[::-1]]
        ok = False
        for bits in candidates:
            # final register order is X little-endian, Y little-endian, A, S.
            x_bits = bits[0:n]
            y_bits = bits[n:2*n]
            a_bits = bits[2*n:3*n]
            s_bits = bits[3*n:]
            x_val = int(x_bits[::-1], 2) if x_bits else 0
            y_val = int(y_bits[::-1], 2) if y_bits else 0
            a_val = int(a_bits[::-1], 2) if a_bits else 0
            s_val = int(s_bits[::-1], 2) if s_bits else 0
            if (x_val, y_val, a_val, s_val) == (exp_x, exp_y, expected_a, expected_s):
                ok = True
                break
        if not ok:
            raise AssertionError(f"unexpected final marginal key={key!r}, expected X,Y,A,S={(exp_x, exp_y, 0, 0)}")
        accepted.add(key)
    return {
        "aer_small_passed": True,
        "n": n,
        "p": p,
        "P1": P1,
        "P2": P2,
        "expected": expected,
        "rejected_candidate_points": rejected,
        "domain_conditions": {"x1_minus_x2_nonzero": True, "x2_minus_x3_nonzero": True},
        "shots": shots,
        "aer_method": aer_method,
        "distinct_final_marginals": len(accepted),
        "aer_preparation": aer_prep_report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-only", action="store_true", help="run pure-Python algebraic tests only")
    ap.add_argument("--qiskit-build-widths", action="store_true", help="build Qiskit top-level circuits for --n-list")
    ap.add_argument("--audit-small", action="store_true", help="run small-n no-opaque audit script")
    ap.add_argument("--aer-small", action="store_true", help="attempt qiskit-aer simulation of a small dynamic circuit")
    ap.add_argument("--aer-method", default="matrix_product_state", help="AerSimulator method for --aer-small; use matrix_product_state to avoid dense statevector memory")
    ap.add_argument("--aer-shots", type=int, default=8, help="shots for --aer-small")
    ap.add_argument("--n-list", default="256,384,512", help="comma-separated n values for width/build checks")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="point_addition_correctness_s835_report.json")
    args = ap.parse_args()

    report: dict = {"script": Path(__file__).name}
    report["reference"] = run_reference_tests(seed=args.seed)
    n_values = [int(x) for x in args.n_list.split(",") if x.strip()]
    report["dry_widths"] = [dry_s835_width(n) for n in n_values]

    if args.qiskit_build_widths:
        try:
            report["qiskit_widths"] = build_widths_qiskit(n_values)
        except Exception as ex:
            report["qiskit_widths"] = {"skipped_or_failed": True, "error": repr(ex)}
    if args.audit_small:
        report["small_no_opaque_audit"] = audit_small_no_opaque()
    if args.aer_small:
        report["aer_small"] = aer_small_point_addition_test(shots=args.aer_shots, aer_method=args.aer_method)

    text = json.dumps(report, indent=2, sort_keys=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

# Pytest-compatible entry points.  These are intentionally pure-Python and do
# not require Qiskit.  Optional Qiskit/Aer checks are exposed through the CLI.
def test_fig14_reference_schedule_matches_affine() -> None:
    rep = run_reference_tests(seed=1)
    assert rep["reference_tests_passed"] is True


def test_s835_dry_widths_for_256_and_512() -> None:
    assert dry_s835_width(256)["quantum_qubits"] == 835
    assert dry_s835_width(512)["quantum_qubits"] == 1607
