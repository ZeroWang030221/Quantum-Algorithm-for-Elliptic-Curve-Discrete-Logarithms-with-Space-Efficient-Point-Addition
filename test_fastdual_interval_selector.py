"""Exhaustive selector regression for the paper's fast dual-unary R block.

The test targets intervals of size ``2^d+1`` (5 and 9 labels), where the top
endpoint is handled outside the pruned power-of-two unary tree.  For every
reachable pair ``L <= R`` and for add/sub, sign/no-sign, work1/work2 targets,
it compares the emitted dynamic-endpoint block against a direct fixed-interval
Figure-11 ripple circuit.  It also checks endpoint restoration, scratch cleanup,
and the inactive ``Ctrl=0`` branch.
"""

import json
import random
from typing import Sequence

try:
    import qiskit  # type: ignore
except Exception:
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()

from qiskit import QuantumCircuit, QuantumRegister

import eea_circuit_s835_fastdual as fast
import eea_circuit_updated as core
from extended_semantic_sweep import VecSim, idx


def _set_reg_mask(qc, sim: VecSim, reg: Sequence, values: list[int]) -> None:
    for bit, q in enumerate(reg):
        mask = 0
        for case, value in enumerate(values):
            if (int(value) >> bit) & 1:
                mask |= 1 << case
        sim.q[idx(qc, q)] = mask


def _read_reg(qc, sim: VecSim, reg: Sequence, case: int) -> int:
    value = 0
    for bit, q in enumerate(reg):
        value |= ((sim.q[idx(qc, q)] >> case) & 1) << bit
    return value


def _candidate_registers(circuit, *, M: int, width: int) -> dict[str, Sequence]:
    """Recover the production block's logical registers by flat qubit order.

    Real Qiskit intentionally anonymizes a circuit converted with ``to_gate()``:
    ``gate.definition`` contains one quantum register named ``q`` rather than the
    original ``Ctrl``, ``Sign``, ``Work1``, ... registers.  The flat qubit order
    is preserved, however, so reconstruct the logical slices from the production
    circuit's declared register order.  The bundled mini runtime retains the
    original qregs, but this positional mapping works in both environments.
    """
    qubits = list(circuit.qubits)
    cursor = 0

    def take(size: int):
        nonlocal cursor
        out = qubits[cursor:cursor + size]
        if len(out) != size:
            raise AssertionError(
                f"candidate definition too small while taking {size} qubits at offset {cursor}"
            )
        cursor += size
        return out

    out = {
        "Ctrl": take(1),
        "Sign": take(1),
        "Work1": take(M),
        "Work2": take(M),
        "l_t": take(width),
        "l_q": take(width),
        "l_s": take(width),
    }
    out["Scratch"] = qubits[cursor:]
    if not out["Scratch"]:
        raise AssertionError("candidate definition has no Scratch qubits")
    return out


def _direct_fixed_interval(*, M: int, L: int, R: int, k: int, mode: str,
                           sign_update: bool, target: str) -> QuantumCircuit:
    Ctrl = QuantumRegister(1, "Ctrl")
    Sign = QuantumRegister(1, "Sign")
    Work1 = QuantumRegister(M, "Work1")
    Work2 = QuantumRegister(M, "Work2")
    Scratch = QuantumRegister(2, "Scratch")  # carry, clean v-chain helper
    qc = QuantumCircuit(Ctrl, Sign, Work1, Work2, Scratch, name="DIRECT_INTERVAL")
    carry, helper = Scratch

    def qpair(j_abs: int):
        pos = j_abs - k
        if target == "work1":
            return Work2[pos], Work1[pos]
        return Work1[pos], Work2[pos]

    for j in range(R, L - 1, -1):
        addend, tgt = qpair(j)
        core._apply_cell(qc, mode, "first", Ctrl[0], addend, tgt, carry, [helper])
    if sign_update:
        qc.cx(carry, Sign[0])
    for j in range(L, R + 1):
        addend, tgt = qpair(j)
        core._apply_cell(qc, mode, "second", Ctrl[0], addend, tgt, carry, [helper])
    return qc


def _run_one_width(n: int, *, cases_per_interval: int = 32, seed: int = 835) -> dict:
    k, K = 3, n + 3
    M = K - k + 1
    if M not in {5, 9}:
        raise ValueError("this regression intentionally targets 2^d+1 windows")
    width = int(fast.paper_len_width(n))
    rng = random.Random(seed + n)
    maskw = (1 << width) - 1
    records = []

    fast.set_measurement_uncompute(False)
    for mode in ("add", "sub"):
        for sign_update in (False, True):
            for target in ("work1", "work2"):
                candidate = fast.lc_interval_addsub_unary_gate(
                    n=n, k=k, K=K, len_width=width, shift_width=width,
                    mode=mode, sign_update=sign_update, target=target,
                    name=f"SELECTOR_{n}_{mode}_{target}_{int(sign_update)}",
                )
                cq = candidate.definition
                if cq is None:
                    raise AssertionError("candidate gate has no definition")
                by = _candidate_registers(cq, M=M, width=width)
                for L in range(k, K + 1):
                    for R in range(L, K + 1):
                        # Include both inactive and active cases, plus extremal data.
                        ctrls = [i & 1 for i in range(cases_per_interval)]
                        signs = [rng.randrange(2) for _ in ctrls]
                        w1 = [rng.randrange(1 << M) for _ in ctrls]
                        w2 = [rng.randrange(1 << M) for _ in ctrls]
                        if cases_per_interval >= 4:
                            w1[:4] = [0, (1 << M) - 1, 0, (1 << M) - 1]
                            w2[:4] = [0, 0, (1 << M) - 1, (1 << M) - 1]

                        # Choose ell_t=1, ell_q=L-3, ell_s=n+3-R.
                        ell_t = 1
                        ell_q = L - 3
                        ell_s = n + 3 - R
                        raw_lt = (ell_t - 1) & maskw
                        raw_lq = (ell_q - 1) & maskw
                        raw_ls = (ell_s - 1) & maskw

                        init = {}
                        csim = VecSim(cq.num_qubits, cq.num_clbits,
                                      list(range(cases_per_interval)), init)
                        _set_reg_mask(cq, csim, by["Ctrl"], ctrls)
                        _set_reg_mask(cq, csim, by["Sign"], signs)
                        _set_reg_mask(cq, csim, by["Work1"], w1)
                        _set_reg_mask(cq, csim, by["Work2"], w2)
                        _set_reg_mask(cq, csim, by["l_t"], [raw_lt] * cases_per_interval)
                        _set_reg_mask(cq, csim, by["l_q"], [raw_lq] * cases_per_interval)
                        _set_reg_mask(cq, csim, by["l_s"], [raw_ls] * cases_per_interval)
                        csim.run_circuit(cq, list(range(cq.num_qubits)), list(range(cq.num_clbits)))

                        ref = _direct_fixed_interval(M=M, L=L, R=R, k=k, mode=mode,
                                                     sign_update=sign_update, target=target)
                        rb = {reg.name: reg for reg in ref.qregs}
                        rsim = VecSim(ref.num_qubits, ref.num_clbits,
                                      list(range(cases_per_interval)), {})
                        _set_reg_mask(ref, rsim, rb["Ctrl"], ctrls)
                        _set_reg_mask(ref, rsim, rb["Sign"], signs)
                        _set_reg_mask(ref, rsim, rb["Work1"], w1)
                        _set_reg_mask(ref, rsim, rb["Work2"], w2)
                        rsim.run_circuit(ref, list(range(ref.num_qubits)), list(range(ref.num_clbits)))

                        for case in range(cases_per_interval):
                            got = (
                                _read_reg(cq, csim, by["Ctrl"], case),
                                _read_reg(cq, csim, by["Sign"], case),
                                _read_reg(cq, csim, by["Work1"], case),
                                _read_reg(cq, csim, by["Work2"], case),
                            )
                            exp = (
                                _read_reg(ref, rsim, rb["Ctrl"], case),
                                _read_reg(ref, rsim, rb["Sign"], case),
                                _read_reg(ref, rsim, rb["Work1"], case),
                                _read_reg(ref, rsim, rb["Work2"], case),
                            )
                            if got != exp:
                                raise AssertionError(
                                    f"selector mismatch n={n} mode={mode} target={target} "
                                    f"sign={sign_update} L={L} R={R} case={case}: "
                                    f"got={got}, expected={exp}"
                                )
                            if _read_reg(cq, csim, by["l_t"], case) != raw_lt:
                                raise AssertionError("l_t was not restored")
                            if _read_reg(cq, csim, by["l_q"], case) != raw_lq:
                                raise AssertionError("l_q was not restored")
                            if _read_reg(cq, csim, by["l_s"], case) != raw_ls:
                                raise AssertionError("l_s was not restored")
                            if _read_reg(cq, csim, by["Scratch"], case) != 0:
                                raise AssertionError("candidate scratch was not cleaned")
                            if _read_reg(ref, rsim, rb["Scratch"], case) != 0:
                                raise AssertionError("reference scratch was not cleaned")

                        records.append({
                            "mode": mode, "target": target,
                            "sign_update": sign_update, "L": L, "R": R,
                            "cases": cases_per_interval,
                        })

    return {
        "n": n,
        "window": [k, K],
        "label_count": M,
        "configurations": len(records),
        "basis_cases": sum(r["cases"] for r in records),
        "passed": True,
    }


def main() -> None:
    out = {
        "test": "fast-dual unary interval selector against direct Figure-11 ripple",
        "results": [_run_one_width(4), _run_one_width(8)],
        "passed": True,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
