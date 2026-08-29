"""Fail-closed structural audit against the latest paper circuit specification.

This test is deliberately independent of the large semantic sweeps.  It guards
against the exact regression that previously rebound the paper's pruned dual
unary R decoder to a serial per-position equality decoder, and it checks the
resource-critical Figure-11 / Table-5 implementation choices.
"""

import inspect
import json
from pathlib import Path
from typing import Any

try:
    import qiskit  # type: ignore
except Exception:
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()

from qiskit import QuantumCircuit, QuantumRegister

import eea_circuit_s835_fastdual as fast
import eea_circuit_updated as core
import under1000_eea_shared_s835_fastdual_wrapped as wrapped
from ccx_recursive_block_counter import CounterPolicy, count_gate_or_circuit


def _count(obj: Any):
    c = count_gate_or_circuit(
        obj,
        policy=CounterPolicy(mcx_policy="clean-vchain", expand_swap_to_cx=True),
    )
    opaque = {k: int(v) for k, v in c.items() if str(k).startswith("OPAQUE::") and int(v)}
    stopped = {k: int(v) for k, v in c.items() if str(k).startswith("STOP::") and int(v)}
    if opaque or stopped:
        raise AssertionError(f"non-primitive terms: opaque={opaque}, stopped={stopped}")
    return c


def _primitive_names(qc: QuantumCircuit) -> list[str]:
    out: list[str] = []
    for item in qc.data:
        op = item.operation if hasattr(item, "operation") else item[0]
        out.append(str(getattr(op, "name", op)).lower())
    return out


def main() -> None:
    fast.set_measurement_uncompute(True)

    # Figure 11: the linear CNOT skeleton remains unconditional.  Only the
    # nonlinear carry operation is promoted to a controlled multi-Toffoli, and
    # UMA has the separately controlled sum-write Toffoli.
    q = QuantumRegister(7, "q")
    maj = QuantumCircuit(q, name="AUDIT_CONTROLLED_MAJ")
    core.controlled_maj(maj, q[0], q[1], q[2], q[3], q[4:])
    maj_names = _primitive_names(maj)
    if maj_names[:2] != ["cx", "cx"]:
        raise AssertionError(f"controlled MAJ lost its unconditional CNOT skeleton: {maj_names}")

    uma = QuantumCircuit(q, name="AUDIT_CONTROLLED_UMA")
    core.controlled_uma(uma, q[0], q[1], q[2], q[3], q[4:])
    uma_names = _primitive_names(uma)
    if uma_names[-2:] != ["cx", "cx"] or "ccx" not in uma_names:
        raise AssertionError(f"controlled UMA does not match Figure 11: {uma_names}")

    # The production symbol must be the local fast dual-unary implementation,
    # never the serial low-aux equality decoder.
    binding_module = fast.lc_interval_addsub_unary_gate.__module__
    if binding_module != "eea_circuit_s835_fastdual":
        raise AssertionError(
            "R-side decoder regression: production binding is " + binding_module
        )

    n = 256
    layout = wrapped.shared_eea_layout(n, p=wrapped.SECP256K1_P, T_max=1620)
    if layout.step_aux != 20 or layout.s_qubits != 66:
        raise AssertionError(f"n=256 layout changed: {layout.as_dict()}")
    pa_qubits = 1 + 3 * n + layout.s_qubits
    if pa_qubits != 835:
        raise AssertionError(f"point-addition width={pa_qubits}, expected 835")

    # Maximum R window: the recursively expanded count must remain on the
    # paper's 11(K-k)+O(log n) scale, not the old ~18.9k serial-equality scale.
    k, K = 3, 259
    r_gate = fast.lc_interval_addsub_unary_gate(
        n=n,
        k=k,
        K=K,
        len_width=layout.len_width,
        shift_width=layout.shift_width,
        mode="sub",
        sign_update=True,
        target="work1",
        name="AUDIT_MAX_R_FAST_DUAL_UNARY",
    )
    r_count = _count(r_gate)
    r_paper_leading = 11 * (K - k)
    if int(r_count.get("ccx", 0)) > r_paper_leading + 512:
        raise AssertionError(
            f"R block CCX={r_count.get('ccx',0)} is not on paper scale "
            f"11(K-k)+O(log n)={r_paper_leading}+O(log n)"
        )

    # The corrected K5=n+3 length block must also stay on the paper's
    # 24(K-k)+O(log n) scale after dirty-write / MBU alignment.
    len_gate = fast.len_update_lrp_unary_gate(
        n=n, k=k, K=K, len_width=layout.len_width, name="AUDIT_MAX_LEN_RP"
    )
    len_count = _count(len_gate)
    len_paper_leading = 24 * (K - k)
    if int(len_count.get("ccx", 0)) > len_paper_leading + 512:
        raise AssertionError(
            f"len(r') CCX={len_count.get('ccx',0)} is not on paper scale "
            f"24(K-k)+O(log n)={len_paper_leading}+O(log n)"
        )

    windows = fast.active_windows(n, 4)
    if int(windows["len_update_lrp"][1]) != n + 3:
        raise AssertionError(f"K5 is not n+3: {windows['len_update_lrp']}")

    # The emitted forward and explicit inverse Algorithm-3 blocks must use the
    # same measurement-assisted primitive cost, not a coherent q.inverse()
    # surrogate mixed with a measurement-based JSON.
    fwd = wrapped._algorithm3_step_fastdual_gate(
        n, layout.len_width, layout.shift_width, layout.T_max, layout.step_aux, 1,
        inverse=False,
    )
    inv = wrapped._algorithm3_step_fastdual_gate(
        n, layout.len_width, layout.shift_width, layout.T_max, layout.step_aux, 1,
        inverse=True,
    )
    fwd_count = _count(fwd)
    inv_count = _count(inv)
    for key in ("ccx", "cx", "h", "measure", "reset", "cz"):
        if int(fwd_count.get(key, 0)) != int(inv_count.get(key, 0)):
            raise AssertionError(
                f"forward/inverse cost mismatch for {key}: "
                f"{fwd_count.get(key,0)} != {inv_count.get(key,0)}"
            )

    # Production code must not fall back to the removed Instruction.condition
    # API for measurement feed-forward.
    legacy_hits: list[str] = []
    for path in Path(__file__).parent.glob("*.py"):
        if path.name.startswith("test_") or path.name == "mini_qiskit_runtime.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ".condition =" in text or ".c_if(" in text:
            legacy_hits.append(path.name)
    if legacy_hits:
        raise AssertionError(f"legacy Qiskit conditional API remains in {legacy_hits}")

    report = {
        "passed": True,
        "production_r_decoder_module": binding_module,
        "controlled_maj_primitives": maj_names,
        "controlled_uma_primitives": uma_names,
        "n256": {
            "step_aux": layout.step_aux,
            "s_qubits": layout.s_qubits,
            "point_addition_qubits": pa_qubits,
        },
        "max_r_block": {
            "window": [k, K],
            "ccx": int(r_count.get("ccx", 0)),
            "cx": int(r_count.get("cx", 0)),
            "paper_leading_ccx": r_paper_leading,
        },
        "max_len_rp_block": {
            "window": [k, K],
            "ccx": int(len_count.get("ccx", 0)),
            "cx": int(len_count.get("cx", 0)),
            "paper_leading_ccx": len_paper_leading,
        },
        "step1_forward_inverse": {
            key: int(fwd_count.get(key, 0))
            for key in ("ccx", "cx", "h", "measure", "reset", "cz")
        },
        "opaque_terms": {},
        "stopped_terms": {},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
