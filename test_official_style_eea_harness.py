"""Artifact-derived, phase-clean EEA validation inspired by ecdsa.fail.

The tested values are derived from the recursively lowered Qiskit operation
stream, not from a source-file hash.  Actual outputs are produced by executing
that operation stream through ``PhaseBatchSimulator``; Python modular inversion
is used only as an assertion oracle.
"""

import argparse
import json
from pathlib import Path

from official_style_qiskit_harness import (
    USING_REAL_QISKIT,
    PhaseBatchSimulator,
    bit_index,
    canonical_opstream_bytes,
    circuit_fingerprint,
    derive_case_integers,
    encode_values,
    flatten_circuit_fail_closed,
    write_json,
)
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # type: ignore

import under1000_eea_shared_s835_fastdual_wrapped as shared


def build_circuit(p: int, *, roundtrip: bool):
    n = int(p).bit_length()
    layout = shared.shared_eea_layout(n, p=p)
    X = QuantumRegister(n, "X")
    A = QuantumRegister(n, "A")
    S = QuantumRegister(layout.s_qubits, "S")
    c_fwd = ClassicalRegister(n, "c_fwd")
    if roundtrip:
        c_inv = ClassicalRegister(n, "c_inv")
        qc = QuantumCircuit(X, A, S, c_fwd, c_inv, name="OFFICIAL_STYLE_EEA_ROUNDTRIP")
    else:
        c_inv = None
        qc = QuantumCircuit(X, A, S, c_fwd, name="OFFICIAL_STYLE_EEA_FORWARD")
    fwd = shared.eea_forward_shared_instruction(n, p, T_max=layout.T_max, lazy_definition=True)
    qc.append(fwd, [*X, *A, *S], [*c_fwd])
    if roundtrip:
        inv = shared.eea_inverse_shared_instruction(n, p, T_max=layout.T_max, lazy_definition=True)
        qc.append(inv, [*X, *A, *S], [*c_inv])
    return qc, X, A, S, layout


def case_list(p: int, fingerprint: str, shots: int, include_all: bool) -> list[int]:
    fixed = [1, 2, 3, p // 2 - 1, p // 2, p // 2 + 1, p - 3, p - 2, p - 1]
    derived = derive_case_integers(fingerprint, p, shots, domain=f"eea|p={p}")
    values = list(range(1, p)) if include_all else []
    values.extend(x for x in fixed if 1 <= x < p)
    values.extend(derived)
    return values


def run_one(p: int, shots: int, include_all: bool, roundtrip: bool):
    qc, X, A, S, layout = build_circuit(p, roundtrip=roundtrip)
    ops = flatten_circuit_fail_closed(qc)
    fp = circuit_fingerprint(qc, ops)
    cases = case_list(p, fp, shots, include_all)
    masks = encode_values(cases, len(X))
    initial = {bit_index(qc, q): masks[i] for i, q in enumerate(X)}
    seed_material = canonical_opstream_bytes(qc, ops)
    sim = PhaseBatchSimulator(
        qc.num_qubits,
        qc.num_clbits,
        len(cases),
        seed_material=seed_material,
        initial_qubit_masks=initial,
    )
    sim.run_ops(ops)

    got_x = sim.register_values(qc, X)
    a_zero = sim.register_zero_mask(qc, A)
    s_zero = sim.register_zero_mask(qc, S)
    failures = []
    for i, x in enumerate(cases):
        expected = x if roundtrip else pow(x, -1, p)
        ok = got_x[i] == expected and ((a_zero >> i) & 1) == 1
        if roundtrip:
            ok = ok and ((s_zero >> i) & 1) == 1
        if not ok and len(failures) < 100:
            failures.append({
                "case_index": i,
                "x": x,
                "got_x": got_x[i],
                "expected_x": expected,
                "A_zero": bool((a_zero >> i) & 1),
                "S_zero": bool((s_zero >> i) & 1),
            })
    phase_error_indices = [i for i in range(len(cases)) if (sim.phase >> i) & 1]
    if phase_error_indices:
        failures.append({
            "error": "phase_garbage",
            "count": len(phase_error_indices),
            "first_indices": phase_error_indices[:32],
        })
    passed = not failures
    report = {
        "qiskit_version": __import__("qiskit").__version__ if USING_REAL_QISKIT else "mini-data-model",
        "real_qiskit": USING_REAL_QISKIT,
        "mode": "roundtrip" if roundtrip else "forward",
        "p": p,
        "n": p.bit_length(),
        "T_max": layout.T_max,
        "shots": len(cases),
        "derived_shots_requested": shots,
        "unique_inputs": len(set(cases)),
        "artifact_fingerprint_sha256": fp,
        "lowered_ops": len(ops),
        "measurement_ops": sim.stats.measurement_ops,
        "phase_errors": len(phase_error_indices),
        "passed": passed,
        "failures": failures,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p", type=lambda s: int(s, 0))
    ap.add_argument("--shots", type=int, default=9024)
    ap.add_argument("--include-all", action="store_true")
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--require-real-qiskit", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.require_real_qiskit and not USING_REAL_QISKIT:
        raise RuntimeError("real Qiskit is required but unavailable")
    report = run_one(args.p, args.shots, args.include_all, args.roundtrip)
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in (
        "mode", "p", "n", "T_max", "shots", "unique_inputs",
        "lowered_ops", "measurement_ops", "phase_errors", "passed",
    )}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
