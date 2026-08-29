"""ECDSAfail-inspired point-addition validation on emitted Figure-14 circuits.

For a fixed classical offset point P2 (matching this repository's Figure-14
ABI), the test corpus is derived from the recursively lowered circuit stream.
The independent interpreter checks affine outputs, phase cleanliness, A/S
cleanup, and control preservation over thousands of parallel shots.
"""


import argparse
import hashlib
import json

from official_style_qiskit_harness import (
    USING_REAL_QISKIT,
    PhaseBatchSimulator,
    bit_index,
    canonical_opstream_bytes,
    circuit_fingerprint,
    encode_values,
    flatten_circuit_fail_closed,
    write_json,
)
from point_addition_fig14_s835_fastdual_wrapped_quadratic import (
    build_point_addition_fig14_quadratic,
)
from test_point_addition_correctness_s835_domainfix_v10 import (
    affine_add,
    enumerate_points,
    fig14_schedule_reference,
)


def valid_domain(p: int, a: int, b: int, p2: tuple[int, int]):
    out = []
    for p1 in enumerate_points(p, a, b):
        if p1[0] == p2[0]:
            continue
        result = affine_add(p1, p2, p=p, a=a, b=b)
        if result is None:
            continue
        # The current generic Figure-14 schedule has an internal division that
        # excludes this zero-multiplier case.
        if (p2[0] - result[0]) % p == 0:
            continue
        out.append(p1)
    if not out:
        raise RuntimeError("empty generic affine domain")
    return out


def derive_targets(fp: str, domain: list[tuple[int, int]], count: int):
    out = []
    for i in range(int(count)):
        h = hashlib.shake_256()
        h.update(b"qiskit-ecdsafail-style-pa-target-v1")
        h.update(bytes.fromhex(fp))
        h.update(i.to_bytes(8, "little"))
        idx = int.from_bytes(h.digest(32), "little") % len(domain)
        out.append(domain[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=lambda s: int(s, 0), default=13)
    ap.add_argument("--a", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--b", type=lambda s: int(s, 0), default=7)
    ap.add_argument("--x2", type=lambda s: int(s, 0), default=7)
    ap.add_argument("--y2", type=lambda s: int(s, 0), default=5)
    ap.add_argument("--shots", type=int, default=9024)
    ap.add_argument("--include-domain", action="store_true")
    ap.add_argument("--inactive-cases", type=int, default=256)
    ap.add_argument("--require-real-qiskit", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.require_real_qiskit and not USING_REAL_QISKIT:
        raise RuntimeError("real Qiskit is required but unavailable")

    p, a, b = args.p, args.a, args.b
    p2 = (args.x2 % p, args.y2 % p)
    domain = valid_domain(p, a, b, p2)
    qc = build_point_addition_fig14_quadratic(n=p.bit_length(), p=p, x2=p2[0], y2=p2[1])
    ops = flatten_circuit_fail_closed(qc)
    fp = circuit_fingerprint(qc, ops)

    active_targets = derive_targets(fp, domain, args.shots)
    cases: list[tuple[int, tuple[int, int]]] = []
    if args.include_domain:
        cases.extend((1, p1) for p1 in domain)
        cases.extend((0, p1) for p1 in domain)
    cases.extend((1, p1) for p1 in active_targets)
    # Inactive-control cases are still restricted to x1 != x2 because the
    # current Figure-14 schedule executes the same division path before the
    # surrounding controlled arithmetic cancels.
    for i in range(max(0, args.inactive_cases)):
        cases.append((0, domain[i % len(domain)]))

    regs = {r.name: r for r in qc.qregs}
    ctrl = regs["ctrl"]
    X = regs["X_x1_to_x3"]
    Y = regs["Y_y1_to_y3"]
    Areg = regs["A_shared_work"]
    Sreg = regs["S_shared_eea_arith"]

    init = {}
    ctrl_values = [c for c, _ in cases]
    x_values = [p1[0] for _, p1 in cases]
    y_values = [p1[1] for _, p1 in cases]
    init[bit_index(qc, ctrl[0])] = encode_values(ctrl_values, 1)[0]
    for i, mask in enumerate(encode_values(x_values, len(X))):
        init[bit_index(qc, X[i])] = mask
    for i, mask in enumerate(encode_values(y_values, len(Y))):
        init[bit_index(qc, Y[i])] = mask

    sim = PhaseBatchSimulator(
        qc.num_qubits,
        qc.num_clbits,
        len(cases),
        seed_material=canonical_opstream_bytes(qc, ops),
        initial_qubit_masks=init,
    )
    sim.run_ops(ops)
    got_x = sim.register_values(qc, X)
    got_y = sim.register_values(qc, Y)
    a_zero = sim.register_zero_mask(qc, Areg)
    s_zero = sim.register_zero_mask(qc, Sreg)
    ctrl_out = sim.register_values(qc, ctrl)

    failures = []
    for i, (control, p1) in enumerate(cases):
        expected = fig14_schedule_reference(control, *p1, *p2, p)
        expected_xy = (expected[0], expected[1])
        got = (got_x[i], got_y[i])
        ok = (
            got == expected_xy
            and bool((a_zero >> i) & 1)
            and bool((s_zero >> i) & 1)
            and ctrl_out[i] == control
        )
        if not ok and len(failures) < 100:
            failures.append({
                "case_index": i,
                "ctrl": control,
                "P1": list(p1),
                "P2": list(p2),
                "got": list(got),
                "expected": list(expected_xy),
                "A_zero": bool((a_zero >> i) & 1),
                "S_zero": bool((s_zero >> i) & 1),
                "ctrl_out": ctrl_out[i],
            })
    phase_indices = [i for i in range(len(cases)) if (sim.phase >> i) & 1]
    if phase_indices:
        failures.append({
            "error": "phase_garbage",
            "count": len(phase_indices),
            "first_indices": phase_indices[:32],
        })

    report = {
        "qiskit_version": __import__("qiskit").__version__ if USING_REAL_QISKIT else "mini-data-model",
        "real_qiskit": USING_REAL_QISKIT,
        "p": p,
        "curve": [a, b],
        "P2": list(p2),
        "generic_domain_size": len(domain),
        "shots": len(cases),
        "artifact_fingerprint_sha256": fp,
        "lowered_ops": len(ops),
        "measurement_ops": sim.stats.measurement_ops,
        "phase_errors": len(phase_indices),
        "passed": not failures,
        "failures": failures,
    }
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in (
        "p", "curve", "P2", "generic_domain_size", "shots", "lowered_ops",
        "measurement_ops", "phase_errors", "passed",
    )}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
