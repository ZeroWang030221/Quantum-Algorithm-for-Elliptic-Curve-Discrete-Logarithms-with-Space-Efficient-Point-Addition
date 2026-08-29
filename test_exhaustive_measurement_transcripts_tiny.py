"""Exercise dynamic measurement transcripts on tiny emitted EEA circuits.

For circuits with at most ``--max-exhaustive-measurements`` measurements, every
transcript is enumerated.  The current paper-aligned unary implementation uses
measurement-assisted AND uncomputation throughout the decoder, so even p=3 has
thousands of measurements.  In that regime exhaustive enumeration is
mathematically infeasible; the test instead uses a deterministic coverage set:
all-zero/all-one/alternating records, evenly spaced single-bit perturbations,
and SHAKE-derived records tied to the actual lowered operation stream.

Each record is replayed on two distinct basis branches.  The test checks
classical semantics, A/S cleanup, absolute phase cleanliness, and equal phase
between the two branches.
"""

import argparse
import hashlib
import json
from typing import Iterable

from official_style_qiskit_harness import (
    PhaseBatchSimulator,
    bit_index,
    canonical_opstream_bytes,
    encode_values,
    flatten_circuit_fail_closed,
    write_json,
)
try:
    import qiskit  # type: ignore
except Exception:
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # type: ignore
import under1000_eea_shared_s835_fastdual_wrapped as shared


class TranscriptListStream:
    """Return one measurement mask for a fixed list of paired transcripts."""

    def __init__(self, transcripts: list[int], measurements: int):
        self.records = [int(t) for t in transcripts]
        self.measurements = int(measurements)
        self.position = 0

    def next_mask(self, label: bytes = b"hmr") -> int:
        j = self.position
        self.position += 1
        if j >= self.measurements:
            raise AssertionError("circuit requested more HMR masks than counted")
        mask = 0
        for t, record in enumerate(self.records):
            if (record >> j) & 1:
                mask |= (1 << (2 * t)) | (1 << (2 * t + 1))
        return mask


def build(p: int, *, roundtrip: bool):
    n = p.bit_length()
    layout = shared.shared_eea_layout(n, p=p)
    X = QuantumRegister(n, "X")
    A = QuantumRegister(n, "A")
    S = QuantumRegister(layout.s_qubits, "S")
    c_fwd = ClassicalRegister(n, "c_fwd")
    registers = [X, A, S, c_fwd]
    if roundtrip:
        c_inv = ClassicalRegister(n, "c_inv")
        registers.append(c_inv)
    else:
        c_inv = None
    qc = QuantumCircuit(*registers)
    qc.append(
        shared.eea_forward_shared_instruction(n, p, T_max=layout.T_max, lazy_definition=True),
        [*X, *A, *S],
        [*c_fwd],
    )
    if roundtrip:
        qc.append(
            shared.eea_inverse_shared_instruction(n, p, T_max=layout.T_max, lazy_definition=True),
            [*X, *A, *S],
            [*c_inv],
        )
    return qc, X, A, S, layout


def _coverage_records(
    measurements: int,
    seed_material: bytes,
    *,
    max_exhaustive_measurements: int,
    hash_records: int,
    single_bit_positions: int,
) -> tuple[str, list[int]]:
    m = int(measurements)
    if m <= int(max_exhaustive_measurements):
        return "exhaustive", list(range(1 << m))

    mask = (1 << m) - 1
    records: set[int] = {
        0,
        mask,
        sum(1 << j for j in range(0, m, 2)),
        sum(1 << j for j in range(1, m, 2)),
    }

    take = min(m, max(0, int(single_bit_positions)))
    if take:
        positions = sorted({round(i * (m - 1) / max(1, take - 1)) for i in range(take)})
        for j in positions:
            records.add(1 << j)
            records.add(mask ^ (1 << j))

    shake = hashlib.shake_256(seed_material + b"|measurement-transcript-coverage|")
    byte_width = (m + 7) // 8
    for _ in range(max(0, int(hash_records))):
        records.add(int.from_bytes(shake.digest(byte_width), "little") & mask)

    return "deterministic-coverage", sorted(records)


def run_case(
    p: int,
    x0: int,
    x1: int,
    *,
    roundtrip: bool,
    max_exhaustive_measurements: int,
    hash_records: int,
    single_bit_positions: int,
):
    qc, X, A, S, layout = build(p, roundtrip=roundtrip)
    ops = flatten_circuit_fail_closed(qc)
    measurements = sum(op.kind == "measure" for op in ops)
    seed_material = canonical_opstream_bytes(qc, ops)
    coverage_mode, transcripts = _coverage_records(
        measurements,
        seed_material,
        max_exhaustive_measurements=max_exhaustive_measurements,
        hash_records=hash_records,
        single_bit_positions=single_bit_positions,
    )
    cases = [x for _ in transcripts for x in (x0, x1)]
    init = {}
    for i, mask in enumerate(encode_values(cases, len(X))):
        init[bit_index(qc, X[i])] = mask
    sim = PhaseBatchSimulator(
        qc.num_qubits,
        qc.num_clbits,
        len(cases),
        seed_material=seed_material,
        initial_qubit_masks=init,
    )
    sim.rng = TranscriptListStream(transcripts, measurements)
    sim.run_ops(ops)

    got = sim.register_values(qc, X)
    a_zero = sim.register_zero_mask(qc, A)
    s_zero = sim.register_zero_mask(qc, S)
    failures = []
    absolute_phase_errors = 0
    relative_phase_errors = 0
    for t, record in enumerate(transcripts):
        i0, i1 = 2 * t, 2 * t + 1
        expected0 = x0 if roundtrip else pow(x0, -1, p)
        expected1 = x1 if roundtrip else pow(x1, -1, p)
        ok = (
            got[i0] == expected0
            and got[i1] == expected1
            and bool((a_zero >> i0) & 1)
            and bool((a_zero >> i1) & 1)
        )
        if roundtrip:
            ok = ok and bool((s_zero >> i0) & 1) and bool((s_zero >> i1) & 1)
        ph0 = (sim.phase >> i0) & 1
        ph1 = (sim.phase >> i1) & 1
        absolute_phase_errors += ph0 + ph1
        relative_phase_errors += ph0 ^ ph1
        if (not ok or ph0 or ph1) and len(failures) < 32:
            failures.append({
                "record_index": t,
                "record_hex": hex(record),
                "got": [got[i0], got[i1]],
                "expected": [expected0, expected1],
                "A_zero": [bool((a_zero >> i0) & 1), bool((a_zero >> i1) & 1)],
                "S_zero": [bool((s_zero >> i0) & 1), bool((s_zero >> i1) & 1)],
                "phase": [ph0, ph1],
            })
    return {
        "p": p,
        "mode": "roundtrip" if roundtrip else "forward",
        "coverage_mode": coverage_mode,
        "inputs": [x0, x1],
        "T_max": layout.T_max,
        "measurement_ops": measurements,
        "transcripts_tested": len(transcripts),
        "basis_branches": 2 * len(transcripts),
        "absolute_phase_errors": absolute_phase_errors,
        "relative_phase_errors": relative_phase_errors,
        "passed": not failures,
        "failures": failures,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-exhaustive-measurements", type=int, default=16)
    ap.add_argument("--hash-records", type=int, default=1024)
    ap.add_argument("--single-bit-positions", type=int, default=256)
    args = ap.parse_args()
    kwargs = dict(
        max_exhaustive_measurements=args.max_exhaustive_measurements,
        hash_records=args.hash_records,
        single_bit_positions=args.single_bit_positions,
    )
    tests = [
        run_case(3, 1, 2, roundtrip=False, **kwargs),
        run_case(3, 1, 2, roundtrip=True, **kwargs),
        run_case(5, 1, 3, roundtrip=False, **kwargs),
    ]
    report = {
        "tests": len(tests),
        "transcripts_tested": sum(t["transcripts_tested"] for t in tests),
        "basis_branches": sum(t["basis_branches"] for t in tests),
        "passed": all(t["passed"] for t in tests),
        "results": tests,
    }
    write_json(args.out, report)
    print(json.dumps({
        "tests": report["tests"],
        "transcripts_tested": report["transcripts_tested"],
        "basis_branches": report["basis_branches"],
        "passed": report["passed"],
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
