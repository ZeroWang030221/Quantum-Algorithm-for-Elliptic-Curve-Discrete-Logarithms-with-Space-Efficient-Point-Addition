#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p validation_real
python - <<'PY'
import qiskit
print('Qiskit:', qiskit.__version__)
PY

echo '=== Paper/code structural alignment ==='
python test_paper_alignment_gate_structure.py | tee validation_real/paper_alignment_gate_structure.log
python test_fastdual_interval_selector.py | tee validation_real/fastdual_interval_selector.log
python test_explicit_measurement_inverse_blocks.py | tee validation_real/explicit_measurement_inverse_blocks.log
python test_reachable_step_forward_inverse.py 37 | tee validation_real/reachable_step_forward_inverse_p37.log

echo '=== EEA semantics and coherence ==='
python extended_semantic_sweep.py 419 --all --require-real-qiskit --out validation_real/p419_all.json
python test_p419_per_step_oracle.py --require-real-qiskit --out validation_real/p419_per_step.json
python test_p419_x153_quantum_semantics.py --shots 8 --require-real-qiskit --verbose | tee validation_real/p419_coherent.log

echo '=== Terminal padding and complete small-field point addition ==='
python test_terminal_shift_epoch_835.py | tee validation_real/terminal_padding.log
python test_small_field_point_addition_semantics.py | tee validation_real/small_pa_semantics.log
python test_small_field_point_addition_coherence.py --shots 8 --out validation_real/small_pa_coherence.json | tee validation_real/small_pa_coherence.log

echo '=== Large-prime basis tests and fail-closed lowering ==='
python hash_derived_large_prime_sweep.py 65521 --count 32 --require-real-qiskit --out validation_real/hash_p16.json
python hash_derived_large_prime_sweep.py 4294967291 --count 12 --require-real-qiskit --out validation_real/hash_p32.json
python test_no_opaque_counts.py | tee validation_real/no_opaque_counts.log

echo '=== Strict block/layout/count assembly tests ==='
python test_eea_strict_main.py | tee validation_real/strict_eea.log
python test_point_addition_strict_main.py --skip-large-primes | tee validation_real/strict_point_addition.log

echo '[PASS] Base real-Qiskit validation completed.'
