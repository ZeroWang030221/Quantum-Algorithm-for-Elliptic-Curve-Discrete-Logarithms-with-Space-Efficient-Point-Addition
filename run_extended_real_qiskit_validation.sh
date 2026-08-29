#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p validation_extended_real
python -c 'import qiskit; print("Qiskit:", qiskit.__version__)'

bash ./run_all_real_qiskit_validation.sh

python ./test_official_style_eea_harness.py 419 --shots 9024 --include-all --require-real-qiskit --out ./validation_extended_real/official_eea_forward_p419.json
python ./test_official_style_eea_harness.py 419 --shots 9024 --include-all --roundtrip --require-real-qiskit --out ./validation_extended_real/official_eea_roundtrip_p419.json
python ./test_official_style_eea_harness.py 65521 --shots 9024 --roundtrip --require-real-qiskit --out ./validation_extended_real/official_eea_roundtrip_p16.json

python ./test_official_style_point_add_harness.py --p 13 --a 0 --b 7 --x2 7 --y2 5 --shots 9024 --include-domain --inactive-cases 256 --require-real-qiskit --out ./validation_extended_real/official_pa_p13.json
python ./test_official_style_point_add_harness.py --p 31 --a 0 --b 7 --x2 0 --y2 10 --shots 9024 --include-domain --inactive-cases 512 --require-real-qiskit --out ./validation_extended_real/official_pa_p31.json
python ./test_official_style_point_add_roundtrip.py --p 31 --a 0 --b 7 --x2 0 --y2 10 --shots 9024 --include-domain --require-real-qiskit --out ./validation_extended_real/official_pa_roundtrip_p31.json

python ./test_exhaustive_small_prime_matrix.py --max-prime 127 --extra 251,257,419,509,1021,2039,4093 --jobs 4 --require-real-qiskit --out ./validation_extended_real/exhaustive_small_prime_matrix.json
for p in 37 97 251 419; do
  python ./test_generic_per_step_oracle.py "$p" --require-real-qiskit --out "./validation_extended_real/per_step_p${p}.json"
done
python ./test_terminal_padding_width_matrix.py --out ./validation_extended_real/terminal_padding_width_matrix.json
python ./test_exhaustive_measurement_transcripts_tiny.py --out ./validation_extended_real/exhaustive_measurement_transcripts_tiny.json
python ./test_harness_mutation_sensitivity.py --out ./validation_extended_real/harness_mutation_sensitivity.json
python ./test_no_opaque_matrix.py --out ./validation_extended_real/no_opaque_matrix.json

if [[ "${FULL:-0}" == "1" ]]; then
  python ./test_no_opaque_matrix.py --full --out ./validation_extended_real/no_opaque_matrix_full.json
  python ./test_secp256k1_schedule_witness.py --require-real-qiskit --out ./validation_extended_real/secp256k1_1536_witness.json
  python ./test_coherent_eea_pair_matrix.py --pairs 4 --shots 1 --require-real-qiskit --out ./validation_extended_real/coherent_pair_matrix.json
  python ./hash_derived_large_prime_sweep.py 18446744073709551557 --count 16 --require-real-qiskit --out ./validation_extended_real/hash_p64.json
fi

echo '[PASS] Extended validation suite completed.'
