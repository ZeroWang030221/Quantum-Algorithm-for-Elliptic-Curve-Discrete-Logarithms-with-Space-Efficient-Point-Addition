# Quantum Algorithm for Elliptic Curve Discrete Logarithms with Space-Efficient Point Addition

This repository contains Qiskit code for resource estimation of the space-efficient quantum modular inversion and affine point-addition circuits used in elliptic-curve discrete logarithm settings.

The current codebase is centered on three workflows:

1. chunkwise recursive counting of the modular inversion circuit;
2. optional local NCT-template optimization of the circuit;
3. compiled blockwise resource estimation of the wrapped affine point-addition circuit.


---

## Repository structure

```text
.
├── README.md
│
├── eea_model/: original classical EEA reference implementation used for algorithm prototyping and correctness validation.
│
├── run_eea_s835_fastdual_recursive_chunks_checkpoint.py
├── run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py
├── count_s835_fastdual_wrapped_point_addition_blocks_compiled.py
│
├── eea_circuit.py
├── eea_circuit_s835_fastdual.py
├── eea_circuit_s835_lowaux.py
├── eea_circuit_updated.py
├── under1000_eea_shared_s835_fastdual_wrapped.py
├── under1000_modular_arithmetic_base.py
│
├── point_addition_fig14_s835_fastdual_wrapped_quadratic.py
├── quadratic_fig15_inplace_s835_fastdual_wrapped.py
├── quadratic_gidney_arithmetic.py
├── quadratic_lazy_instruction.py
├── quadratic_modular_arithmetic.py
├── quadratic_squ_minus.py
│
├── ccx_recursive_block_counter.py
├── nct_template_segment_optimizer.py
├── run_eea_s835_fastdual_recursive_chunks_checkpoint.py
├── run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py
├── count_s835_fastdual_wrapped_point_addition_blocks_compiled.py
│
├── test_paper_alignment_gate_structure.py
├── test_fastdual_interval_selector.py
├── test_explicit_measurement_inverse_blocks.py
├── test_reachable_step_forward_inverse.py
│
├── extended_semantic_sweep.py
├── test_p419_per_step_oracle.py
├── test_generic_per_step_oracle.py
├── test_p419_x153_quantum_semantics.py
├── test_exhaustive_small_prime_matrix.py
├── hash_derived_large_prime_sweep.py
│
├── test_terminal_shift_epoch_835.py
├── test_terminal_padding_width_matrix.py
│
├── test_small_field_point_addition_semantics.py
├── test_small_field_point_addition_coherence.py
├── official_style_qiskit_harness.py
├── test_official_style_eea_harness.py
├── test_official_style_point_add_harness.py
├── test_official_style_point_add_roundtrip.py
├── test_exhaustive_measurement_transcripts_tiny.py
├── test_harness_mutation_sensitivity.py
│
├── test_no_opaque_counts.py
├── test_no_opaque_matrix.py
├── test_eea_strict_main.py
├── test_point_addition_strict_main.py
│
├── test_coherent_eea_pair_matrix.py
├── test_secp256k1_schedule_witness.py
│
├── run_all_real_qiskit_validation.ps1
├── run_all_real_qiskit_validation.sh
├── run_extended_real_qiskit_validation.ps1
└── run_extended_real_qiskit_validation.sh
```

### Main entry scripts

- `run_eea_s835_fastdual_recursive_chunks_checkpoint.py`  
  Counts the EEA Algorithm-3 steps recursively, in checkpointed chunks.

- `run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py`  
  Same EEA counting workflow, but with local NCT-template optimization.

- `count_s835_fastdual_wrapped_point_addition_blocks_compiled.py`  
  Counts the wrapped point-addition circuit by recursively counting reusable compiled subblocks and assembling the repeated arithmetic components with exact multiplicities.

### EEA files

- `eea_circuit_s835_fastdual.py`:	Main implementation of the production EEA circuit.
- `eea_circuit_s835_lowaux.py`:	Low-auxiliary helper routines used by the main implementation.
- `eea_circuit_updated.py`:	Shared EEA building blocks and recursive resource-counting utilities.
- `eea_circuit.py`:	Backward-compatibility wrapper for tests.

### Point-addition and arithmetic files

- `point_addition_fig14_s835_fastdual_wrapped_quadratic.py`: builds the wrapped affine point-addition circuit corresponding to the Fig.14 schedule.
- `quadratic_fig15_inplace_s835_fastdual_wrapped.py`: builds the Fig.15 in-place division and in-place multiplication structure with EEA, multiplication, measurement, reset, and feed-forward phase correction.
- `quadratic_modular_arithmetic.py`: modular addition/subtraction, multiplication, inverse multiplication, doubling, and halving instructions used by the point-addition counter.
- `quadratic_gidney_arithmetic.py`: Gidney-style arithmetic primitives and measurement and feed-forward helpers used by the quadratic modular arithmetic layer.
- `quadratic_squ_minus.py`: square-minus block used in the affine point-addition schedule.
- `under1000_eea_shared_s835_fastdual_wrapped.py`: shared EEA wrapper and helper used by the point-addition circuit.
- `under1000_modular_arithmetic_base.py`: small shared modular-arithmetic utilities.

### Counting and optimization utilities

- `ccx_recursive_block_counter.py`: recursive counter for Qiskit circuits, with policies for MCX expansion and SWAP expansion.
- `nct_template_segment_optimizer.py`: local template-based optimizer for `{X, CX, CCX}` segments.

---

## Requirements

Recommended environment:

- Python 3.10+
- Qiskit

Install the main dependency with:

```bash
python -m pip install --upgrade pip
python -m pip install qiskit
```

---

## Quick start

Run the test suite:

```bash
bash ./run_all_real_qiskit_validation.sh
bash ./run_extended_real_qiskit_validation.sh
FULL=1 bash ./run_extended_real_qiskit_validation.sh
```

---

## 1. EEA recursive chunk counting

The standard EEA counting entry point is:

```bash
python run_eea_s835_fastdual_recursive_chunks_checkpoint.py \
  --n 192 \
  --chunk-size 25 \
  --measurement-uncompute \
  --resume \
  --workdir eea_s835_fastdual_chunks25 \
  --out eea_s835_fastdual_algorithm3_recursive_chunks_n192_measurement.json
```

Important arguments:

- `--n`: bit width.
- `--T-max`: optional override for the number of Algorithm-3 steps; by default the value from `eea.get_n_config(n)` is used.
- `--chunk-size`: number of Algorithm-3 steps counted per checkpoint chunk.
- `--aux-size`: optional override for the helper-qubit pool; if omitted, the layout helper size is computed automatically.
- `--measurement-uncompute`: enables measurement-based uncomputation in the counted EEA blocks.
- `--resume`: reuses existing non-empty chunk JSON files in `--workdir`.
- `--workdir`: directory for per-chunk checkpoint files.
- `--out`: cumulative JSON summary written after every chunk.

The script writes per-chunk files such as:

```text
eea_s835_fastdual_chunks25/eea_s835_fastdual_n192_T0001_0025.json
```

and a cumulative output JSON containing fields such as:

```text
mode
n
T_max
num_qubits
len_width
shift_width
aux_size
measurement_based
ops
chunks
elapsed_s_so_far
```

---

## 2. EEA counting with NCT-template optimization

The optimized counting entry point is:

```bash
python run_eea_s835_fastdual_recursive_chunks_checkpoint_nctopt.py \
  --n 128 \
  --chunk-size 25 \
  --measurement-uncompute \
  --templates small-nct \
  --rounds 1 \
  --max-nct-segment-gates 40 \
  --segment-timeout-s 10 \
  --timeout-mode auto \
  --resume \
  --workdir eea_s835_fastdual_chunks_nctopt_failopen_r1_128_seg40_to10 \
  --out eea_s835_fastdual_algorithm3_recursive_chunks_n128_measurement_nctopt_failopen_r1_seg40_to10.json
```

This workflow attempts local template optimization on  `{X, CX, CCX}` segments.  It is designed as a bounded fail-open counter: if an optimized step times out or raises an exception, that step is counted exactly without template rounds and then checkpointed, so the final reported counts remain complete.

Useful arguments in addition to the standard EEA arguments:

- `--templates {small-nct,all-nct}`: template library selection.
- `--rounds`: number of template-optimization rounds.
- `--max-nct-segment-gates`: maximum size of a reversible segment sent to template optimization.
- `--max-nct-segment-qubits`: maximum number of qubits in a segment.
- `--segment-timeout-s`: timeout for individual segment optimization.
- `--step-timeout-s`: timeout for a whole Algorithm-3 step before falling back to unchanged counting.
- `--fallback-step-timeout-s`: timeout for the exact fallback count.
- `--force`: recompute even if step/chunk checkpoints already exist.
- `--ignore-policy-mismatch`: reuse old checkpoints even when the optimization policy differs; this is mainly for debugging.

The optimized workflow writes both step-level checkpoints under:

```text
<workdir>/steps/
```

and chunk-level summaries under:

```text
<workdir>/
```

---

## 3. Wrapped point-addition compiled blockwise counting

The point-addition counter depends on an EEA Algorithm-3 JSON produced by one of the EEA workflows above.  The `--n` value of the point-addition counter should match the `n` field in the EEA JSON.

Example for `n=64`:

```bash
python run_eea_s835_fastdual_recursive_chunks_checkpoint.py \
  --n 64 \
  --chunk-size 25 \
  --measurement-uncompute \
  --resume \
  --workdir eea_s835_fastdual_chunks25_n64 \
  --out eea_s835_fastdual_algorithm3_recursive_chunks_n64_measurement.json
```

Then run:

```bash
python count_s835_fastdual_wrapped_point_addition_blocks_compiled.py \
  --n 64 \
  --eea-steps-json eea_s835_fastdual_algorithm3_recursive_chunks_n64_measurement.json \
  --out point_addition_s835_fastdual_wrapped_blocks_compiled_counts_n64.json
```

Example for the optimized `n=128` EEA output:

```bash
python count_s835_fastdual_wrapped_point_addition_blocks_compiled.py \
  --n 128 \
  --eea-steps-json eea_s835_fastdual_algorithm3_recursive_chunks_n128_measurement_nctopt_failopen_r1_seg40_to10.json \
  --out point_addition_s835_fastdual_wrapped_blocks_compiled_counts_n128.json
```

Important arguments:

- `--n`: bit width.
- `--p`: modulus; defaults to the secp256k1 prime.
- `--s-qubits`: optional override for the shared EEA arithmetic register size.
- `--point-constant {secp256k1-generator,zero,custom}`: point constant selection for the Fig.14 constant-coordinate updates.
- `--x2`, `--y2`: custom point coordinates; required when `--point-constant custom` is used.
- `--eea-steps-json`: JSON file containing recursive Algorithm-3 EEA counts.
- `--allow-eea-n-mismatch`: debug-only override allowing the EEA JSON `n` to differ from the requested `--n`.
- `--mcx-policy {clean-vchain,keep}`: MCX expansion policy for recursive counting.
- `--validate-full-mul`: for small `n`, recursively count full multiplication/squaring definitions and compare them with the assembled block counts.
- `--out`: output JSON path.

The output report includes:

```text
counting_mode
n
p
point_constant_kind
qiskit_width_report
eea_meta
block_summaries
raw_block_counters
key_ccx
validation
elapsed_s
```

The point-addition counter builds reusable Qiskit circuits, recursively counts them in the `{CCX, CX, X}` basis, and then assembles larger repeated blocks such as multiplication, inverse multiplication, in-place division, in-place multiplication, square-minus, and the total Fig.14 point-addition block.

---

## Tests

The validation suite uses emitted Qiskit circuit definitions to produce the result under test. The classical EEA and affine point-addition implementations are used only to compute independent expected values.

The different tests target different failure modes.

### 1. Paper/circuit structural alignment

Goal: verify that the production circuit has the intended paper-level structure rather than only the correct final classical answer.

Relevant tests:

```text
test_paper_alignment_gate_structure.py
test_fastdual_interval_selector.py
test_explicit_measurement_inverse_blocks.py
test_reachable_step_forward_inverse.py
test_eea_strict_main.py
test_point_addition_strict_main.py
```

These checks include:

- the Figure-11 matched MAJ/UMA primitive structure;
- the fast dual-unary R-side implementation;
- the active-window formulas;
- explicit measurement-assisted inverse blocks;
- forward/inverse consistency of reachable Algorithm-3 steps;
- the `n = 256` 835-qubit register layout;
- Figure-14/Figure-15 operation ordering;
- recursively compiled arithmetic-block assembly.

The fast dual-unary selector has been compared with the corresponding direct ripple implementation over **15,360 tested basis-state configurations**, all of which pass.

### 2. Exhaustive EEA test for p = 419

Goal: verify arithmetic correctness and fixed-schedule control flow for every nonzero input of the reported small field.

Run:

```bash
python extended_semantic_sweep.py 419 --all --require-real-qiskit
```

Because `p = 419` is a 9-bit prime, this test uses the complete 9-bit schedule:

```text
n = 9
T_max = 60
```

Current result:

```text
418 / 418 nonzero inputs passed
```

### 3. Step-by-step Algorithm-3 comparison

Goal: locate an incorrect state transition immediately instead of checking only the final modular inverse.

Run:

```bash
python test_p419_per_step_oracle.py --require-real-qiskit
```

After each Algorithm-3 step, the state encoded by the Qiskit circuit is decoded into the corresponding EEA variables and phase/length information and compared with an independent classical implementation advanced through the same step.

Current result for `p = 419`:

```text
19,528 / 19,528 step-state comparisons passed
```

This test includes the previously problematic `x = 1`, step-34 region.

### 4. Coherent EEA forward/inverse regression

Goal: verify that measurement-assisted uncomputation and classical feed-forward preserve relative phase between different input branches, while also restoring the input and cleaning the workspace after the inverse.

Run:

```bash
python test_p419_x153_quantum_semantics.py \
  --shots 8 \
  --require-real-qiskit \
  --verbose
```

The test uses the two input branches `x = 153` and `x = 155`. It executes the emitted dynamic circuit, including Hadamard gates, mid-circuit measurements, resets, and feed-forward `Z/CZ` corrections.

In the current real-Qiskit run:

- `153 -> 241` and `155 -> 246` in the forward map;
- all eight tested coherent forward/inverse trajectories have fidelity 1;
- the relative phase is preserved;
- `X` is restored by the round trip;
- every `A/S` workspace qubit is returned to zero.

### 5. n = 256 fixed-schedule terminal behavior

Goal: verify that branches that finish their EEA arithmetic early remain reversible through the rest of the common `T_max(256) = 1620` schedule.

Run:

```bash
python test_terminal_shift_epoch_835.py
```

For `n = 256`:

```text
T_max                = 1620
maximum padding      = 596 steps
reachable test cases = 149
logical qubits       = 835
```

All 149 reachable terminal-padding depths pass, including the low-word wrap boundary and the maximum 596-step padding case.

The broader width matrix covers widths from 4 through 512 bits and verifies terminal canonicalization and selected forward/inverse round trips.

### 6. Complete Figure-14 point addition

Goal: verify that the modular-arithmetic, Figure-15, and coordinate-update components remain correct after composition into the complete Figure-14 point-addition circuit.

Relevant tests:

```text
test_small_field_point_addition_semantics.py
test_small_field_point_addition_coherence.py
test_official_style_point_add_harness.py
test_official_style_point_add_roundtrip.py
```

Current small-field semantic result:

```text
68 / 68 generic affine cases passed
```

The validation additionally checks:

- active and inactive control branches;
- workspace cleanup;
- a coherent two-branch point-addition example;
- add-P followed by add-(-P) round trips;
- representative larger-prime regressions.

The real-Qiskit base suite also passes the current 16-bit and 32-bit regression sets:

```text
p = 65521       : 41 / 41
p = 4294967291  : 21 / 21
```

### 7. Fail-closed recursive decomposition

Goal: ensure that an undecomposed arithmetic block, unsupported gate, or stopped subcircuit cannot be silently treated as one elementary operation.

Relevant tests:

```text
test_no_opaque_counts.py
test_no_opaque_matrix.py
```

The recursive inspection enters supported circuit definitions and control-flow bodies and reports failure if an unresolved term is encountered.

The current tested paths report:

```text
opaque_terms  = {}
stopped_terms = {}
```

The full no-opaque matrix also includes representative Algorithm-3 steps up to `T = 1620` at `n = 256`.

---

## Current validation status

The base validation suite has been run under **Qiskit 2.4.1** and completed successfully.

Representative completed results include:

| Test | Result |
|---|---:|
| `p = 419`, all nonzero inputs, complete `T_max(9) = 60` schedule | 418 / 418 |
| `p = 419` per-step state comparisons | 19,528 / 19,528 |
| coherent EEA 153/155 regression | 8 / 8 trajectories, fidelity 1 |
| fast dual-unary selector | 15,360 / 15,360 |
| `n = 256` reachable terminal-padding depths | 149 / 149 |
| complete small-field Figure-14 cases | 68 / 68 |
| 16-bit regression (`p = 65521`) | 41 / 41 |
| 32-bit regression (`p = 4294967291`) | 21 / 21 |
| strict EEA groups | 10 / 10 |
| strict point-addition groups | 4 / 4 |
| recursive no-opaque base audit | passed |

The extended suite additionally contains:

- a 37-prime exhaustive EEA matrix;
- per-step matrices over several primes;
- ECDSAfail-style forward and forward/inverse tests;
- complete Figure-14 artifact-derived tests;
- terminal-padding width matrices;
- measurement-transcript tests;
- mutation-sensitivity tests;
- larger no-opaque matrices.

The `-Full` / `FULL=1` mode is intentionally long-running and should be treated separately from the completed base validation.

---

## Reproducing the main statistics

Our paper reports numerical resource-estimation results for:

```text
n = 64, 128, 160, 192, 224, 256, 384, 512
```

A typical workflow is:

1. run the EEA Algorithm-3 counter for a given `n`;
2. optionally run the NCT-optimized version for the same `n`;
3. pass the resulting EEA JSON to the wrapped point-addition counter;
4. collect the `key_ccx`, `block_summaries`, and `qiskit_width_report` fields from the output report.

For large widths, use `--resume` and keep the `--workdir` directories, since chunk and step checkpoints are meant to support interrupted long runs.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{luo2026quantumalgorithmellipticcurve,
      title={Quantum Algorithm for Elliptic Curve Discrete Logarithms with Space-Efficient Point Addition}, 
      author={Han Luo and Ziyi Yang and Jingquan Luo and Ziruo Wang and Yuexin Su and Xiaoming Sun and Lvzhou Li and Tongyang Li},
      year={2026},
      eprint={2607.13816},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2607.13816}, 
}
```
