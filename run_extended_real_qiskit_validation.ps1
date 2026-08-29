param(
    [switch]$Full
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force validation_extended_real | Out-Null

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 0
    try {
        & python @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($exitCode -ne 0) {
        throw ("python " + ($Arguments -join " ") + " failed with exit code $exitCode.")
    }
}

Invoke-PythonChecked -Arguments @("-c", "import qiskit; print('Qiskit:', qiskit.__version__)")

Write-Host "=== Existing Paper-16 validation ==="
& powershell -ExecutionPolicy Bypass -File .\run_all_real_qiskit_validation.ps1
if ($LASTEXITCODE -ne 0) {
    throw "run_all_real_qiskit_validation.ps1 failed with exit code $LASTEXITCODE."
}

Write-Host "=== ECDSAfail-style EEA: 9,024 artifact-derived trajectories ==="
Invoke-PythonChecked -Arguments @(".\test_official_style_eea_harness.py", "419", "--shots", "9024", "--include-all", "--require-real-qiskit", "--out", ".\validation_extended_real\official_eea_forward_p419.json")
Invoke-PythonChecked -Arguments @(".\test_official_style_eea_harness.py", "419", "--shots", "9024", "--include-all", "--roundtrip", "--require-real-qiskit", "--out", ".\validation_extended_real\official_eea_roundtrip_p419.json")
Invoke-PythonChecked -Arguments @(".\test_official_style_eea_harness.py", "65521", "--shots", "9024", "--roundtrip", "--require-real-qiskit", "--out", ".\validation_extended_real\official_eea_roundtrip_p16.json")

Write-Host "=== ECDSAfail-style full Figure-14 tests ==="
Invoke-PythonChecked -Arguments @(".\test_official_style_point_add_harness.py", "--p", "13", "--a", "0", "--b", "7", "--x2", "7", "--y2", "5", "--shots", "9024", "--include-domain", "--inactive-cases", "256", "--require-real-qiskit", "--out", ".\validation_extended_real\official_pa_p13.json")
Invoke-PythonChecked -Arguments @(".\test_official_style_point_add_harness.py", "--p", "31", "--a", "0", "--b", "7", "--x2", "0", "--y2", "10", "--shots", "9024", "--include-domain", "--inactive-cases", "512", "--require-real-qiskit", "--out", ".\validation_extended_real\official_pa_p31.json")
Invoke-PythonChecked -Arguments @(".\test_official_style_point_add_roundtrip.py", "--p", "31", "--a", "0", "--b", "7", "--x2", "0", "--y2", "10", "--shots", "9024", "--include-domain", "--require-real-qiskit", "--out", ".\validation_extended_real\official_pa_roundtrip_p31.json")

Write-Host "=== Exhaustive small-prime and per-step matrices ==="
Invoke-PythonChecked -Arguments @(".\test_exhaustive_small_prime_matrix.py", "--max-prime", "127", "--extra", "251,257,419,509,1021,2039,4093", "--jobs", "4", "--require-real-qiskit", "--out", ".\validation_extended_real\exhaustive_small_prime_matrix.json")
foreach ($p in 37,97,251,419) {
    Invoke-PythonChecked -Arguments @(".\test_generic_per_step_oracle.py", "$p", "--require-real-qiskit", "--out", ".\validation_extended_real\per_step_p$p.json")
}

Write-Host "=== Terminal, transcript, mutation, and opaque audits ==="
Invoke-PythonChecked -Arguments @(".\test_terminal_padding_width_matrix.py", "--out", ".\validation_extended_real\terminal_padding_width_matrix.json")
Invoke-PythonChecked -Arguments @(".\test_exhaustive_measurement_transcripts_tiny.py", "--out", ".\validation_extended_real\exhaustive_measurement_transcripts_tiny.json")
Invoke-PythonChecked -Arguments @(".\test_harness_mutation_sensitivity.py", "--out", ".\validation_extended_real\harness_mutation_sensitivity.json")
Invoke-PythonChecked -Arguments @(".\test_no_opaque_matrix.py", "--out", ".\validation_extended_real\no_opaque_matrix.json")

if ($Full) {
    Write-Host "=== Long-running optional validation ==="
    Invoke-PythonChecked -Arguments @(".\test_no_opaque_matrix.py", "--full", "--out", ".\validation_extended_real\no_opaque_matrix_full.json")
    Invoke-PythonChecked -Arguments @(".\test_secp256k1_schedule_witness.py", "--require-real-qiskit", "--out", ".\validation_extended_real\secp256k1_1536_witness.json")
    Invoke-PythonChecked -Arguments @(".\test_coherent_eea_pair_matrix.py", "--pairs", "4", "--shots", "1", "--require-real-qiskit", "--out", ".\validation_extended_real\coherent_pair_matrix.json")
    Invoke-PythonChecked -Arguments @(".\hash_derived_large_prime_sweep.py", "18446744073709551557", "--count", "16", "--require-real-qiskit", "--out", ".\validation_extended_real\hash_p64.json")
}

Write-Host "[PASS] Extended validation suite completed."
