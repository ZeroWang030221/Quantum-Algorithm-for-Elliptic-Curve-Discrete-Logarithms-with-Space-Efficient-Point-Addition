$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force validation_real | Out-Null

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$LogPath = ""
    )

    # Windows PowerShell 5.1 converts native stderr piped through `2>&1` into
    # NativeCommandError records.  With `$ErrorActionPreference = "Stop"`, the
    # first line of a Python traceback can terminate the pipeline and hide the
    # actual exception.  Temporarily keep native stderr non-terminating, then
    # fail explicitly from Python's real process exit code.
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 0
    try {
        if ($LogPath) {
            & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        }
        else {
            & python @Arguments
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }

    if ($exitCode -ne 0) {
        $commandText = "python " + ($Arguments -join " ")
        if ($LogPath) {
            throw "$commandText failed with exit code $exitCode. Full output: $LogPath"
        }
        throw "$commandText failed with exit code $exitCode."
    }
}

Invoke-PythonChecked -Arguments @("-c", "import qiskit; print('Qiskit:', qiskit.__version__)")

Write-Host "=== Paper/code structural alignment ==="
Invoke-PythonChecked -Arguments @(".\test_paper_alignment_gate_structure.py") -LogPath ".\validation_real\paper_alignment_gate_structure.log"
Invoke-PythonChecked -Arguments @(".\test_fastdual_interval_selector.py") -LogPath ".\validation_real\fastdual_interval_selector.log"
Invoke-PythonChecked -Arguments @(".\test_explicit_measurement_inverse_blocks.py") -LogPath ".\validation_real\explicit_measurement_inverse_blocks.log"
Invoke-PythonChecked -Arguments @(".\test_reachable_step_forward_inverse.py", "37") -LogPath ".\validation_real\reachable_step_forward_inverse_p37.log"

Write-Host "=== EEA semantics and coherence ==="
Invoke-PythonChecked -Arguments @(".\extended_semantic_sweep.py", "419", "--all", "--require-real-qiskit", "--out", ".\validation_real\p419_all.json")
Invoke-PythonChecked -Arguments @(".\test_p419_per_step_oracle.py", "--require-real-qiskit", "--out", ".\validation_real\p419_per_step.json")
Invoke-PythonChecked -Arguments @(".\test_p419_x153_quantum_semantics.py", "--shots", "8", "--require-real-qiskit", "--verbose") -LogPath ".\validation_real\p419_coherent.log"

Write-Host "=== Terminal padding and complete small-field point addition ==="
Invoke-PythonChecked -Arguments @(".\test_terminal_shift_epoch_835.py") -LogPath ".\validation_real\terminal_padding.log"
Invoke-PythonChecked -Arguments @(".\test_small_field_point_addition_semantics.py") -LogPath ".\validation_real\small_pa_semantics.log"
Invoke-PythonChecked -Arguments @(".\test_small_field_point_addition_coherence.py", "--shots", "8", "--out", ".\validation_real\small_pa_coherence.json") -LogPath ".\validation_real\small_pa_coherence.log"

Write-Host "=== Large-prime basis tests and fail-closed lowering ==="
Invoke-PythonChecked -Arguments @(".\hash_derived_large_prime_sweep.py", "65521", "--count", "32", "--require-real-qiskit", "--out", ".\validation_real\hash_p16.json")
Invoke-PythonChecked -Arguments @(".\hash_derived_large_prime_sweep.py", "4294967291", "--count", "12", "--require-real-qiskit", "--out", ".\validation_real\hash_p32.json")
Invoke-PythonChecked -Arguments @(".\test_no_opaque_counts.py") -LogPath ".\validation_real\no_opaque_counts.log"

Write-Host "=== Strict block/layout/count assembly tests ==="
Invoke-PythonChecked -Arguments @(".\test_eea_strict_main.py") -LogPath ".\validation_real\strict_eea.log"
Invoke-PythonChecked -Arguments @(".\test_point_addition_strict_main.py", "--skip-large-primes") -LogPath ".\validation_real\strict_point_addition.log"

Write-Host "[PASS] Base real-Qiskit validation completed."
