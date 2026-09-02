param(
    [switch]$Approved
)

$ErrorActionPreference = 'Stop'
if (-not $Approved) {
    throw 'V3 sequence formal training is gated. Re-run with -Approved only after user authorization.'
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.envs\microwakeword\Scripts\python.exe'
$runner = Join-Path $projectRoot 'phase2\scripts\run_microwakeword_v3_sequence_training.py'
$config = Join-Path $projectRoot 'configs\models\microwakeword_tiny_v3_sequence.yaml'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $projectRoot "runs\qingxiaojia\microwakeword_tiny_v3_sequence\formal\$stamp"
New-Item -ItemType Directory -Path $runDir | Out-Null

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path $projectRoot 'src')
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$arguments = @(
    '-u',
    $runner,
    '--config', $config,
    '--run-dir', $runDir,
    '--allow-formal-training'
)
$stdout = Join-Path $runDir 'launcher.stdout.log'
$stderr = Join-Path $runDir 'launcher.stderr.log'
$resume = "`"$python`" -u `"$runner`" --config `"$config`" --run-dir `"$runDir`" --resume --allow-formal-training"
Set-Content -LiteralPath (Join-Path $runDir 'RESUME_COMMAND.txt') -Value $resume -Encoding UTF8

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

$statusPath = Join-Path $runDir 'TRAINING_STATUS.json'
$deadline = [DateTime]::UtcNow.AddSeconds(45)
while (-not (Test-Path -LiteralPath $statusPath)) {
    if ($process.HasExited) {
        $errorText = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Raw } else { '' }
        throw "V3 formal trainer exited before status initialization. ExitCode=$($process.ExitCode) stderr=$errorText"
    }
    if ([DateTime]::UtcNow -ge $deadline) {
        throw "Timed out waiting for TRAINING_STATUS.json; PID=$($process.Id)"
    }
    Start-Sleep -Milliseconds 500
    $process.Refresh()
}

$status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
[pscustomobject]@{
    Status = $status.status
    PID = $process.Id
    CurrentStep = $status.current_step
    BestValidationSequenceF1 = $status.best_validation_sequence_f1
    RunDirectory = $runDir
    EstimatedCompletionTimeUtc = $status.estimated_completion_time_utc
    StatusFile = $statusPath
    TrainingLog = Join-Path $runDir 'training.log'
    ResumeCommand = Join-Path $runDir 'RESUME_COMMAND.txt'
}
