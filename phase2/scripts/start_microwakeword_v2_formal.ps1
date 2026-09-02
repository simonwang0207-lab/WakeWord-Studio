param(
    [switch]$Approved
)

$ErrorActionPreference = 'Stop'
if (-not $Approved) {
    throw 'Formal v2 training is gated. Re-run with -Approved only after START V2 FORMAL TRAINING.'
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.envs\microwakeword\Scripts\python.exe'
$runner = Join-Path $projectRoot 'phase2\scripts\run_microwakeword_training.py'
$config = Join-Path $projectRoot 'configs\models\microwakeword_tiny_v2.yaml'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $projectRoot "runs\qingxiaojia\microwakeword_tiny_v2\formal\$stamp"
New-Item -ItemType Directory -Path $runDir | Out-Null

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path $projectRoot 'src')
$arguments = @(
    '-u',
    $runner,
    '--config', $config,
    '--mode', 'formal',
    '--run-dir', $runDir,
    '--allow-formal-training'
)
$stdout = Join-Path $runDir 'launcher.stdout.log'
$stderr = Join-Path $runDir 'launcher.stderr.log'
$resume = "`"$python`" -u `"$runner`" --config `"$config`" --mode formal --run-dir `"$runDir`" --resume --allow-formal-training"
Set-Content -LiteralPath (Join-Path $runDir 'RESUME_COMMAND.txt') -Value $resume -Encoding UTF8

$startParameters = @{
    FilePath = $python
    ArgumentList = $arguments
    WorkingDirectory = $projectRoot
    WindowStyle = 'Hidden'
    RedirectStandardOutput = $stdout
    RedirectStandardError = $stderr
    PassThru = $true
}
$process = Start-Process @startParameters

[pscustomobject]@{
    PID = $process.Id
    RunDirectory = $runDir
    StatusFile = Join-Path $runDir 'TRAINING_STATUS.json'
    TrainingLog = Join-Path $runDir 'training.log'
    ResumeCommand = Join-Path $runDir 'RESUME_COMMAND.txt'
}
