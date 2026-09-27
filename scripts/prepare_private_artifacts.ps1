param(
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,

    [Parameter(Mandatory=$true)]
    [string]$Tokenizer,

    [string]$Destination = ".\\private"
)

$ErrorActionPreference = "Stop"

$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$tokenizerPath = (Resolve-Path -LiteralPath $Tokenizer).Path

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path

$checkpointOut = Join-Path $destinationPath "model.pt"
$tokenizerOut = Join-Path $destinationPath "tokenizer.json"

Copy-Item -LiteralPath $checkpointPath -Destination $checkpointOut -Force
Copy-Item -LiteralPath $tokenizerPath -Destination $tokenizerOut -Force

$checkpointSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpointOut).Hash.ToLower()
$tokenizerSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $tokenizerOut).Hash.ToLower()

$envText = @"
TALGIN_CHECKPOINT_PATH=/model/model.pt
TALGIN_TOKENIZER_PATH=/model/tokenizer.json
TALGIN_CHECKPOINT_SHA256=$checkpointSha
TALGIN_TOKENIZER_SHA256=$tokenizerSha
"@

$envFile = Join-Path $destinationPath "model.env"
Set-Content -LiteralPath $envFile -Value $envText -Encoding utf8

Write-Host ""
Write-Host "Private artifacts prepared:" -ForegroundColor Green
Write-Host "  $checkpointOut"
Write-Host "  $tokenizerOut"
Write-Host "  $envFile"
Write-Host ""
Write-Host "Checkpoint SHA256: $checkpointSha"
Write-Host "Tokenizer  SHA256: $tokenizerSha"
Write-Host ""
Write-Host "This folder is gitignored. Do NOT commit it." -ForegroundColor Yellow
