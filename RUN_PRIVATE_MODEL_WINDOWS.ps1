$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\\private\\model.pt")) {
    throw "private\\model.pt not found. Run scripts\\prepare_private_artifacts.ps1 first."
}
if (-not (Test-Path ".\\private\\tokenizer.json")) {
    throw "private\\tokenizer.json not found. Run scripts\\prepare_private_artifacts.ps1 first."
}
if (-not (Test-Path ".\\private\\model.env")) {
    throw "private\\model.env not found. Run scripts\\prepare_private_artifacts.ps1 first."
}

docker build -t talginnosana .
docker run --rm --gpus all -p 8000:8000 --env-file ".\\private\\model.env" -v "$($PSScriptRoot)\\private:/model:ro" talginnosana
