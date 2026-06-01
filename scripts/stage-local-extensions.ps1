param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",
    [string]$DestinationRoot = (Join-Path $env:LOCALAPPDATA "cove\extensions")
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

$extensions = @(
    "AI.Core",
    "AI.Tagging",
    "AI.Faces",
    "AI.Visual",
    "AI.Audio",
    "AI.Full"
)

Write-Host "Building extension UI bundles..."
Push-Location $repoRoot
try {
    npm run build:ui
    if ($LASTEXITCODE -ne 0) {
        throw "npm run build:ui failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Building AI.Extensions ($Configuration)..."
dotnet build (Join-Path $repoRoot "AI.Extensions.slnx") -c $Configuration --nologo
if ($LASTEXITCODE -ne 0) {
    throw "dotnet build failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null

foreach ($extension in $extensions) {
    $outputDir = Join-Path $repoRoot "extensions\$extension\bin\$Configuration\net10.0"
    $manifestPath = Join-Path $outputDir "extension.json"

    if (-not (Test-Path $manifestPath)) {
        throw "Missing manifest at $manifestPath"
    }

    $manifest = Get-Content -Raw -Path $manifestPath | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($manifest.id)) {
        throw "Manifest at $manifestPath does not define an id"
    }

    $targetDir = Join-Path $DestinationRoot $manifest.id
    if (Test-Path $targetDir) {
        Remove-Item -Path $targetDir -Recurse -Force
    }

    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    Copy-Item -Path (Join-Path $outputDir "*") -Destination $targetDir -Recurse -Force

    Write-Host ("Staged {0} -> {1}" -f $manifest.id, $targetDir)
}

Write-Host "Local AI extensions are ready. Restart Cove to reload them if it is already running."