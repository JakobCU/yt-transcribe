<#
.SYNOPSIS
  Back up the transkript::studio Postgres database + stored audio (server\media).

.DESCRIPTION
  Dumps Postgres via the Docker container (no local pg_dump needed) and zips the
  media folder, with timestamped filenames and retention. Schedule it daily with
  Task Scheduler. Restore a dump with:
    Get-Content dump.sql | docker compose -f deploy\docker-compose.yml exec -T db psql -U tcuser -d tcdb

.EXAMPLE
  .\backup.ps1 -KeepDays 14
#>
[CmdletBinding()]
param(
  [string]$OutDir = (Join-Path $PSScriptRoot "backups"),
  [int]$KeepDays = 14,
  [string]$DbUser = "tcuser",
  [string]$DbName = "tcdb"
)
$ErrorActionPreference = "Stop"
$compose = Join-Path $PSScriptRoot "docker-compose.yml"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$media = Join-Path $repo "server\media"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

# --- database -> plain SQL dump ---------------------------------------------
$dump = Join-Path $OutDir "db-$stamp.sql"
Write-Host "Dumping Postgres -> $dump"
docker compose -f "$compose" exec -T db pg_dump -U $DbUser -d $DbName | Out-File -Encoding utf8 $dump
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $dump) -or (Get-Item $dump).Length -eq 0) {
  throw "pg_dump failed (is the db container up? docker compose -f deploy\docker-compose.yml ps)"
}

# --- stored audio -----------------------------------------------------------
if (Test-Path $media) {
  $zip = Join-Path $OutDir "media-$stamp.zip"
  Write-Host "Archiving media -> $zip"
  Compress-Archive -Path (Join-Path $media "*") -DestinationPath $zip -Force
} else {
  Write-Host "No media folder yet — skipping audio archive."
}

# --- retention --------------------------------------------------------------
$cutoff = (Get-Date).AddDays(-$KeepDays)
Get-ChildItem $OutDir -File | Where-Object { $_.LastWriteTime -lt $cutoff } | ForEach-Object {
  Write-Host "Pruning old backup: $($_.Name)"; Remove-Item $_.FullName -Force
}
Write-Host "Backup complete ($stamp)."
