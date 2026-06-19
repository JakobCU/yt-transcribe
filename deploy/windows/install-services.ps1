<#
.SYNOPSIS
  Install transkript::studio as Windows services (app + Caddy) via NSSM, so it
  starts on boot and restarts on crash. For the native Windows GPU host.

.DESCRIPTION
  Registers two auto-start services:
    • TranskriptStudio       -> uvicorn (the app; GPU pipeline + API + frontend)
    • TranskriptStudioCaddy  -> Caddy reverse proxy with automatic HTTPS

  The app reads its config (DATABASE_URL, COOKIE_SECURE, HF_TOKEN, API keys, ...)
  from the repo-root .env via python-dotenv, so you do NOT pass secrets here.
  Caddy needs the domain/email, which this script puts in the Caddy service env.

  Prereqs (install once):
    winget install NSSM.NSSM
    winget install CaddyServer.Caddy
  Plus: Postgres running (docker compose -f deploy\docker-compose.yml up -d),
  the repo-root .env filled in (see deploy\.env.production.example with
  COOKIE_SECURE=1 and the postgres DATABASE_URL), DNS for -Domain pointing here,
  and inbound TCP 80 + 443 open in the firewall.

.EXAMPLE
  .\install-services.ps1 -Domain studio.example.org -TlsEmail admin@example.org
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Domain,
  [Parameter(Mandatory = $true)][string]$TlsEmail,
  [int]$Port = 8000,
  [string]$Python = "C:\Users\TX.Lab\miniconda3\envs\yt-transcribe\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$caddyfile = Join-Path $repo "deploy\Caddyfile"
$logs = Join-Path $repo "deploy\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Need($cmd, $hint) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    throw "'$cmd' not found. Install it first: $hint"
  }
}
Need nssm  "winget install NSSM.NSSM"
Need caddy "winget install CaddyServer.Caddy"
if (-not (Test-Path $Python)) { throw "Python not found at $Python — pass -Python <path>." }
if (-not (Test-Path (Join-Path $repo ".env"))) {
  Write-Warning "No .env at $repo — the app needs it (DATABASE_URL, COOKIE_SECURE=1, HF_TOKEN, ...). See deploy\.env.production.example."
}

$caddyExe = (Get-Command caddy).Source

Write-Host "Repo:      $repo"
Write-Host "Python:    $Python"
Write-Host "Caddy:     $caddyExe"
Write-Host "Domain:    $Domain  (port $Port -> Caddy 443/80)"

# --- app service -------------------------------------------------------------
# EXACTLY one worker (in-process job queue + resident GPU models are per-process).
# --proxy-headers + forwarded-allow-ips 127.0.0.1 so Secure cookies work behind Caddy.
nssm install TranskriptStudio "$Python" `
  -m uvicorn server.app:app --host 127.0.0.1 --port $Port `
  --workers 1 --proxy-headers --forwarded-allow-ips 127.0.0.1
nssm set TranskriptStudio AppDirectory "$repo"
nssm set TranskriptStudio AppStdout "$logs\app.out.log"
nssm set TranskriptStudio AppStderr "$logs\app.err.log"
nssm set TranskriptStudio AppEnvironmentExtra "PYTHONUNBUFFERED=1"
nssm set TranskriptStudio Start SERVICE_AUTO_START
nssm set TranskriptStudio AppExit Default Restart

# --- caddy service -----------------------------------------------------------
nssm install TranskriptStudioCaddy "$caddyExe" run --config "$caddyfile"
nssm set TranskriptStudioCaddy AppDirectory "$repo\deploy"
nssm set TranskriptStudioCaddy AppStdout "$logs\caddy.out.log"
nssm set TranskriptStudioCaddy AppStderr "$logs\caddy.err.log"
nssm set TranskriptStudioCaddy AppEnvironmentExtra `
  "TC_DOMAIN=$Domain" "TC_TLS_EMAIL=$TlsEmail" "TC_UPSTREAM=127.0.0.1" "TC_APP_PORT=$Port"
nssm set TranskriptStudioCaddy Start SERVICE_AUTO_START
nssm set TranskriptStudioCaddy AppExit Default Restart

Start-Service TranskriptStudio
Start-Service TranskriptStudioCaddy

Write-Host ""
Write-Host "Installed + started. Check status:  Get-Service TranskriptStudio*, "
Write-Host "logs in deploy\logs\. Visit https://$Domain/ (first request may pause"
Write-Host "while Caddy obtains the certificate)."
