<#
.SYNOPSIS
  Stop and remove the transkript::studio Windows services installed by
  install-services.ps1. Does NOT touch your data (Postgres volume, .env, media).
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "SilentlyContinue"

foreach ($svc in "TranskriptStudio", "TranskriptStudioCaddy") {
  if (Get-Service $svc -ErrorAction SilentlyContinue) {
    Write-Host "Removing $svc ..."
    nssm stop $svc
    nssm remove $svc confirm
  } else {
    Write-Host "$svc not installed — skipping."
  }
}
Write-Host "Done. (Postgres container, .env, and server\media were left untouched.)"
