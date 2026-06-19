# Deploying transkript::studio (self-hosted, hardened)

This folder turns the dev setup into a hardened deployment: **PostgreSQL** for
the team database, **Caddy** for automatic HTTPS (Let's Encrypt) on a public
domain, run as **always-on services**, with **backups**.

There are two target shapes; the app code and config are identical, only the
wrapping differs:

| | **A — now: this Windows GPU box** | **B — soon: a Linux GPU server** |
|---|---|---|
| App (Whisper+pyannote on CUDA) | runs **natively** (conda) as an NSSM service | runs in a **Docker** container (GPU via nvidia-container-toolkit) |
| Postgres | Docker container on the same box | Docker container |
| Caddy (HTTPS) | NSSM service | Docker container |
| Start command | `windows\install-services.ps1` | `docker compose --profile full up -d --build` |

> **Why native on Windows?** The transcription pipeline runs in-process on the
> GPU, and GPU-in-Docker on Windows is unreliable. So on Windows the *app* runs
> natively; only Postgres is containerized. On Linux, GPU passthrough to Docker
> works, so the whole stack can be containers.

---

## Prerequisites (both)

- A **domain** (e.g. `studio.example.org`) with a DNS A/AAAA record pointing at
  the server, and inbound **TCP 80 + 443** open (Caddy needs them for Let's Encrypt).
- **Docker Desktop** (Windows) or **Docker Engine + compose** (Linux).
- A filled-in env file — copy and edit:
  ```
  cp deploy/.env.production.example deploy/.env
  ```
  Set a strong `POSTGRES_PASSWORD`, your `TC_DOMAIN` / `TC_TLS_EMAIL`,
  `ALLOWED_EMAIL_DOMAINS`, `HF_TOKEN`, and any LLM keys. Keep `COOKIE_SECURE=1`.
  `deploy/.env` is gitignored — never commit it.

---

## A — Windows GPU box (now)

1. **Start Postgres** (Docker, localhost-only):
   ```powershell
   docker compose -f deploy\docker-compose.yml up -d
   ```

2. **Point the app at Postgres.** The native app reads the **repo-root** `.env`
   (via python-dotenv). Put these there (alongside HF_TOKEN / API keys):
   ```
   DATABASE_URL=postgresql+psycopg://tcuser:<your-password>@127.0.0.1:5432/tcdb
   COOKIE_SECURE=1
   ALLOWED_EMAIL_DOMAINS=ait.ac.at
   ```
   Install the Postgres driver once:
   ```powershell
   pip install -e ".[server,postgres]"
   ```

3. **Verify the database** (creates the schema, checks locking) before going live:
   ```powershell
   $env:DATABASE_URL="postgresql+psycopg://tcuser:<pw>@127.0.0.1:5432/tcdb"
   python deploy\verify_db.py     # expect "ALL PASS"
   ```

4. **Install the services** (needs NSSM + Caddy):
   ```powershell
   winget install NSSM.NSSM
   winget install CaddyServer.Caddy
   .\deploy\windows\install-services.ps1 -Domain studio.example.org -TlsEmail admin@example.org
   ```
   This registers `TranskriptStudio` (uvicorn, 1 worker, bound to 127.0.0.1) and
   `TranskriptStudioCaddy` (HTTPS reverse proxy) as auto-start services. Logs land
   in `deploy\logs\`. First visit to `https://studio.example.org/` may pause a few
   seconds while Caddy fetches the certificate.

   Remove later with `deploy\windows\uninstall-services.ps1` (leaves data intact).

---

## B — Linux GPU server (soon)

1. Install the **nvidia-container-toolkit** so Docker can use the GPU.
2. Fill `deploy/.env` (as above; `DATABASE_URL` is set automatically to the `db`
   container, so you can leave it unset there).
3. Bring up the whole stack:
   ```bash
   docker compose -f deploy/docker-compose.yml --profile full up -d --build
   ```
   This builds the app image (CUDA + the project), starts Postgres, the app
   (GPU-enabled, 1 worker), and Caddy (HTTPS on 80/443). Verify:
   ```bash
   docker compose -f deploy/docker-compose.yml exec app python deploy/verify_db.py
   ```

---

## Operating notes

- **Exactly one app worker.** The in-process job queue and the resident GPU
  models are per-process state; never run uvicorn with `--workers > 1` (the
  service definitions already pin 1). Scale by queueing, not by forking workers.
- **HTTPS & cookies.** Caddy terminates TLS and forwards `X-Forwarded-Proto`;
  uvicorn runs with `--proxy-headers`, and `COOKIE_SECURE=1` makes the session
  cookie HTTPS-only. SSE progress streams are configured to flush immediately.
- **Backups.** `deploy/backup.ps1` (Windows) / `deploy/backup.sh` (Linux) dump
  Postgres + archive `server/media` with retention. Schedule daily (Task
  Scheduler / cron). Restore instructions are in the script headers.
- **Backups go to `deploy/backups/`** (gitignored). Copy them off-box regularly.
- **Schema upgrades.** New SQLite columns auto-migrate; on Postgres the current
  schema is created by `create_all` on first run. Future *destructive* schema
  changes will need a real migration tool (Alembic) — additive columns can be
  added by hand with `ALTER TABLE` in the meantime.
- **Privacy.** For sensitive interview material, prefer the local **Ollama** LLM
  provider (data stays on the box) over cloud providers, and keep `KEEP_UPLOADS`
  unset so raw uploads are deleted after transcription.
