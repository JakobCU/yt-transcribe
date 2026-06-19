# Backend (FastAPI)

Serves the Transkript-Checker tool **and** runs the Whisper + pyannote pipeline
as background jobs — so the flow is: upload audio → transcribe/diarize → the
finished transcript opens straight in the tool, as one pipeline.

This is the single-process design for one machine / a small team: one uvicorn
process + one worker thread (GPU jobs run strictly one at a time) + an in-memory
job registry. Models are loaded once and kept resident across jobs. The team
hardening (separate worker process, database, auth, multi-user coding layers)
is the next phase — see `tool/docs/ENGINEERING_BRIEF.md` §5.

## Install

```bash
# in the yt-transcribe conda env (has torch/whisper/pyannote already)
pip install -e ".[server]"
```

## Run

```bash
# from the repo root
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> → click **🎙 Transkribieren** → pick an audio/video
file → watch progress → the transcript loads in the tool.

- **Diarization** needs a Hugging Face token in `.env` (`HF_TOKEN=...`), same as
  the CLI. Without it, the server transcribes only (the checkbox auto-disables).
- **`TRANSCRIBE_FAKE=1`** returns a canned transcript without loading any model —
  for testing the upload/job/progress/load plumbing without a GPU.
- **`KEEP_UPLOADS=1`** keeps uploaded audio in `server/media/` after processing.
  By default uploads (and the derived WAV) are deleted once the job finishes —
  disk hygiene + data retention for sensitive interview material.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | `{ok, diarizationAvailable, fake}` — frontend probes this to show the button |
| POST | `/api/transcribe` | multipart: `audio` file + `model`, `language`, `diarize` → `{job_id}` |
| GET  | `/api/jobs/{id}` | job status `{status, stage, progress, error, ...}` |
| GET  | `/api/jobs/{id}/result` | `{text, language, name, diarized, device}` once done |
| GET  | `/api/jobs/{id}/events` | Server-Sent Events stream of progress |
| POST | `/api/code` | LLM coding job: `{codes[], segments[], provider, model, context}` → `{job_id}` (status/result/events via the same `/api/jobs/...` routes) |

## LLM-assisted coding

The codes panel's **🤖 KI-Kodieren** button sends the current codebook + transcript
to `/api/code`. The model applies codes **deductively** (codebook only), grounds
every code in a verbatim quote (server rejects any quote that isn't a substring —
anti-hallucination), and returns suggestions with a rationale + confidence. They
land as `status:'suggested'` for the human to review (accept / reject / jump);
nothing is auto-applied. Pluggable per request:

- **claude** — needs `ANTHROPIC_API_KEY` in `.env`. Best quality; segments leave the machine.
- **ollama** — local model at `OLLAMA_HOST` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL` (default `llama3.1:8b`). Data stays local — the privacy-friendly default.
- **fake** — `LLM_FAKE=1`, canned suggestions for testing the loop without a key/GPU.

`/api/health` reports which providers are available so the UI only offers the live ones.

The result `text` is the `[HH:MM:SS] SPEAKER: text` format the tool parses (TXT
only; no server-side SRT). The same `src/` frontend is served here and also
builds into the offline single-file (`tool/build.py`) — the offline mode simply
has no backend and hides the transcribe button.

## Team mode (auth + projects + DB)

Persistence lives in SQLite (`server/data/app.db`, `DATABASE_URL` to switch to
Postgres). Registration is open but restricted to `ALLOWED_EMAIL_DOMAINS`
(default `ait.ac.at`); the first user becomes admin. Login is a server-side
session cookie (argon2 passwords).

Sharing model: a **project** groups transcripts + a shared **codebook** +
members (admin / coder). A **document** is the shared, optimistic-locked
transcript text; **codes/highlights/comments are per user** (`UserLayer`), so
coders annotate the same material independently (intercoder / LLM-vs-human).

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/auth/register` · `/login` · `/logout` | accounts; GET `/api/auth/me` |
| GET/POST | `/api/projects` | list / create projects |
| GET | `/api/projects/{id}` | members + role + codebook |
| POST | `/api/projects/{id}/members` | add member (admin) |
| GET/PUT | `/api/projects/{id}/codebook` | shared codebook |
| GET/POST | `/api/projects/{id}/documents` | list / create (from text) |
| GET | `/api/documents/{id}` | merged v2 doc (shared + codebook + caller's layer) |
| PUT | `/api/documents/{id}/text` | save shared text (optimistic `rev` → 409) |
| PUT | `/api/documents/{id}/layer` | save the caller's codes/highlights/comments |

`/api/transcribe` (with a `project_id`) persists its result as a project
document; `/api/transcribe` and `/api/code` now require login.

## Production deploy

For a hardened, self-hosted setup — **PostgreSQL** (`pip install -e ".[server,postgres]"`,
set `DATABASE_URL=postgresql+psycopg://…`), **HTTPS** via Caddy + Let's Encrypt,
run as always-on services, with backups — see **[`deploy/README.md`](../deploy/README.md)**.
Native on the Windows GPU box now; a full Docker stack (GPU passthrough) for a
Linux GPU server later. Set `COOKIE_SECURE=1` behind TLS.

## Notes / limits (single-process design)

- Upload cap 5 GiB; the job registry keeps the last 200 jobs in memory.
- Auth is on (argon2 session cookies, domain-restricted registration). Still,
  bind the app to `127.0.0.1` and let Caddy (or your reverse proxy) face the network.
- **Exactly one uvicorn worker** — the in-process job queue and resident GPU
  models are per-process state; never run `--workers > 1`.
- On Windows, run natively (conda + CUDA). Don't containerize the GPU worker
  (GPU-in-Docker on Windows is a pain zone — see the brief). On Linux, GPU-in-Docker
  works (see `deploy/Dockerfile`).
