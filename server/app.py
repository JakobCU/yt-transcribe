"""FastAPI backend: serves the Transkript-Checker frontend and runs the
Whisper + pyannote pipeline as background jobs.

Run (from the repo root, in the yt-transcribe conda env):
    uvicorn server.app:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/ — upload an audio file, watch progress, and the
transcript opens straight in the tool. Set TRANSCRIBE_FAKE=1 to test the flow
without loading any models.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from server import auth, coding, db, documents, jobs, llm, pipeline, projects

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "tool" / "src"
MEDIA_DIR = REPO_ROOT / "server" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD = 5 * 1024 ** 3   # 5 GiB cap so a client can't fill the disk
SSE_MAX_SECONDS = 6 * 3600   # stop streaming a single job after 6h

load_dotenv(REPO_ROOT / ".env")
db.init_db()

app = FastAPI(title="Transkript-Checker")
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(documents.router)


def _transcribe_runner(payload: dict, progress) -> dict:
    """Run the pipeline, then persist the result as a project document (if a
    project was given) so it survives the client closing."""
    res = pipeline.run(payload, progress)
    pid = payload.get("project_id")
    if pid:
        with db.SessionLocal() as s:
            d = documents.create_document_from_text(
                s, pid, payload.get("name"), res["text"], created_by=payload.get("user_id", ""))
            res["document_id"] = d.id
    return res


jobs.register_runner("transcribe", _transcribe_runner)
jobs.register_runner("code", coding.run)


def _has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


def _safe_name(filename: str) -> str:
    stem = Path(filename or "audio").name
    return re.sub(r"[^\w.\- ]+", "_", stem) or "audio"


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "diarizationAvailable": _has_hf_token(),
        "fake": os.environ.get("TRANSCRIBE_FAKE") == "1",
        "llm": llm.available_providers(),
        "auth": {"allowedDomains": auth.allowed_domains()},
    }


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form("large-v3"),
    language: str = Form(""),
    diarize: bool = Form(True),
    device: str = Form(""),
    project_id: str = Form(None),
    user: db.User = Depends(auth.require_user),
    s=Depends(auth.get_db),
):
    if project_id:
        projects.member_or_403(s, user, project_id)
    safe = _safe_name(audio.filename)
    dest = MEDIA_DIR / f"{uuid.uuid4().hex}_{safe}"
    total = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await audio.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, "Datei zu groß")
                await asyncio.to_thread(f.write, chunk)  # don't block the event loop
    except BaseException:
        dest.unlink(missing_ok=True)  # clean up the partial/aborted upload
        raise

    job_id = jobs.create_job("transcribe", {
        "path": str(dest),
        "name": Path(safe).stem,
        "model": model,
        "language": language.strip() or None,
        "diarize": diarize,
        "device": device.strip() or None,
        "project_id": project_id or None,
        "user_id": user.id,
    })
    return {"job_id": job_id}


@app.post("/api/code")
def code(body: dict, user: db.User = Depends(auth.require_user)):
    """Start an LLM coding job. body: {codes[], segments[], provider, model, context, name}."""
    if not body.get("codes"):
        raise HTTPException(400, "codes (codebook) required")
    if not body.get("segments"):
        raise HTTPException(400, "segments required")
    provider = body.get("provider") or "fake"
    avail = llm.available_providers()
    if provider not in avail or not avail[provider]["available"]:
        raise HTTPException(400, f"provider '{provider}' not available")
    job_id = jobs.create_job("code", {
        "codes": body["codes"],
        "segments": body["segments"],
        "provider": provider,
        "model": body.get("model") or avail[provider]["default_model"],
        "context": int(body.get("context", 1)),
        "name": body.get("name") or "coding",
    })
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, f"job not done (status={job['status']})")
    return jobs.get_result(job_id)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    async def gen():
        last = None
        waited = 0.0
        while True:
            if await request.is_disconnected():
                return  # client closed the stream — stop polling the registry
            job = jobs.get_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'error': 'unknown job'})}\n\n"
                return
            snapshot = (job["status"], job["stage"], job["progress"])
            if snapshot != last:
                last = snapshot
                yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.4)
            waited += 0.4
            if waited > SSE_MAX_SECONDS:
                return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Frontend last, so /api/* routes take precedence over this catch-all.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
