"""In-process job queue, generic over job KIND.

Each kind ('transcribe', 'code', ...) gets its own worker thread and queue, so
GPU transcription and LLM coding don't block each other, while status/result/SSE
endpoints stay shared (keyed by job id). Transcription deliberately runs one at a
time (single worker per kind) so GPU jobs don't contend for VRAM. This is the
single-process design for one machine / a small team; the full team server later
swaps this for a worker process + DB-backed queue, same API.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from typing import Callable, Optional

log = logging.getLogger("transcribe.jobs")

MAX_JOBS = 300  # cap the in-memory registry; oldest finished jobs are evicted
_PRIVATE = ("_input", "result")  # never exposed to API clients

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queues: dict[str, "queue.Queue[str]"] = {}
_workers: dict[str, threading.Thread] = {}
_runners: dict[str, Callable[[dict, Callable[[str, float], None]], dict]] = {}


def register_runner(kind: str, fn: Callable[[dict, Callable[[str, float], None]], dict]) -> None:
    """Register the worker function for a job kind: runner(input, progress) -> result."""
    _runners[kind] = fn
    _queues.setdefault(kind, queue.Queue())
    _ensure_worker(kind)


def create_job(kind: str, payload: dict) -> str:
    if kind not in _runners:
        raise ValueError(f"no runner registered for kind {kind!r}")
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",   # queued | running | done | error
            "stage": "queued",
            "progress": 0.0,
            "error": None,
            "result": None,
            "name": payload.get("name"),
            "created": time.time(),
            "_input": dict(payload),
        }
        _evict_locked()
    _queues[kind].put(job_id)
    _ensure_worker(kind)
    return job_id


def get_job(job_id: str, include_result: bool = False) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        view = {k: v for k, v in job.items() if k not in _PRIVATE}
        if include_result:
            view["result"] = job.get("result")
        return view


def get_result(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return job.get("result") if job else None


def _evict_locked() -> None:
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        (j for j in _jobs.values() if j["status"] in ("done", "error")),
        key=lambda j: j["created"],
    )
    for j in finished:
        if len(_jobs) <= MAX_JOBS:
            break
        _jobs.pop(j["id"], None)


def _set(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _ensure_worker(kind: str) -> None:
    w = _workers.get(kind)
    if w is None or not w.is_alive():
        t = threading.Thread(target=_loop, args=(kind,), name=f"{kind}-worker", daemon=True)
        _workers[kind] = t
        t.start()


def _loop(kind: str) -> None:
    q = _queues[kind]
    while True:
        job_id = q.get()
        try:
            with _lock:
                job = _jobs.get(job_id)
                payload = dict(job["_input"]) if job else None
            if payload is None:
                continue
            _set(job_id, status="running", stage="starting")

            def progress(stage: str, frac: float, _id=job_id) -> None:
                _set(_id, stage=stage, progress=round(float(frac), 3))

            try:
                runner = _runners.get(kind)
                if runner is None:
                    raise RuntimeError(f"no runner for kind {kind!r}")
                result = runner(payload, progress)
                _set(job_id, status="done", stage="done", progress=1.0, result=result)
            except Exception as exc:  # noqa: BLE001 - surface failures to the client
                log.exception("%s job %s failed", kind, job_id)
                _set(job_id, status="error", error=str(exc))
        finally:
            q.task_done()
