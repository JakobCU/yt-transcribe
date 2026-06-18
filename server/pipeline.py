"""Bridges the job queue to the yt_transcribe pipeline.

Set TRANSCRIBE_FAKE=1 to return a canned transcript without loading any ML
models — used to exercise the upload/job/progress/load plumbing end-to-end
without a GPU.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

_FAKE_TRANSCRIPT = """[00:00:00] SPEAKER_00: Dies ist ein Test im Fake-Modus.
[00:00:03] SPEAKER_01: Die Pipeline-Anbindung funktioniert, ohne ein Modell zu laden.
[00:00:07] SPEAKER_00: Schalte TRANSCRIBE_FAKE aus, um echt zu transkribieren."""


def _resolve_hf_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def run(payload: dict, progress: Callable[[str, float], None]) -> dict:
    """Run one transcription job. payload: {path, name, model, language, diarize}."""
    name = payload.get("name") or "Aufnahme"
    audio_path = payload.get("path")

    if os.environ.get("TRANSCRIBE_FAKE") == "1":
        for stage in ("convert", "transcribe", "diarize", "merge"):
            progress(stage, 0.5)
            time.sleep(0.3)
        progress("done", 1.0)
        return {"text": _FAKE_TRANSCRIPT, "language": "de", "name": name,
                "diarized": payload.get("diarize", True), "device": "fake"}

    if not audio_path or not Path(audio_path).is_file():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    # Imported lazily so the server starts (and serves the tool) even before any
    # heavy ML import, and so a missing GPU dep only breaks transcription jobs.
    from yt_transcribe.transcribe import transcribe_segments

    diarize = bool(payload.get("diarize", True))
    hf_token = _resolve_hf_token(payload.get("hf_token"))
    if diarize and not hf_token:
        diarize = False  # no token -> transcription only, rather than failing

    try:
        result = transcribe_segments(
            audio_path=audio_path,
            model=payload.get("model") or "large-v3",
            language=payload.get("language") or None,
            no_diarize=not diarize,
            hf_token=hf_token,
            device=payload.get("device") or None,
            progress=progress,
        )
    finally:
        # Ad-hoc (no project): delete the upload + derived WAV now. For project jobs
        # the runner keeps the source so it can transcode a compact copy to store,
        # then cleans up itself (see _transcribe_runner). KEEP_UPLOADS=1 keeps all.
        if not payload.get("project_id") and os.environ.get("KEEP_UPLOADS") != "1":
            _cleanup(audio_path)

    return {
        "text": result["text"],
        "language": result.get("language"),
        "name": name,
        "diarized": result.get("diarized", False),
        "device": result.get("device"),
    }


def _cleanup(audio_path: str) -> None:
    src = Path(audio_path)
    for p in {src, src.with_suffix(".wav")}:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
