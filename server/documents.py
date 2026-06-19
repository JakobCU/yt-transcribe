"""Documents: the shared transcript (text + speakers, optimistic-locked) plus
each user's own coding layer. GET reassembles the exact v2 doc the offline tool
already understands (shared doc + project codebook + the caller's layer)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from server import db
from server.auth import get_db, require_user
from server.projects import member_or_403
from server.transcripts import parse_transcript

router = APIRouter()

AUDIO_DIR = Path(__file__).resolve().parent / "media" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def store_audio(src_path: str, doc_id: str) -> str:
    """Transcode the source to a compact mono Opus copy for storage/serving.
    Returns the stored filename, or '' on failure (audio is optional). The WAV
    Whisper needs is transient (created during transcription) and not kept here."""
    for ext, codec, br in ((".ogg", "libopus", "24k"), (".m4a", "aac", "48k")):
        out = AUDIO_DIR / f"{doc_id}{ext}"
        try:
            subprocess.run(
                ["ffmpeg", "-i", src_path, "-vn", "-ac", "1", "-c:a", codec, "-b:a", br, "-y", str(out)],
                capture_output=True, check=True,
            )
            return out.name
        except Exception:
            out.unlink(missing_ok=True)
    return ""


def create_document_from_text(s: DBSession, project_id: str, name: str, text: str,
                              created_by: str = "") -> db.Document:
    shared = parse_transcript(text, name or "Transkript")
    d = db.Document(project_id=project_id, name=name or "Transkript", doc=shared, rev=0,
                    created_by=created_by, updated_by=created_by)
    s.add(d)
    s.commit()
    return d


def _layer(s: DBSession, document_id: str, user_id: str) -> db.UserLayer:
    layer = s.scalar(select(db.UserLayer).where(
        db.UserLayer.document_id == document_id, db.UserLayer.user_id == user_id))
    if layer is None:
        layer = db.UserLayer(document_id=document_id, user_id=user_id,
                             code_applications=[], highlights=[], comments=[])
        s.add(layer)
    return layer


@router.post("/api/projects/{pid}/documents")
def create_document(pid: str, body: dict, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    member_or_403(s, user, pid)
    text = body.get("text") or ""
    if not text.strip():
        raise HTTPException(400, "Kein Transkript-Text")
    d = create_document_from_text(s, pid, body.get("name") or "Transkript", text, created_by=user.id)
    return {"id": d.id, "name": d.name}


@router.get("/api/documents/{did}")
def get_document(did: str, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    doc = s.get(db.Document, did)
    if doc is None:
        raise HTTPException(404, "Dokument nicht gefunden")
    proj, _ = member_or_403(s, user, doc.project_id)
    layer = s.scalar(select(db.UserLayer).where(
        db.UserLayer.document_id == did, db.UserLayer.user_id == user.id))
    cb = proj.codebook or {}
    shared = dict(doc.doc or {})
    shared.update({
        "schemaVersion": 2,
        "docId": doc.id, "rev": doc.rev, "projectId": doc.project_id, "name": doc.name,
        "coding": cb.get("coding") or {"mode": "inductive"},
        "codeSystem": cb.get("codeSystem") or [],
        "codebookRev": proj.codebook_rev or 0,
        "codeApplications": layer.code_applications if layer else [],
        "highlights": layer.highlights if layer else [],
        "comments": layer.comments if layer else [],
        "layerRev": (layer.layer_rev or 0) if layer else 0,
        "hasAudio": bool(doc.audio),
    })
    return shared


@router.get("/api/documents/{did}/audio")
def get_audio(did: str, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    doc = s.get(db.Document, did)
    if doc is None:
        raise HTTPException(404, "Dokument nicht gefunden")
    member_or_403(s, user, doc.project_id)
    if not doc.audio:
        raise HTTPException(404, "Kein Audio gespeichert")
    path = AUDIO_DIR / doc.audio  # doc.audio is server-generated ("{id}.ogg"); no user input
    if not path.is_file():
        raise HTTPException(404, "Audiodatei fehlt")
    from fastapi.responses import FileResponse
    mime = "audio/ogg" if path.suffix == ".ogg" else "audio/mp4"
    return FileResponse(str(path), media_type=mime)


@router.put("/api/documents/{did}/text")
def save_text(did: str, body: dict, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    doc = s.get(db.Document, did)
    if doc is None:
        raise HTTPException(404, "Dokument nicht gefunden")
    member_or_403(s, user, doc.project_id)
    if "rev" in body and body["rev"] != doc.rev:
        raise HTTPException(409, {"message": "Das Transkript wurde zwischenzeitlich geändert.", "rev": doc.rev})
    doc.doc = {"schemaVersion": 2, "name": doc.name, "header": body.get("header", ""),
               "speakers": body.get("speakers") or [], "segments": body.get("segments") or []}
    doc.rev += 1
    doc.updated_by = user.id
    s.commit()
    return {"rev": doc.rev}


@router.put("/api/documents/{did}/layer")
def save_layer(did: str, body: dict, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    doc = s.get(db.Document, did)
    if doc is None:
        raise HTTPException(404, "Dokument nicht gefunden")
    member_or_403(s, user, doc.project_id)
    layer = _layer(s, did, user.id)
    cur = layer.layer_rev or 0
    if "rev" in body and body["rev"] != cur:
        # same user's layer changed elsewhere (another tab/device) — don't clobber it
        raise HTTPException(409, {"message": "Deine Kodier-Ebene wurde anderswo geändert.", "rev": cur})
    layer.code_applications = body.get("codeApplications") or []
    layer.highlights = body.get("highlights") or []
    layer.comments = body.get("comments") or []
    layer.layer_rev = cur + 1
    s.commit()
    return {"rev": layer.layer_rev}


@router.delete("/api/documents/{did}")
def delete_document(did: str, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    doc = s.get(db.Document, did)
    if doc is None:
        raise HTTPException(404, "Dokument nicht gefunden")
    _, role = member_or_403(s, user, doc.project_id)
    if role != "admin":
        raise HTTPException(403, "Nur Projekt-Admins können Dokumente löschen")
    s.query(db.UserLayer).filter(db.UserLayer.document_id == did).delete()
    s.delete(doc)
    s.commit()
    return {"ok": True}
