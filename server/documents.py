"""Documents: the shared transcript (text + speakers, optimistic-locked) plus
each user's own coding layer. GET reassembles the exact v2 doc the offline tool
already understands (shared doc + project codebook + the caller's layer)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from server import db
from server.auth import get_db, require_user
from server.projects import member_or_403
from server.transcripts import parse_transcript

router = APIRouter()


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
        "codeApplications": layer.code_applications if layer else [],
        "highlights": layer.highlights if layer else [],
        "comments": layer.comments if layer else [],
    })
    return shared


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
    layer.code_applications = body.get("codeApplications") or []
    layer.highlights = body.get("highlights") or []
    layer.comments = body.get("comments") or []
    s.commit()
    return {"ok": True}


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
