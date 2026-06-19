"""Projects: group transcripts + a shared codebook + members (admin / coder)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DBSession

from server import db
from server.auth import get_db, require_user

router = APIRouter(prefix="/api/projects")


def member_or_403(s: DBSession, user: db.User, project_id: str):
    """Return (project, role) or raise. Used across project/document routes."""
    proj = s.get(db.Project, project_id)
    if proj is None:
        raise HTTPException(404, "Projekt nicht gefunden")
    pm = s.scalar(select(db.ProjectMember).where(
        db.ProjectMember.project_id == project_id, db.ProjectMember.user_id == user.id))
    if pm is None:
        raise HTTPException(403, "Kein Zugriff auf dieses Projekt")
    return proj, pm.role


def _count(s, model, **where):
    q = select(func.count()).select_from(model)
    for k, v in where.items():
        q = q.where(getattr(model, k) == v)
    return s.scalar(q) or 0


@router.post("")
def create_project(body: dict, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Projektname fehlt")
    proj = db.Project(name=name, owner_id=user.id, codebook={"codeSystem": [], "coding": {"mode": "inductive"}})
    s.add(proj)
    s.flush()
    s.add(db.ProjectMember(project_id=proj.id, user_id=user.id, role="admin"))
    s.commit()
    return {"id": proj.id, "name": proj.name, "role": "admin"}


@router.get("")
def list_projects(s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    rows = s.execute(
        select(db.Project, db.ProjectMember.role)
        .join(db.ProjectMember, db.ProjectMember.project_id == db.Project.id)
        .where(db.ProjectMember.user_id == user.id)
        .order_by(db.Project.created_at)
    ).all()
    return [{"id": p.id, "name": p.name, "role": role,
             "documents": _count(s, db.Document, project_id=p.id),
             "members": _count(s, db.ProjectMember, project_id=p.id)}
            for p, role in rows]


@router.get("/{pid}")
def get_project(pid: str, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    proj, role = member_or_403(s, user, pid)
    members = s.execute(
        select(db.User.email, db.User.name, db.ProjectMember.role)
        .join(db.ProjectMember, db.ProjectMember.user_id == db.User.id)
        .where(db.ProjectMember.project_id == pid)
    ).all()
    return {"id": proj.id, "name": proj.name, "role": role, "codebook": proj.codebook or {},
            "members": [{"email": e, "name": n, "role": r} for e, n, r in members]}


@router.post("/{pid}/members")
def add_member(pid: str, body: dict, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    proj, role = member_or_403(s, user, pid)
    if role != "admin":
        raise HTTPException(403, "Nur Projekt-Admins können Mitglieder hinzufügen")
    email = (body.get("email") or "").strip().lower()
    target = s.scalar(select(db.User).where(db.User.email == email))
    if target is None:
        raise HTTPException(404, "Kein:e Nutzer:in mit dieser E-Mail (muss sich erst registrieren)")
    existing = s.scalar(select(db.ProjectMember).where(
        db.ProjectMember.project_id == pid, db.ProjectMember.user_id == target.id))
    if existing:
        existing.role = body.get("role") or existing.role
    else:
        s.add(db.ProjectMember(project_id=pid, user_id=target.id, role=body.get("role") or "coder"))
    s.commit()
    return {"ok": True}


@router.get("/{pid}/codebook")
def get_codebook(pid: str, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    proj, _ = member_or_403(s, user, pid)
    return proj.codebook or {"codeSystem": [], "coding": {"mode": "inductive"}}


@router.put("/{pid}/codebook")
def put_codebook(pid: str, body: dict, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    proj, _ = member_or_403(s, user, pid)
    cur = proj.codebook_rev or 0
    if "rev" in body and body["rev"] != cur:
        # the shared codebook was edited by another coder — refuse to silently overwrite
        raise HTTPException(409, {"message": "Das Codebook wurde zwischenzeitlich geändert.", "rev": cur})
    proj.codebook = {"codeSystem": body.get("codeSystem") or [], "coding": body.get("coding") or {}}
    proj.codebook_rev = cur + 1
    s.commit()
    return {"rev": proj.codebook_rev}


@router.get("/{pid}/documents")
def list_documents(pid: str, s: DBSession = Depends(get_db), user: db.User = Depends(require_user)):
    member_or_403(s, user, pid)
    docs = s.scalars(select(db.Document).where(db.Document.project_id == pid)
                     .order_by(db.Document.created_at)).all()
    return [{"id": d.id, "name": d.name, "rev": d.rev,
             "segments": len([x for x in (d.doc or {}).get("segments", []) if x.get("type") == "turn"]),
             "updated_at": d.updated_at.isoformat() if d.updated_at else None}
            for d in docs]
