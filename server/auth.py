"""Authentication: argon2 passwords + opaque session cookies.

Registration is open but restricted to an allowed e-mail domain
(ALLOWED_EMAIL_DOMAINS, default ait.ac.at). Sessions are random opaque tokens
stored server-side (revocable); the cookie is HttpOnly. No JWT, no e-mail
verification yet (a later hardening step).
"""
from __future__ import annotations

import os
import re
import secrets
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from server import db

_ph = PasswordHasher()
COOKIE_NAME = "sid"
COOKIE_SECURE = os.environ.get("COOKIE_SECURE") == "1"


def allowed_domains() -> list[str]:
    raw = os.environ.get("ALLOWED_EMAIL_DOMAINS", "ait.ac.at")
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def _email_ok(email: str) -> bool:
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False
    domain = email.rsplit("@", 1)[1].lower()
    return any(domain == d or domain.endswith("." + d) for d in allowed_domains())


def get_db():
    s = db.SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _new_session(s: DBSession, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    s.add(db.Session(token=token, user_id=user_id))
    s.commit()
    return token


def _set_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax",
                    secure=COOKIE_SECURE, max_age=30 * 24 * 3600, path="/")


def current_user(sid: Optional[str] = Cookie(default=None),
                 s: DBSession = Depends(get_db)) -> Optional[db.User]:
    if not sid:
        return None
    sess = s.get(db.Session, sid)
    if sess is None:
        return None
    if sess.expires_at and sess.expires_at.replace(tzinfo=db.now().tzinfo) < db.now():
        s.delete(sess)
        s.commit()
        return None
    return s.get(db.User, sess.user_id)


def require_user(user: Optional[db.User] = Depends(current_user)) -> db.User:
    if user is None:
        raise HTTPException(401, "Anmeldung erforderlich")
    return user


router = APIRouter(prefix="/api/auth")


@router.post("/register")
def register(body: dict, response: Response, s: DBSession = Depends(get_db)):
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    password = body.get("password") or ""
    if not _email_ok(email):
        raise HTTPException(400, f"Nur E-Mail-Adressen dieser Domain(s): {', '.join(allowed_domains())}")
    if len(password) < 8:
        raise HTTPException(400, "Passwort muss mindestens 8 Zeichen haben")
    if s.scalar(select(db.User).where(db.User.email == email)):
        raise HTTPException(409, "E-Mail ist bereits registriert")
    first_user = s.scalar(select(db.User).limit(1)) is None
    user = db.User(email=email, name=name or email.split("@")[0],
                   password_hash=_ph.hash(password), is_admin=first_user)
    s.add(user)
    s.commit()
    _set_cookie(response, _new_session(s, user.id))
    return user.public()


@router.post("/login")
def login(body: dict, response: Response, s: DBSession = Depends(get_db)):
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    user = s.scalar(select(db.User).where(db.User.email == email))
    if user is None:
        raise HTTPException(401, "E-Mail oder Passwort falsch")
    try:
        _ph.verify(user.password_hash, password)
    except VerifyMismatchError:
        raise HTTPException(401, "E-Mail oder Passwort falsch")
    if _ph.check_needs_rehash(user.password_hash):
        user.password_hash = _ph.hash(password)
        s.commit()
    _set_cookie(response, _new_session(s, user.id))
    return user.public()


@router.post("/logout")
def logout(response: Response, sid: Optional[str] = Cookie(default=None),
           s: DBSession = Depends(get_db)):
    if sid:
        sess = s.get(db.Session, sid)
        if sess:
            s.delete(sess)
            s.commit()
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: Optional[db.User] = Depends(current_user)):
    if user is None:
        raise HTTPException(401, "nicht angemeldet")
    return user.public()
