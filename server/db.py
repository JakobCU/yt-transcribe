"""Database layer (SQLAlchemy 2.0).

SQLite to start (single file, zero ops); the schema is Postgres-compatible so the
team can switch with one URL change once there are concurrent writers (see the
brief). Source of truth for the team server lives here instead of the browser's
localStorage.

Sharing model: the corrected transcript text (Document.doc) is shared per
project; codes/highlights/comments are PER USER (UserLayer), so several coders
annotate the same material independently. The codebook is shared per project.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (JSON, DateTime, ForeignKey, Integer, String, create_engine,
                        UniqueConstraint)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'app.db')}")

engine = create_engine(
    DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def now() -> datetime:
    return datetime.now(timezone.utc)


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: uid("usr"))
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String, default="")
    password_hash: Mapped[str] = mapped_column(String)
    is_admin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name, "is_admin": self.is_admin}


class Session(Base):
    __tablename__ = "sessions"
    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: now() + timedelta(days=30))


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: uid("prj"))
    name: Mapped[str] = mapped_column(String)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    codebook: Mapped[dict] = mapped_column(JSON, default=dict)  # {codeSystem, coding}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

    members: Mapped[list["ProjectMember"]] = relationship(cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(cascade="all, delete-orphan")


class ProjectMember(Base):
    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "user_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(String, default="coder")  # admin | coder


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: uid("doc"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    doc: Mapped[dict] = mapped_column(JSON, default=dict)   # shared: schemaVersion, speakers, segments, header, media
    audio: Mapped[str] = mapped_column(String, default="")  # stored compact audio filename (in media/audio/), '' = none
    rev: Mapped[int] = mapped_column(Integer, default=0)    # optimistic lock token for the shared text
    created_by: Mapped[str] = mapped_column(String, default="")
    updated_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)


class UserLayer(Base):
    __tablename__ = "user_layers"
    __table_args__ = (UniqueConstraint("document_id", "user_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    code_applications: Mapped[list] = mapped_column(JSON, default=list)
    highlights: Mapped[list] = mapped_column(JSON, default=list)
    comments: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate()


def _migrate() -> None:
    """Lightweight additive migrations for existing SQLite DBs (add new columns)."""
    if not DB_URL.startswith("sqlite"):
        return
    with engine.begin() as conn:
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(documents)")}
        if "audio" not in cols:
            conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN audio VARCHAR DEFAULT ''")
