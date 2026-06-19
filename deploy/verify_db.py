"""Post-deploy DB smoke test — run this against your PRODUCTION database to prove
the schema and optimistic-locking behave correctly on whatever DATABASE_URL points
at (especially PostgreSQL, which the dev box doesn't exercise).

It uses the real app via TestClient, so it also confirms create_all built every
table/column. It writes throwaway rows (two test users + a project + a doc) — run
it against a fresh/empty DB, or clean up afterwards.

Usage (PowerShell, from the repo root, conda env active):
    $env:DATABASE_URL="postgresql+psycopg://tcuser:PW@127.0.0.1:5432/tcdb"
    $env:ALLOWED_EMAIL_DOMAINS="ait.ac.at"
    python deploy/verify_db.py

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Pick up DATABASE_URL from the repo-root .env, just like the app does, so this
# works out of the box on the native deploy without re-exporting the env var.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if os.environ.get("DATABASE_URL", "").startswith("sqlite") or not os.environ.get("DATABASE_URL"):
    print("WARNING: DATABASE_URL is not set to a non-sqlite DB — this test is meant "
          "to validate your production (e.g. Postgres) database.\n"
          "Set DATABASE_URL first. Aborting to avoid a misleading 'pass'.")
    sys.exit(2)

from fastapi.testclient import TestClient  # noqa: E402
from server.app import app  # noqa: E402


def main() -> int:
    print(f"Verifying against: {os.environ['DATABASE_URL'].split('@')[-1]}")
    cA, cB = TestClient(app), TestClient(app)
    fails: list[str] = []

    def ck(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    import uuid
    suf = uuid.uuid4().hex[:8]  # unique emails so re-runs don't collide on the unique index
    a_mail, b_mail = f"alice_{suf}@ait.ac.at", f"bob_{suf}@ait.ac.at"

    ck(cA.post("/api/auth/register", json={"email": a_mail, "name": "A", "password": "passw0rd!"}).status_code == 200,
       "register A")
    ck(cB.post("/api/auth/register", json={"email": b_mail, "name": "B", "password": "passw0rd!"}).status_code == 200,
       "register B")
    pid = cA.post("/api/projects", json={"name": f"verify-{suf}"}).json()["id"]
    did = cA.post(f"/api/projects/{pid}/documents",
                  json={"name": "doc", "text": "[00:00:01] S1: hallo welt.\n[00:00:05] S2: ja."}).json()["id"]
    ck(cA.post(f"/api/projects/{pid}/members", json={"email": b_mail, "role": "coder"}).status_code == 200,
       "add B as member")

    d = cA.get(f"/api/documents/{did}").json()
    ck(d.get("rev") == 0 and d.get("layerRev") == 0 and d.get("codebookRev") == 0,
       f"baseline revs 0/0/0 (got {d.get('rev')}/{d.get('layerRev')}/{d.get('codebookRev')})")

    lb = {"codeApplications": [{"id": "x"}], "highlights": [], "comments": []}
    ck(cA.put(f"/api/documents/{did}/layer", json={**lb, "rev": 0}).json().get("rev") == 1, "layer rev0 -> 1")
    ck(cA.put(f"/api/documents/{did}/layer", json={**lb, "rev": 0}).status_code == 409, "layer stale rev0 -> 409")
    ck(cB.get(f"/api/documents/{did}").json().get("codeApplications") == [], "per-user layer isolation (B empty)")

    cb = {"codeSystem": [{"id": "c1"}], "coding": {"mode": "inductive"}}
    ck(cA.put(f"/api/projects/{pid}/codebook", json={**cb, "rev": 0}).json().get("rev") == 1, "codebook rev0 -> 1")
    ck(cB.put(f"/api/projects/{pid}/codebook", json={**cb, "rev": 0}).status_code == 409, "codebook stale rev0 -> 409")

    ck(cA.put(f"/api/documents/{did}/text", json={"speakers": [], "segments": [], "header": "h", "rev": 0}).json().get("rev") == 1,
       "text rev0 -> 1")
    ck(cA.put(f"/api/documents/{did}/text", json={"speakers": [], "segments": [], "header": "h", "rev": 0}).status_code == 409,
       "text stale rev0 -> 409")

    print("\n" + ("ALL PASS — database is good." if not fails else f"{len(fails)} FAILED: " + "; ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
