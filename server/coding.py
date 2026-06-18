"""LLM coding job runner.

Codes each focal segment with ±context neighbours as read-only context, using the
pluggable LLM backend. Validates every returned code: the code_id must be in the
codebook and the quote must be a verbatim substring of the focal segment — only
then is it emitted (with server-computed char offsets). Suggestions come back as
status='suggested' for the human to review (anti-anchoring), never auto-applied.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from server import llm


def run(payload: dict, progress: Callable[[str, float], None]) -> dict:
    codes = [c for c in (payload.get("codes") or []) if c.get("isCodable") is not False]
    segments = payload.get("segments") or []
    valid_ids = {c["id"] for c in codes}
    ctx = int(payload.get("context", 1))
    provider = payload.get("provider") or "fake"
    model = payload.get("model") or ""
    concurrency = int(payload.get("concurrency") or (1 if provider == "ollama" else 4))

    if not codes:
        raise ValueError("codebook is empty — define codes before coding")
    if not segments:
        raise ValueError("no segments to code")

    total = len(segments)
    suggestions: list[dict] = []
    suggested_new: list[dict] = []
    stats = {"segments": total, "suggested": 0, "invalid_quotes": 0, "unknown_codes": 0, "errors": 0}
    done = 0

    def code_one(i: int) -> dict:
        focal = segments[i]
        before = segments[max(0, i - ctx):i]
        after = segments[i + 1:i + 1 + ctx]
        system, user = llm.build_messages(codes, focal, before, after)
        raw = llm.complete(provider, model, system, user)
        return {"i": i, "focal": focal, "parsed": llm.parse_response(raw)}

    progress("code", 0.0)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(code_one, i) for i in range(total)]
        for fut in as_completed(futures):
            done += 1
            try:
                out = fut.result()
            except Exception:  # noqa: BLE001 - one bad segment shouldn't kill the job
                stats["errors"] += 1
                progress("code", done / total)
                continue
            focal, parsed = out["focal"], out["parsed"]
            text = focal.get("text", "") or ""
            for c in (parsed.get("codes") or []):
                cid, quote = c.get("code_id"), (c.get("quote") or "")
                if cid not in valid_ids:
                    stats["unknown_codes"] += 1
                    continue
                pos = text.find(quote) if quote else -1
                if pos < 0:
                    stats["invalid_quotes"] += 1
                    continue
                suggestions.append({
                    "segment_id": focal.get("id"),
                    "code_id": cid,
                    "quote": quote,
                    "char_start": pos,
                    "char_end": pos + len(quote),
                    "rationale": (c.get("rationale") or "").strip(),
                    "confidence": c.get("confidence"),
                })
                stats["suggested"] += 1
            snc = parsed.get("suggested_new_code")
            if snc and snc.get("name"):
                suggested_new.append({"name": snc["name"], "rationale": snc.get("rationale", ""),
                                      "segment_id": focal.get("id")})
            progress("code", done / total)

    progress("done", 1.0)
    return {"suggestions": suggestions, "suggested_new_codes": suggested_new,
            "stats": stats, "provider": provider, "model": model}
