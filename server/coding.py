"""LLM coding job runner (candidate codes only — a human reviews & decides).

Two paths, picked by mode:

* deductive — each segment coded independently (parallel) against the FIXED
  codebook; reuse needs no memory, so independence is fine.
* inductive / hybrid — segments coded in BATCHES, SEQUENTIALLY, carrying a
  "running codebook" forward so the model reuses existing codes instead of
  inventing a new one per segment (the documented failure mode in the
  LLM-thematic-analysis literature: De Paoli & Mathis 2024 — 534 codes → 66
  unique). A final consolidation pass merges near-duplicate emergent codes.

Every returned code is grounded in a VERBATIM quote that must be a substring of
its segment (server-side anti-hallucination); only then is it emitted with
server-computed char offsets, as status='suggested' for human review.
Each run records an audit/repro record (provider, model, temperature, prompt
version, mode, counts) per Lincoln & Guba dependability/confirmability.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

from server import llm

TEMPERATURE = 0.1          # low temp for reproducibility (Xiao et al.; Breazu et al.)
BATCH = 12                 # segments per inductive/hybrid batch
PROMPT_VERSION = "tc-coding/v2-running-codebook"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(payload: dict, progress: Callable[[str, float], None]) -> dict:
    codes = [c for c in (payload.get("codes") or []) if c.get("isCodable") is not False]
    segments = payload.get("segments") or []
    mode = payload.get("mode") or "deductive"
    provider = payload.get("provider") or "fake"
    model = payload.get("model") or ""

    if mode == "deductive" and not codes:
        raise ValueError("codebook is empty — define codes or switch to inductive coding")
    if not segments:
        raise ValueError("no segments to code")

    started = _now()
    if mode == "deductive":
        result = _run_deductive(payload, progress, codes, segments, provider, model)
    else:
        result = _run_running_codebook(payload, progress, codes, segments, mode, provider, model)

    # audit / reproducibility record (Lincoln & Guba dependability + confirmability)
    result["run"] = {
        "provider": provider, "model": model, "temperature": TEMPERATURE,
        "mode": mode, "prompt_version": PROMPT_VERSION,
        "batch_size": BATCH if mode != "deductive" else None,
        "segments": len(segments), "started": started, "finished": _now(),
        "stats": result.get("stats", {}),
    }
    result["provider"] = provider
    result["model"] = model
    progress("done", 1.0)
    return result


# --------------------------------------------------------------------------- #
# deductive — independent per-segment coding against the fixed codebook
# --------------------------------------------------------------------------- #

def _run_deductive(payload, progress, codes, segments, provider, model) -> dict:
    valid_ids = {c["id"] for c in codes}
    ctx = int(payload.get("context", 1))
    concurrency = int(payload.get("concurrency") or (1 if provider == "ollama" else 4))
    total = len(segments)
    suggestions: list[dict] = []
    stats = {"segments": total, "suggested": 0, "invalid_quotes": 0, "unknown_codes": 0, "errors": 0, "merged": 0}
    done = 0

    def code_one(i: int) -> dict:
        focal = segments[i]
        before = segments[max(0, i - ctx):i]
        after = segments[i + 1:i + 1 + ctx]
        system, user = llm.build_messages(codes, focal, before, after, "deductive")
        raw = llm.complete(provider, model, system, user, TEMPERATURE)
        return {"focal": focal, "parsed": llm.parse_response(raw)}

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
                cid = c.get("code_id")
                quote = c.get("quote") or ""
                pos = text.find(quote) if quote else -1
                if pos < 0:
                    stats["invalid_quotes"] += 1
                    continue
                if not (cid and cid in valid_ids):
                    stats["unknown_codes"] += 1
                    continue
                suggestions.append({
                    "segment_id": focal.get("id"), "quote": quote,
                    "char_start": pos, "char_end": pos + len(quote),
                    "rationale": (c.get("rationale") or "").strip(),
                    "confidence": c.get("confidence"), "code_id": cid,
                })
                stats["suggested"] += 1
            progress("code", done / total)

    return {"suggestions": suggestions, "suggested_new_codes": [], "stats": stats}


# --------------------------------------------------------------------------- #
# inductive / hybrid — batched, running codebook, then consolidation
# --------------------------------------------------------------------------- #

def _run_running_codebook(payload, progress, seed_codes, segments, mode, provider, model) -> dict:
    seed_by_name = {(c.get("name") or "").strip().lower(): c for c in seed_codes if c.get("name")}
    seg_by_id = {str(s.get("id")): s for s in segments}   # ids may be ints; the model echoes them as JSON strings
    running: dict[str, dict] = {}          # lower name -> {"name", "count"}
    suggestions: list[dict] = []
    stats = {"segments": len(segments), "suggested": 0, "invalid_quotes": 0,
             "unknown_codes": 0, "errors": 0, "merged": 0, "emergent_codes": 0}

    batches = [segments[i:i + BATCH] for i in range(0, len(segments), BATCH)]
    progress("code", 0.0)
    for bi, batch in enumerate(batches):
        try:
            system, user = llm.build_batch_messages(seed_codes, list(running.values()), batch, mode)
            parsed = llm.parse_response(llm.complete(provider, model, system, user, TEMPERATURE))
        except Exception:  # noqa: BLE001 - one bad batch shouldn't kill the job
            stats["errors"] += 1
            progress("code", (bi + 1) / len(batches) * 0.9)
            continue
        for sr in (parsed.get("segments") or parsed.get("assignments") or []):
            raw_id = sr.get("id")
            raw_id = sr.get("segment_id") if raw_id is None else raw_id
            focal = seg_by_id.get(str(raw_id))
            if not focal:
                continue
            text = focal.get("text", "") or ""
            fid = focal.get("id")                         # original id (int or str) for the client
            for c in (sr.get("codes") or []):
                name = (c.get("code") or c.get("name") or "").strip()
                quote = c.get("quote") or ""
                pos = text.find(quote) if quote else -1
                if pos < 0:
                    stats["invalid_quotes"] += 1
                    continue
                if not name:
                    stats["unknown_codes"] += 1
                    continue
                sug = {
                    "segment_id": fid, "quote": quote,
                    "char_start": pos, "char_end": pos + len(quote),
                    "rationale": (c.get("rationale") or "").strip(),
                    "confidence": c.get("confidence"),
                }
                low = name.lower()
                if low in seed_by_name:           # reused a seed (hybrid) code
                    sug["code_id"] = seed_by_name[low]["id"]
                else:                             # emergent code (carried forward by name)
                    sug["code_name"] = name
                    r = running.get(low) or {"name": name, "count": 0}
                    r["count"] += 1
                    running[low] = r
                suggestions.append(sug)
                stats["suggested"] += 1
        progress("code", (bi + 1) / len(batches) * 0.9)

    # consolidation pass — merge near-duplicate emergent codes into canonical ones
    if len(running) >= 2:
        try:
            cmap = _consolidate(provider, model, list(running.values()))
        except Exception:  # noqa: BLE001
            cmap = {}
        if cmap:
            merged = set()
            for s in suggestions:
                nm = s.get("code_name")
                if nm and nm.lower() in cmap:
                    merged.add(nm.lower())
                    s["code_name"] = cmap[nm.lower()]
            stats["merged"] = len(merged)
    stats["emergent_codes"] = len({s["code_name"] for s in suggestions if s.get("code_name")})

    return {"suggestions": suggestions, "suggested_new_codes": [], "stats": stats}


def _consolidate(provider, model, running_codes) -> dict:
    """Ask the model to group synonymous emergent codes. Returns {variant_lower: canonical}."""
    system, user = llm.build_consolidation_messages(running_codes)
    parsed = llm.parse_response(llm.complete(provider, model, system, user, TEMPERATURE))
    cmap: dict[str, str] = {}
    for m in (parsed.get("merges") or []):
        canon = (m.get("canonical") or "").strip()
        if not canon:
            continue
        for v in (m.get("variants") or []):
            vl = (v or "").strip().lower()
            if vl and vl != canon.lower():
                cmap[vl] = canon
    return cmap
