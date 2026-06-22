"""Pluggable LLM backend for deductive qualitative coding.

Providers (switchable per request):
  - claude  : Anthropic API, key from ANTHROPIC_API_KEY (server-side only)
  - ollama  : local model at OLLAMA_HOST (default http://127.0.0.1:11434) — data
              stays on the machine, the privacy-friendly default
  - fake    : deterministic canned suggestions (LLM_FAKE=1) for testing without
              a key / GPU

The model applies ONLY codes from the researcher's codebook (deductive), must
ground every code in a verbatim quote, and gives a short rationale before the
decision (chain-of-thought). We compute character offsets from the quote
server-side and reject any code whose quote is not a substring of the segment —
the primary anti-hallucination guard.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def available_providers() -> dict:
    """What the frontend can offer. Cheap checks only."""
    providers = {}
    providers["claude"] = {
        "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "default_model": DEFAULT_CLAUDE_MODEL,
        "label": "Claude (Cloud)",
    }
    providers["openai"] = {
        "available": bool(os.environ.get("OPENAI_API_KEY")),
        "default_model": os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "label": "OpenAI (Cloud)",
    }
    providers["ollama"] = {
        "available": _ollama_reachable(),
        "default_model": os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        "label": "Lokal (Ollama)",
    }
    if os.environ.get("LLM_FAKE") == "1":
        providers["fake"] = {"available": True, "default_model": "fake", "label": "Test (Fake)"}
    return providers


def _ollama_reachable() -> bool:
    try:
        import requests
        requests.get(ollama_host() + "/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def _format_codebook(codes: list[dict]) -> str:
    lines = []
    for c in codes:
        if c.get("isCodable") is False:
            continue
        lines.append(f"- id: {c['id']}")
        lines.append(f"  name: {c.get('name', c['id'])}")
        if c.get("definition"):
            lines.append(f"  definition: {c['definition']}")
        if c.get("inclusion"):
            lines.append(f"  apply when: {c['inclusion']}")
        if c.get("exclusion"):
            lines.append(f"  do NOT apply when: {c['exclusion']}")
        ex = c.get("examples") or []
        if ex:
            lines.append("  examples: " + " | ".join(str(e) for e in ex))
    return "\n".join(lines)


_COMMON = (
    "- A segment may get zero, one, or several codes. Do not over-code: if nothing"
    " substantive fits (greetings, filler, procedural chatter), return no codes.\n"
    "- For every code you MUST quote the exact verbatim span from the FOCAL segment"
    " (copied character-for-character, no paraphrase).\n"
    "- Give a one-sentence rationale BEFORE deciding (think, then decide).\n"
    "- Neighbouring context is provided only to resolve references; code ONLY the focal segment.\n"
    "Respond with STRICT JSON only, no prose."
)

_SYS_DEDUCTIVE = (
    "You are a meticulous qualitative coder applying a FIXED codebook to interview transcript segments.\n"
    "- Apply ONLY codes defined in the codebook below, using their exact id in \"code_id\". Never invent codes.\n"
    + _COMMON + "\n"
    '{"codes":[{"code_id":"<id from codebook>","quote":"<verbatim span>","rationale":"<one sentence>","confidence":0.0}],"no_code":false}'
)

_SYS_INDUCTIVE = (
    "You are doing INDUCTIVE (open) qualitative coding. There is NO predefined codebook.\n"
    "- For the focal segment, identify the key theme(s) and assign each a SHORT code name"
    " (2-5 words, Title Case) that captures what it is about, in the transcript's language.\n"
    "- Reuse the SAME wording for recurring themes so codes stay consistent across segments.\n"
    + _COMMON + "\n"
    '{"codes":[{"name":"<short theme name>","quote":"<verbatim span>","rationale":"<one sentence>","confidence":0.0}],"no_code":false}'
)

_SYS_HYBRID = (
    "You are a qualitative coder with a seed codebook (below).\n"
    "- Apply a codebook code when one clearly fits (its exact id in \"code_id\").\n"
    "- If the segment shows a relevant theme NOT covered by the codebook, instead propose a SHORT"
    " new code name (2-5 words) in \"name\" and leave code_id null. Reuse consistent names.\n"
    + _COMMON + "\n"
    '{"codes":[{"code_id":"<id or null>","name":"<new name or null>","quote":"<verbatim span>","rationale":"<one sentence>","confidence":0.0}],"no_code":false}'
)

_SYS_BY_MODE = {"inductive": _SYS_INDUCTIVE, "hybrid": _SYS_HYBRID, "deductive": _SYS_DEDUCTIVE}


def build_messages(codes, focal, before, after, mode="deductive"):
    def fmt(seg):
        return f"[{seg.get('speaker', '?')}] {seg.get('text', '')}"

    system = _SYS_BY_MODE.get(mode, _SYS_DEDUCTIVE)
    parts = []
    if mode != "inductive" and codes:
        parts.append("CODEBOOK:\n" + _format_codebook(codes))
    if before:
        parts.append("Context before (do not code):\n" + "\n".join(fmt(s) for s in before))
    if after:
        parts.append("Context after (do not code):\n" + "\n".join(fmt(s) for s in after))
    parts.append(
        "FOCAL SEGMENT TO CODE:\n"
        + f"speaker: {focal.get('speaker', '?')}\n"
        + f"text: {focal.get('text', '')}\n\nReturn the JSON now."
    )
    return system, "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Batched inductive/hybrid with a RUNNING codebook (codes accumulate) +
# a final consolidation pass that merges near-duplicate emergent codes.
# --------------------------------------------------------------------------- #

_SYS_BATCH = (
    "You assist qualitative coding — you propose CANDIDATE codes only; a human reviews and decides.\n"
    "You are coding a BATCH of interview segments. A running CODEBOOK SO FAR is given.\n"
    "- REUSE an existing code (its EXACT name) whenever it fits — this is essential so codes accumulate"
    " consistently across the whole transcript. Create a NEW code ONLY for a genuinely new theme:"
    " a SHORT name (2-5 words). Reuse beats inventing.\n"
    "- Name every code in the SAME LANGUAGE as the segment texts — do NOT translate. If the interview is"
    " in German, the code names MUST be German.\n"
    "- A segment may get zero, one, or several codes. Do not over-code: greetings, filler and procedural"
    " chatter get no code.\n"
    "- For every code, quote the EXACT verbatim span from THAT segment (copied character-for-character).\n"
    "- One short rationale per code (think, then decide). Code ONLY the listed segments.\n"
    "Respond with STRICT JSON only, no prose:\n"
    '{"segments":[{"id":"<segment id>","codes":[{"code":"<existing or new name>","quote":"<verbatim span>","rationale":"<one sentence>","confidence":0.0}]}]}'
)

_SYS_CONSOLIDATE = (
    "You consolidate a list of qualitative codes by merging NEAR-DUPLICATES that mean the same thing into"
    " one canonical code. Pick the clearest canonical name (prefer one already in the list). Do NOT merge"
    " codes that capture genuinely distinct themes — when in doubt, keep them separate.\n"
    'Respond with STRICT JSON only: {"merges":[{"canonical":"<name>","variants":["<other name>","..."]}]}'
)


def build_batch_messages(seed_codes, running_codes, segments, mode="inductive"):
    cb = []
    for c in (seed_codes or []):
        if c.get("isCodable") is False:
            continue
        d = c.get("definition")
        cb.append(f"- {c.get('name', c['id'])}" + (f" — {d}" if d else "") + "  [seed]")
    for r in (running_codes or []):
        cb.append(f"- {r['name']}  (bereits {r.get('count', 0)}x verwendet)")
    parts = [
        "CODEBOOK SO FAR (reuse these EXACT names when they fit):\n"
        + ("\n".join(cb) if cb else "—(noch leer — lege passende neue Codes an)"),
        "SEGMENTS TO CODE (code each by its id):\n"
        + "\n".join(f"[{s.get('id')}] ({s.get('speaker', '?')}): {s.get('text', '')}" for s in segments),
        "Return the JSON now.",
    ]
    return _SYS_BATCH, "\n\n".join(parts)


def build_consolidation_messages(running_codes):
    lst = "\n".join(f"- {r['name']}  ({r.get('count', 0)}x)" for r in running_codes)
    return _SYS_CONSOLIDATE, ("CODES (name · uses):\n" + lst
                              + "\n\nReturn the merges JSON now (empty list if nothing should be merged).")


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

def complete(provider: str, model: str, system: str, user: str, temperature: float = 0.1) -> str:
    if provider == "fake":
        return _fake_complete(user)
    if provider == "claude":
        return _claude_complete(model or DEFAULT_CLAUDE_MODEL, system, user, temperature)
    if provider == "openai":
        return _openai_complete(model or DEFAULT_OPENAI_MODEL, system, user, temperature)
    if provider == "ollama":
        return _ollama_complete(model or DEFAULT_OLLAMA_MODEL, system, user, temperature)
    raise ValueError(f"unknown provider {provider!r}")


def _claude_complete(model: str, system: str, user: str, temperature: float) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _openai_complete(model: str, system: str, user: str, temperature: float) -> str:
    import requests
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    r = requests.post(
        base + "/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _ollama_complete(model: str, system: str, user: str, temperature: float) -> str:
    import requests
    r = requests.post(
        ollama_host() + "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def _fake_complete(user: str) -> str:
    """Deterministic canned responses for testing without a model — handles all
    three prompt shapes (batched running-codebook, consolidation, single-segment
    deductive), always quoting a real substring so validation passes."""
    # 1) batched inductive/hybrid: one emergent code per segment, REUSING a single
    #    name so codes accumulate (demonstrates the running-codebook behaviour).
    if "SEGMENTS TO CODE" in user:
        segs = re.findall(r"^\[([^\]]+)\] \([^)]*\): (.+)$", user, re.MULTILINE)
        out = []
        for sid, text in segs:
            t = text.strip()
            if len(t) < 8:
                continue
            out.append({"id": sid, "codes": [{"code": "Fake-Thema",
                       "quote": t[: min(40, len(t))].strip(),
                       "rationale": "Fake-Coder.", "confidence": 0.5}]})
        return json.dumps({"segments": out})
    # 2) consolidation pass: nothing to merge in the fake.
    if user.lstrip().startswith("CODES (name"):
        return json.dumps({"merges": []})
    # 3) single-segment deductive (legacy build_messages format).
    ids = re.findall(r"^- id: (\S+)", user, re.MULTILINE)
    m = re.search(r"text: (.+)$", user, re.MULTILINE)
    text = m.group(1).strip() if m else ""
    if len(text) < 8:
        return json.dumps({"codes": [], "no_code": True})
    quote = text[: min(40, len(text))].strip()
    item = {"quote": quote, "rationale": "Fake-Coder zur Demonstration.", "confidence": 0.5}
    if ids:
        item["code_id"] = ids[0]
    else:
        item["name"] = "Fake-Thema"
    return json.dumps({"codes": [item], "no_code": False})


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_response(text: str) -> dict:
    """Extract the JSON object from a model response (tolerant of code fences/prose)."""
    if not text:
        return {"codes": [], "no_code": True, "suggested_new_code": None}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except Exception:
        # grab the first {...} block
        start, depth = s.find("{"), 0
        if start >= 0:
            for i in range(start, len(s)):
                depth += 1 if s[i] == "{" else (-1 if s[i] == "}" else 0)
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
    return {"codes": [], "no_code": True, "suggested_new_code": None, "_parse_error": True}
