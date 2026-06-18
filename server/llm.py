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


SYSTEM_PROMPT = """\
You are a meticulous qualitative coder applying a FIXED codebook to interview transcript segments.
Rules:
- Apply ONLY codes defined in the codebook below. Never invent codes (you may *suggest* one separately).
- A segment may get zero, one, or several codes. Do not over-code: if nothing clearly fits, return no codes.
- For every code you apply you MUST quote the exact verbatim span from the FOCAL segment that justifies it (copied character-for-character, no paraphrase).
- Give a one-sentence rationale tied to the code's definition BEFORE deciding (think, then decide).
- Neighbouring context is provided only to resolve references; code ONLY the focal segment.
Respond with STRICT JSON only, no prose, in exactly this shape:
{"codes":[{"code_id":"<id from codebook>","quote":"<verbatim span from focal segment>","rationale":"<one sentence>","confidence":0.0}],"no_code":false,"suggested_new_code":null}
If nothing fits, return {"codes":[],"no_code":true,"suggested_new_code":null}.
You may set suggested_new_code to {"name":"...","rationale":"..."} to propose an emergent theme (it will NOT be applied automatically)."""


def build_messages(codes: list[dict], focal: dict, before: list[dict], after: list[dict]) -> tuple[str, str]:
    def fmt(seg):
        return f"[{seg.get('speaker', '?')}] {seg.get('text', '')}"

    ctx = []
    if before:
        ctx.append("Context before (do not code):\n" + "\n".join(fmt(s) for s in before))
    if after:
        ctx.append("Context after (do not code):\n" + "\n".join(fmt(s) for s in after))
    ctx_block = ("\n\n".join(ctx) + "\n\n") if ctx else ""

    user = (
        "CODEBOOK:\n" + _format_codebook(codes) + "\n\n"
        + ctx_block
        + "FOCAL SEGMENT TO CODE:\n"
        + f"speaker: {focal.get('speaker', '?')}\n"
        + f"text: {focal.get('text', '')}\n\n"
        + "Return the JSON now."
    )
    return SYSTEM_PROMPT, user


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

def complete(provider: str, model: str, system: str, user: str, temperature: float = 0.1) -> str:
    if provider == "fake":
        return _fake_complete(user)
    if provider == "claude":
        return _claude_complete(model or DEFAULT_CLAUDE_MODEL, system, user, temperature)
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
    """Deterministic suggestion: code the focal segment with the first codebook
    code whose name/keyword appears, quoting a real substring (so substring
    validation passes). Exercises the whole loop without a model."""
    # crude parse of the prompt we built
    ids = re.findall(r"^- id: (\S+)", user, re.MULTILINE)
    m = re.search(r"text: (.+)$", user, re.MULTILINE)
    text = m.group(1).strip() if m else ""
    if not ids or len(text) < 8:
        return json.dumps({"codes": [], "no_code": True, "suggested_new_code": None})
    quote = text[: min(40, len(text))].strip()
    return json.dumps({
        "codes": [{"code_id": ids[0], "quote": quote,
                   "rationale": "Fake-Coder: erste Codebook-Kategorie zur Demonstration.",
                   "confidence": 0.5}],
        "no_code": False, "suggested_new_code": None,
    })


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
