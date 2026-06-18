"""Server-side transcript parser — mirrors the tool's client-side parse() so a
stored Document holds the same v2 shared shape (stable segment ids 0..n, speaker
entities) that the frontend's installDoc() understands directly. Per-user codes
and the project codebook are merged in at load time (see documents.py).
"""
from __future__ import annotations

import re

PALETTE = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777',
           '#0891b2', '#65a30d', '#9333ea', '#e11d48', '#0d9488', '#ca8a04']
TURN = re.compile(r'^\[(\d{2}:\d{2}:\d{2})\]\s+(.*?):\s?([\s\S]*)$')


def _to_sec(t: str) -> int:
    p = [int(x) for x in t.split(':')]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def _is_divider(line: str) -> bool:
    t = line.strip()
    return bool(re.match(r'^={3,}', t) or re.match(r'^TEIL\s', t) or re.match(r'^ENDE\b', t))


def parse_transcript(raw: str, name: str = "") -> dict:
    segs: list[dict] = []
    header: list[str] = []
    started = False
    idn = 0
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = TURN.match(line)
        if m:
            started = True
            t = m.group(1)
            segs.append({"type": "turn", "id": idn, "time": t, "seconds": _to_sec(t),
                         "speaker": m.group(2).strip(), "text": m.group(3).strip(),
                         "verified": False, "edited": False})
            idn += 1
            continue
        bare_eq = re.match(r'^={3,}$', line.strip())
        if _is_divider(line) and not bare_eq:
            started = True
            segs.append({"type": "divider", "id": idn,
                         "label": re.sub(r'^=+\s*|\s*=+$', '', line).strip()})
            idn += 1
            continue
        if bare_eq:
            if not started:
                header.append(line)
            continue
        if not started:
            header.append(line)
        elif line.strip():
            if segs and segs[-1]["type"] == "turn":
                segs[-1]["text"] += " " + line.strip()

    speakers: list[dict] = []
    by_label: dict[str, dict] = {}
    ci = 0
    for s in segs:
        if s["type"] != "turn":
            continue
        label = s["speaker"] or "?"
        e = by_label.get(label)
        if e is None:
            color = "#6b7280" if label == "UNKNOWN" else PALETTE[ci % len(PALETTE)]
            ci += 1
            e = {"id": f"spk_{len(speakers)}", "label": label, "color": color, "role": "",
                 "aliases": [label] if re.match(r'^SPEAKER_|^UNKNOWN$', label) else []}
            by_label[label] = e
            speakers.append(e)
        s["speakerId"] = e["id"]

    return {"schemaVersion": 2, "name": name, "header": "\n".join(header).strip(),
            "speakers": speakers, "segments": segs}
