<!-- Generated 2026-06-18 from a 5-dimension research workflow. Drives the phased build. -->

# Self-Hosted Transcript Correction + LLM Qualitative Coding Tool — Consolidated Engineering Brief

This brief merges five research dimensions into a single decision-ready document. Where agents disagreed, the pragmatic choice is stated with rationale. The guiding constraint: **Phase 1 ships client-side editor features in the existing single-file HTML tool before any backend exists.**

---

## 1. Editor Feature Backlog (prioritized, deduped)

Legend: **[EXISTS]** already in current tool · **[NEW]** to build · **[CS]** doable client-side only (no backend) · **[BE]** needs backend.

### P0 — Core correction loop (all client-side; this is Phase 1)

| Feature | Status | Notes |
|---|---|---|
| Speaker rename → propagate to all instances | **[EXISTS][CS]** | Keep; back it with speaker entities (see §4) |
| Verify / mark-segment state | **[EXISTS][CS]** | Extend into review states (unverified → verified) |
| Inline `[?? ...]` markers | **[EXISTS][CS]** | Keep, but do NOT reuse this mechanism for span anchors (§4) |
| In-transcript search | **[EXISTS][CS]** | Extend to Find & Replace + "Replace All" |
| Audio-sync (click word → seek) | **[EXISTS][CS]** | Foundation for everything below |
| Multi-format export (txt/srt) | **[EXISTS][CS]** | Extend with JSON round-trip + metadata options |
| **Inline click-to-edit** (no edit-mode toggle) | **[NEW][CS]** | The core editing model |
| **`Tab` = play/pause in place**, `Shift+Tab` = rewind 5s | **[NEW][CS]** | Single highest-value ergonomic |
| **Auto-rewind-on-pause** (~1–2s on resume) | **[NEW][CS]** | Cheap, big productivity win (oTranscribe pattern) |
| **Variable speed 0.5x–1.5x** (incl. slow-down) | **[NEW][CS]** | Correction range, not HappyScribe's 5x |
| **Karaoke follow-along highlight** + auto-scroll | **[NEW][CS]** | Requires word-level timestamps (`words[]`) |
| **Find & Replace** (`Ctrl+E`, Replace All) | **[NEW][CS]** | Fixes recurring name misspellings in one pass |
| **Auto-save** (localStorage now; debounced) + saved indicator | **[NEW][CS]** | Non-negotiable; server-backed later |
| **Undo / Redo** | **[NEW][CS]** | Table stakes |
| **Merge speakers** (collapse Speaker 3 → Speaker 1, reassign all) | **[NEW][CS]** | Genuine market gap; diarization over-splits constantly |
| **Reassign speaker per segment** | **[NEW][CS]** | Click label → pick/add/unassign |
| **Speaker color coding** in highlight | **[EXISTS/partial][CS]** | Already have `speakerColors`; extend to karaoke |

### P1 — Correction efficiency + annotation (mostly client-side)

| Feature | Status | Notes |
|---|---|---|
| **Confidence gradient highlighting** (continuous shading, not binary red) | **[NEW][CS]** | Steal Sonix's "thermometer" over HappyScribe's red. Needs ASR confidence in data (`asrConfidence`) |
| **"Jump to next low-confidence word" hotkey** | **[NEW][CS]** | Real differentiator — neither competitor exposes this |
| **Highlights (multi-color spans)** | **[NEW][CS]** | 3 colors for triage; uses shared `anchor` (§4) |
| **Comments anchored to text/segment** | **[NEW][CS]** | Uses shared `anchor`; thread-ready |
| **Floating selection toolbar** (drag-select → highlight/comment) | **[NEW][CS]** | UI affordance for the above |
| **Version history** (snapshots, restore) | **[NEW][CS→BE]** | localStorage snapshots now; server-backed later |
| **Export with metadata** (timecodes, speaker labels, highlights-only digest with `[HH:MM:SS]` stamps) | **[NEW][CS]** | DOCX/PDF can wait; JSON + enriched txt/srt first |
| **Customizable keyboard shortcuts** | **[NEW][CS]** | Honoring hotkeys = de-facto USB foot-pedal support for free |
| **Per-segment review state** (needs-review → approved + reviewer) | **[NEW][CS→BE]** | Gap across all four competitors; high value for correction tool |

### P2 — Collaboration + power features (backend-dependent)

| Feature | Status | Notes |
|---|---|---|
| Shareable secure links (view/edit) | **[NEW][BE]** | Needs auth |
| Roles & permissions | **[NEW][BE]** | admin/coder (§5) |
| Real-time multi-user editing / presence | **[NEW][BE]** | Use optimistic concurrency, NOT CRDT/locking (§5) |
| Glossary / custom vocabulary | **[NEW][CS]** | Lower priority for pure correction |
| Tags / folders (org layer) | **[NEW][BE]** | Project-level organization |
| Descript-style delete-word-to-mute / filler removal | **[NEW][CS]** | Power feature; defer — not core to correction |

**Opinionated cut:** Drop real-time collaborative editing (OT/CRDT) entirely from scope. For a 6-person team, per-user coding layers + optimistic version-check on text edits (§5) covers the real conflicts at a fraction of the complexity.

---

## 2. Qualitative Coding Layer

### Data model (entities + fields)

Built to the **REFI-QDA standard** so you get import/export interop with MAXQDA/ATLAS.ti/NVivo for free. Use neutral internal names; map on export.

- **Code** — `id`(UUID), `name`, `parent_id` (self-FK hierarchy, support ~10 levels), `color` (hex `#RRGGBB`), `definition` (the codebook entry), `is_codable` (bool — category nodes set false so they can't be applied to text), `created_at/by`, `modified_at`. **Build hierarchy as primary** (REFI supports true nesting; don't copy ATLAS.ti/NVivo flat-list+sets).
- **CodedSegment / CodeApplication** (the heart — join of code ↔ span):
  - `id`, `code_id`, `document_id`, span anchor (§4 — NOT raw offsets as source of truth), `selected_text` (**always denormalized** — survives source edits, is your retrieval display), `coder` + **`is_llm` flag** (essential — enables LLM-vs-human comparison), `memo`, `created_at`, optional confidence/weight.
  - Uniqueness `(code_id, document_id, start, end, coder)` — naturally permits **overlapping codes** (different code or span = new row). Overlaps fall out for free; no special handling.
- **Memo/Note** — standalone entity with polymorphic attachment `target_type ∈ {code, document, segment, project, free}` + `target_id`. Don't hardwire memo as a column on code. Distinguish code-memo (= live codebook definition), document-memo, segment-memo, journal-memo.
- **Speaker as Case** — model `Speaker` first-class; attach each turn's offset range. Enables "coded segments per speaker" retrieval. Treat as Case/Variable on export.
- **Variable/Attribute** — `(name, type, value, target_id, target_type)` for crosstabs (code frequency by speaker role / participant attribute). P1+.
- **Coder/User** — `id`, `name`, `is_llm`, `model_id` (provenance). **Stamp every segment + memo with a coder from day one** — retrofitting attribution to enable IRR is painful.

### MVP feature set

**P0 (it's not a coding tool without these):** code-system CRUD with hierarchy/color/definition · apply code to selected span · overlapping codes (render multiple stripes) · uncoding (remove one code from a multi-coded span) · **code-and-retrieve** ("show all segments coded X" with source + context + speaker — the defining CAQDAS operation) · coder attribution with `is_llm`.

**P1:** code/segment/document memos (surface code memo as live codebook def) · in-vivo coding (select text → new code named after the selection, code it, one action) · code frequency counts · retrieval filtered by code + speaker + attribute · **REFI-QDA `.qdpx` export** (researchers expect to move data to their lab's licensed software).

**P2:** code co-occurrence matrix (pure query over offset overlaps) · code × speaker crosstab · **intercoder reliability (LLM vs human)** — the differentiator · code weighting/confidence (home for LLM confidence) · `.qdc` codebook-only export.

**Workflow support:** design for **abductive/hybrid** (seed codebook + new codes mid-stream). Keep a "suggested codes" inbox separate from the accepted code system; make merge (re-point all segments A→B) and rename cheap.

**Intercoder reliability (the LLM-vs-human layer):** treat LLM as coder A, human as coder B. For code-presence-per-unit use Cohen's κ (2 coders) / Krippendorff's α (≥2). For span agreement, expose an overlap threshold (≥50% or any-overlap) — it materially changes results. **Caveat to encode:** κ is deflated for rare codes ("kappa paradox"); report raw % agreement + α (or Gwet's AC1) alongside. The "compare coders" view (agreements / human-only / LLM-only + click-through to disagreements) doubles as the human-in-the-loop adjudication UI.

### Codebook file format — recommendation

**Two layers:** human-authored **YAML** working format (diff-friendly, git-versionable, doubles as the LLM prompt payload) + **REFI-QDA `.qdc`** for interop export. Use the established **six-component structure** (id, name, definition, inclusion, exclusion, examples) — inclusion/exclusion criteria are exactly what make deductive LLM coding reliable.

```yaml
codebook:
  name: "Parking Garage UX Study"
  version: 3
  created: 2026-06-18
  model: "claude-opus-4-8 / llama3.1:70b"   # audit trail
  codes:
    - id: access-friction          # stable slug/UUID — renames don't orphan segments
      name: "Access Friction"
      color: "#E63946"
      definition: "Participant describes difficulty entering, exiting, or paying."
      inclusion: >                  # when TO apply — drives recall
        Apply to any mention of barriers, ticket/payment trouble,
        gate malfunction, or confusion about where to go.
      exclusion: >                  # when NOT to apply — drives precision
        Do NOT apply to general complaints about price (use Cost),
        or to navigation once already parked (use Wayfinding).
      examples:
        - "I sat at the gate for five minutes because the ticket wouldn't scan."
      children:
        - id: access-payment
          name: "Payment Failure"
          color: "#F1A0A6"
          definition: "..."
          inclusion: "..."
          exclusion: "..."
          examples: ["..."]
```

**Why YAML over the alternatives:** the two coding-research agents split between YAML and YAML+Markdown front-matter; pick **plain YAML** — it's the simplest machine-parseable form that still diffs cleanly in git, and front-matter buys nothing here. **Key gotcha on `.qdc` export:** REFI's `<Code>` has only `name/guid/isCodable/color` + a single `<Description>` — no fields for inclusion/exclusion/examples. Keep the rich six-part structure natively and **flatten** (concatenate definition + inclusion + exclusion + examples into `<Description>`) on export. Keep a stable machine `id` separate from human `name` — when the model misfires, refine the *definition wording*, not the label.

---

## 3. LLM Coding Design

**Mode:** deductive (apply researcher's codebook). This is the regime where LLM-coding reliability is proven (GPT-4 + chain-of-thought reaches mean Cohen's κ ≈ 0.68, human-level; CoT lifted κ from 0.59→0.68). Reserve inductive (open code generation) for a separate explicit mode — LLMs are weakest and most biased there.

### Prompt structure (4 blocks)
1. **Role + task** — "You are a qualitative coder applying a fixed codebook. Apply only codes defined below; do not invent codes."
2. **Codebook** — full definitions with inclusion/exclusion + examples (the YAML payload from §2).
3. **Chain-of-thought** — require a short rationale *before* the code decision. Single most reproducible quality lever.
4. **Output contract** — strict JSON with verbatim-quote grounding.

### Output JSON
```json
{
  "segment_id": "t_0037",
  "codes": [
    {
      "code_id": "access-friction",
      "quote": "verbatim span copied exactly from the segment",
      "char_start": 142,
      "char_end": 211,
      "rationale": "one-sentence justification tied to the code definition",
      "confidence": 0.82
    }
  ],
  "no_code": false,
  "suggested_new_code": null
}
```

**Design rules:** require a verbatim `quote` for every code and **programmatically reject any code whose `quote` is not a substring of the segment** — primary anti-hallucination mechanism (hallucination ran 1.2–12.4% even in careful studies). `rationale` mandatory. `confidence` for triage/sorting only — treat as soft (LLMs overconfident, never auto-accept). Provide an explicit `no_code`/`NO_CODE` escape so the model isn't forced to over-apply. `suggested_new_code` lets the model *propose* (never auto-apply) emergent themes for human triage. **Temperature 0–0.2, pin the model version, record both.**

### Chunking
- **Segment-by-segment is the default** (code at the natural unit — utterance/turn/speaker block). Keeps spans precise, review tractable; avoids lost-in-the-middle recall failures.
- **Sliding context window** of ±1–3 neighboring segments as **read-only context** (resolve pronouns/topic continuity) while only the focal segment is coded.
- **Avoid whole-document coding** even when it fits the context window — degrades mid-context recall, hard to parse/review. Use long windows for optional document-level summaries only.
- **Per-code vs whole-codebook:** per-code prompting scored higher but costs more tokens. Pragmatic compromise: **group ≤5–8 closely related codes per call; isolate hard-to-distinguish codes** into their own pass.

### Human-in-the-loop review (anti-anchoring is critical)
The naive implementation (show AI label → click accept) **manufactures false agreement** — pre-highlighting labels was the *worst* offender (full-consensus jumped 8%→38%; using such labels to score the model inflated F1 by +0.32–0.56). Design against it:
- **Show the `rationale` and `quote` prominently, not just the code chip** — force the reviewer to evaluate evidence, not the label.
- **Offer alternatives where confidence is low** (top candidate + runner-up + `NO_CODE`) — the one mitigation the anchoring authors endorse.
- **Optional blind-first** on a subset (reviewer codes before seeing the suggestion) to calibrate true agreement.
- **Triage by confidence + ambiguity**, not blanket scrolling: route low-confidence / multi-code / `suggested_new_code` items to a priority queue. High-confidence still gets light review — never auto-commit.
- **Reviewer actions:** accept / edit span / change code / reject / add code / promote suggestion to codebook. **Log every edit with who/when — this IS your reliability dataset.** Keep rejected applications (audit trail / training data), filtered from main view.

### Pluggable backend
One internal interface `async def stream_code(messages) -> AsyncIterator[str]`, two implementations: **Claude** (`anthropic` SDK `client.messages.stream(...)`, key in server `.env`) and **Ollama** (`POST localhost:11434/api/chat`, `"stream": true`, async-iterate NDJSON). Per-project switch via `projects.llm_backend` + `llm_model`. The browser never sees the key.

### Pitfalls
Hallucinated codes/quotes (substring-validate) · over-coding (LLMs over-predict — explicit `NO_CODE`, watch per-code prevalence) · **non-random bias** (errors correlate with speaker characteristics → biases downstream stats; validate against human ground truth before trusting any quantitative result) · anchoring/automation bias (the biggest threat to the "human checks everything" guarantee) · latent/affective/sarcastic meaning is the LLM weak spot — keep human-led · reproducibility drift (re-validate after any model upgrade) · never insert a summarization step that severs quote→code linkage.

---

## 4. Extended JSON + Relational Data Model

### Span anchoring — the load-bearing decision

**Recommendation: hybrid anchor — primary quote-based fuzzy anchor + secondary offset *hint*, resolved at load time into a live offset. Never persist raw char offsets as the source of truth.** Raw offsets silently corrupt on edit (wrong, not visibly broken); marker injection pollutes editable text and collides with the existing `[?? ...]` regex. Quote/fuzzy anchoring (W3C `TextQuoteSelector` / Hypothes.is pattern) degrades **gracefully and visibly** (orphaned, not silently wrong).

```jsonc
"anchor": {
  "segmentId": "t_0007",          // hard requirement
  "quote": "Schranke war kaputt", // PRIMARY
  "prefix": "die ", "suffix": " an dem",  // ~16 chars each, disambiguate repeats
  "hint": { "start": 12, "end": 31, "textHash": "9f3a1c" }  // FAST PATH ONLY
}
```
Resolution order: (1) if `hint.textHash` == current segment hash → trust offsets O(1) (common case); (2) else search `prefix+quote+suffix`; (3) else fuzzy-match `quote` alone → flag `shifted`; (4) else `orphaned` — keep the record, surface in a sidebar ("3 highlights lost their anchor — click to re-attach"), never silently drop. **On edit-commit, re-resolve all anchors in that segment and rewrite their `hint`** so drift never accumulates. **One shared `anchor` shape** for highlights, comments, AND code applications — implement resolution once. Whole-turn spans set `"whole": true` and omit quote/prefix/suffix.

### Versioned JSON (v2, backward-compatible with v1 flat `segments[]`)

Envelope gains `schemaVersion: 2`, `docId`, `rev` (monotonic, for optimistic concurrency), `createdAt/updatedAt`, `media{}`. Key changes:
- **`speaker` string → `speakers[]` entities** (`id`, `label`, `color`, `role`, `aliases[]` mapping raw diarization labels) + `segment.speakerId`. Keep legacy `segment.speaker` string + top-level `speakerColors{}` as **downgrade mirrors** (written, not read by new code) so old tool versions open v2 degraded-but-usable.
- **New top-level arrays (all optional, all absent in v1 → old files load clean):** `highlights[]`, `comments[]`, `codeSystem[]`, `codeApplications[]`.
- **New per-turn optional fields:** `words[]` (word-level timestamps in absolute seconds — keyed to audio not characters, so they survive text edits; absence just disables karaoke), `provenance{}` (asr/diarization model, confidences, lastEditedBy/At), `textHash`.
- **`codeApplications`** carry `source` ('llm'|'human'), `confidence`, `rationale`, `status` ('suggested'|'accepted'|'rejected'), `reviewer`, `createdBy` (userId or `model:<id>`). This makes the LLM-suggest → human-review flow first-class. `createdBy` is `"local"` offline, becomes a real FK on the server — no schema change.

**Migration v1→v2** (`migrateV1toV2`): set envelope; collect distinct `speaker` strings → `speakers[]` (color from `speakerColors` or auto), build name→id map; rewrite segments with `speakerId` (keep `speaker` mirror, compute `textHash`); init new arrays empty; persist once. **Lossless and idempotent.** Migrate-on-load, keep the original blob until the migrated version saves successfully, gate every feature on its array's presence.

### Relational mapping (arrays → tables, ~1:1)

`document` (envelope, `rev` = optimistic-lock token) · `speaker` · `segment` (single table, `type` discriminator turn|divider, explicit **`ord` column** — use sparse/fractional/LexoRank keys so inserts don't renumber) · `word` (child of segment) · `highlight` / `comment` / `code_application` — **each embeds the same denormalized anchor columns** (`segment_id`, `anchor_whole`, `quote`, `prefix`, `suffix`, `hint_start`, `hint_end`, `hint_text_hash`) rather than pointing at offsets in a normalized text table · `code` (self-referential hierarchy, `parent_id`, `ord`). **`GET /documents/:id` re-assembles exactly the v2 JSON the offline tool already understands** — same front-end code path loads local files and server docs; localStorage stays a valid offline cache. Move `code` to a `project_id` parent when you want one codebook across many interviews (standard in qualitative research).

---

## 5. Self-Host Architecture

**Guiding principle:** the current tool is a synchronous, single-GPU, file-producing function. The smallest viable team server is **one FastAPI process + one GPU worker process + one database + a DB-backed job queue.** No Kubernetes/Redis/Celery/microservices for a 6-person team — that's permanent operational debt.

### Stack
- **Backend: FastAPI + uvicorn**, serving frontend + API + LLM from one process. The pipeline, Whisper, torch, pyannote are all Python — FastAPI imports `yt_transcribe.transcribe` directly, zero model-code rewrite, same conda/CUDA env. Native async + SSE makes progress + LLM streaming trivial; serves the single-file frontend as a static asset; Pydantic gives typed segment/code APIs. (Rejected: Flask — awkward streaming; Django — heavy; Node — splits the stack, can't call GPU in-process; Streamlit/Gradio — wrong fit for a custom editor.)
- **DB: start SQLite (WAL), schema Postgres-compatible via SQLAlchemy, switch to PostgreSQL the moment there are concurrent writers.** SQLite serializes writes → `database is locked` under genuine concurrent coding. Postgres gives row-level MVCC + `LISTEN/NOTIFY` (useful for worker + live-edit broadcast) + proper migrations. Develop on SQLite, deploy team server on Postgres.

### GPU job handling + progress
**Separate long-lived `worker.py` process + DB-backed queue.** Upload stores audio, inserts `transcripts` (`status='queued'`) + `jobs` row, returns immediately. The worker **loads Whisper large-v3 + pyannote once at startup, keeps them resident in VRAM** (model load is the slow part — never reload per job), loops claiming the oldest queued job atomically (`UPDATE ... SET status='running' ... RETURNING`), runs the existing pipeline refactored to **write segments to DB instead of txt/srt** and update `jobs.progress/stage`. One GPU = one consumer — a dedicated worker serializes GPU jobs naturally (no VRAM contention) and survives web restarts. **Not** `BackgroundTasks`/threads (reload model per request, OOM risk); **not** Celery+Redis (overkill — skip the broker).

**Progress → browser via SSE** (`EventSource`, auto-reconnect, plain HTTP through the proxy). Progress is one-directional server→browser — WebSocket is overkill, polling (`GET /api/jobs/{id}` every 2s) is an acceptable fallback. The SSE generator reads `jobs.progress/stage` (Postgres `LISTEN/NOTIFY` or 1s poll). **The same SSE plumbing carries LLM token streaming** — implement the pattern once. Stage the bar coarsely (`transcribing → diarizing → merging`) so it always moves even when pyannote gives no fine progress.

### Multi-user concurrency (the key call)
**Hybrid, kept simple:**
- **Transcript text → last-write-wins + optimistic version check.** `PATCH /segments/{id}` sends the `text_version` it read; server rejects with `409` on mismatch, UI reloads that one segment. Every write logged to append-only `edits` (audit trail — nothing truly lost).
- **Coding/highlights/comments → per-user layers, zero conflict.** `code_applications.applied_by` / `highlights.created_by` mean two coders coding the same transcript write different rows and never collide. UI offers "show my codes / all codes." This is the cleanest answer for the analysis work coders do most.
- **Skip explicit locking and CRDT/OT** — massive overkill. (Optional later: "user X is editing segment N" presence over the same SSE channel — a hint, not a lock.)

**Auth:** server-side **session cookies** (signed `HttpOnly` + a `sessions` row, argon2 password hashing), not JWT — simpler, revocable, no refresh-token plumbing for one server. **Two roles:** admin (create projects, upload/launch jobs, manage codebook, members, LLM backend) and coder (edit text, apply/remove own codes, highlight, comment, run LLM suggestions), enforced per-project via `project_members.role`.

### LLM key handling + local-only privacy mode
Keys live server-side in `.env` (`ANTHROPIC_API_KEY`, HF token) — **browser never holds them.** Per-project backend switch. **Local-only privacy mode:** retain the single-file `index.html` as a first-class supported mode — runs from `file://`, persists to localStorage, never contacts the server, audio never leaves the analyst's machine. For LLM coding in this mode, point at **local Ollama only** (or disable the LLM panel) so sensitive transcripts never hit an external API. **One shared import/export JSON format** (segments + codes + highlights + comments) bridges offline and server — a sensitive transcript can be coded offline and *optionally* imported to the team server later, or never. This is your IRB/ethics escape hatch.

### Windows deployment
**Run natively on Windows — no Docker for the GPU worker.** GPU-in-Docker on Windows is a known pain zone (Docker Desktop → WSL2 → host NVIDIA driver → nvidia-container-toolkit, and still hits "GPU access blocked by the OS"); the pipeline already has Windows-specific workarounds (soundfile/torchcodec patch, ffmpeg WAV conversion). Keep the existing conda+CUDA env. Run two services under **NSSM** (start on boot, restart on crash): `uvicorn app:app --port 8000` and `python worker.py`. Reverse proxy + TLS via **Caddy** (automatic HTTPS, 3-line Caddyfile) — **disable response buffering on the SSE routes** (`/api/jobs/*/events`, `/api/llm/code`) or progress arrives in chunks. Back up the DB + `media/` folder. (Optional: containerize **web + DB only** CPU-side, leave the GPU worker native — sidesteps the Windows-GPU-Docker problem. Full Docker becomes clean only if the team later moves to Linux.)

---

## 6. Recommended Phased Build Order

Each phase is independently shippable and leaves a working tool. The existing `yt_transcribe` package is reused throughout — the only real backend rewrite is redirecting the pipeline's save step from files to DB rows + emitting progress.

### Phase 1 — Tool features first (CLIENT-SIDE ONLY, no backend)
Everything ships in the existing single-file HTML tool against localStorage:
1. **Migrate the data model to v2 JSON** (§4) — speaker entities, `schemaVersion`, empty `highlights/comments/codeSystem/codeApplications` arrays, the shared `anchor` resolver, migrate-on-load. *Do this first — everything else depends on it.*
2. **Editor P0 core loop:** inline click-to-edit, `Tab` play/pause-in-place, auto-rewind-on-pause, 0.5–1.5x speed, Find & Replace, auto-save indicator, undo/redo.
3. **Speaker management:** merge speakers, per-segment reassign (rename already exists).
4. **Karaoke highlight + confidence gradient + "next low-confidence word"** — *gated on word-level timestamps + confidence being present in the data; if the current pipeline doesn't emit `words[]`/confidence, this slips to when the pipeline is extended (Phase 3+).*
5. **Annotation:** multi-color highlights, anchored comments, floating selection toolbar (all on the shared `anchor`).
6. **Client-side qualitative coding MVP:** code-system CRUD (hierarchy/color/definition), apply-code-to-span, overlapping codes, uncoding, code-and-retrieve, in-vivo coding, coder attribution (`is_llm` flag, `createdBy:"local"`).
7. **Enriched export/import:** the shared v2 JSON round-trip + metadata-bearing txt/srt. (`.qdpx`/`.qdc` and DOCX/PDF are P1 — defer.)

### Phase 2 — Parse-only API (lowest-risk backend intro)
Wrap existing `transcribe()` behind FastAPI: `POST /transcribe` (short clips run synchronously → return v2 JSON), serve `index.html` statically, add a parser turning `[HH:MM:SS] SPEAKER: text` into the JSON segment model. Proves the stack with almost no new code. **No DB, no auth.**

### Phase 3 — Single-user server + SQLite
SQLAlchemy models (§4 relational) on SQLite/WAL. Frontend persists to the server API instead of localStorage (segments, codes, highlights, comments). Keep single-file/localStorage mode alongside via the shared JSON. *If the pipeline gains word-level timestamps + confidence here, light up the karaoke/confidence features deferred from Phase 1.* Still synchronous transcription, still no auth.

### Phase 4 — Async GPU worker + SSE progress
Split out `worker.py` (model resident in VRAM) + `jobs` table. Upload returns immediately; browser subscribes to `/api/jobs/{id}/events` via `EventSource`. Refactor the pipeline's save step to write DB segments + bump `jobs.progress`. Biggest single UX win.

### Phase 5 — Multi-user
Session-cookie auth, `users`/`projects`/`project_members`, admin vs coder roles, per-user coding layers, optimistic version-check on edits (`text_version` + 409). **Migrate SQLite → PostgreSQL** now that there are concurrent writers.

### Phase 6 — LLM coding
The `stream_code` interface with Claude (`.env` key) + Ollama implementations, per-project `llm_backend` switch, `/api/llm/code` SSE reusing Phase 4 plumbing. Persist accepted suggestions as `code_applications` with the anti-anchoring review UI (§3). Add the LLM-vs-human compare view + IRR (κ/α/AC1).

### Phase 7 — Deploy hardening
NSSM services (web + worker), Caddy reverse proxy + TLS with SSE buffering disabled, DB + `media/` backups, document the native-Windows-GPU rationale. Optionally containerize web+DB only. Add `.qdpx`/`.qdc` REFI-QDA export for lab interop.

---

### Cross-cutting decisions where agents diverged (resolved)
- **Span storage:** offsets-as-truth (relational agent's table sketch implies raw offsets) **vs** quote-fuzzy anchor (data-model agent). → **Quote-fuzzy hybrid wins** — offsets are a fast-path hint only. The relational tables must store the denormalized anchor columns, not FK into a normalized text offset.
- **Codebook format:** YAML (CAQDAS agent) vs YAML+Markdown front-matter (LLM agent). → **Plain YAML** — front-matter adds nothing.
- **DB:** SQLite vs Postgres. → **Both, sequenced** — SQLite through Phase 4, Postgres at Phase 5 when concurrent writers appear; SQLAlchemy makes the switch cheap.
- **Concurrency:** real-time collab (HappyScribe-inspired) vs optimistic + per-user layers (arch agent). → **Optimistic + per-user layers** — real-time OT/CRDT is unjustified complexity for 6 users.
- **Progress transport:** SSE vs WebSocket vs polling. → **SSE** — one-directional, reuses for LLM streaming, survives the proxy.
