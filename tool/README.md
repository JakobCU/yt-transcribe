# Transkript-Checker (web tool)

Human correction + (later) LLM-assisted qualitative coding of diarized transcripts.
Reads the `yt_transcribe` pipeline output format: `[HH:MM:SS] SPEAKER: text` lines.

## Two ways to run the same frontend

The frontend is **split for development** and so the backend can serve it as static
assets later, but it also builds into **one self-contained `.html`** for the
privacy-mode "double-click, nothing uploaded" use.

```
tool/
  src/                 # source of truth (edit these)
    index.html         #   structure; references styles.css + app.js, empty embed slot
    styles.css
    app.js
  build.py             # inlines src/ -> one offline single-file
  sample/              # local test media + transcripts (gitignored)
  dist/                # build output (gitignored)
```

### Offline single-file (privacy mode)

```bash
python tool/build.py                                  # dist/transcript-checker.html (empty)
python tool/build.py --embed sample/ws2-transcript.txt --out dist/ws2-checker.html
```

Open the resulting file by double-click. Audio/transcript stay on the machine;
state persists to `localStorage`. This is the IRB/ethics-safe mode for sensitive data.

### Served (dev / future backend)

Serve the `tool/` dir over HTTP and open `/src/index.html`:

```bash
python -m http.server 8777 --directory tool
```

The same `src/` files are what the FastAPI backend will serve in later phases.

## Data format

`[HH:MM:SS] SPEAKER: text` per turn; `===== TEIL ... =====` lines become dividers;
inline `[?? Audio prüfen]` style brackets become highlighted review markers.
See the embedded header in `sample/ws2-transcript.txt` for the full convention.

## Roadmap

Phase 1 (current): client-side editor features + qualitative coding MVP, all on
`localStorage`. Later phases add a FastAPI backend (pipeline-as-a-service, multi-user
projects, pluggable Claude/Ollama LLM coding). See the project memory / engineering
brief for the full phased plan.
