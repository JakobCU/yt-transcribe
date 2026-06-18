"""Build the offline single-file Transkript-Checker.

The frontend lives split for development and so the backend can serve it as static
assets (src/index.html + styles.css + app.js + vendor/*). This script inlines all
local stylesheets and scripts into ONE self-contained .html that works by
double-click from file:// with no server and nothing uploaded -- the privacy-mode
artifact.

Usage:
    python build.py                         # -> dist/transcript-checker.html (no transcript embedded)
    python build.py --embed sample/ws2-transcript.txt
    python build.py --embed sample/ws2-transcript.txt --out dist/ws2-checker.html
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TOOL = Path(__file__).resolve().parent
SRC = TOOL / "src"

EMBED_EMPTY = '<script id="embedded-transcript" type="text/plain"></script>'
LINK_RE = re.compile(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>')
SCRIPT_RE = re.compile(r'<script[^>]*src="([^"]+)"[^>]*>\s*</script>')


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8").rstrip("\n")


def _is_local(href: str) -> bool:
    return not re.match(r"^(https?:)?//", href)


def build(embed: str | None, out: str) -> Path:
    html = (SRC / "index.html").read_text(encoding="utf-8")

    def inline_link(m: re.Match) -> str:
        href = m.group(1)
        return f"<style>\n{_read(href)}\n</style>" if _is_local(href) else m.group(0)

    def inline_script(m: re.Match) -> str:
        src = m.group(1)
        return f"<script>\n{_read(src)}\n</script>" if _is_local(src) else m.group(0)

    html = LINK_RE.sub(inline_link, html)
    html = SCRIPT_RE.sub(inline_script, html)

    if embed:
        text = Path(embed).read_text(encoding="utf-8").strip()
        text = text.replace("</script", "<\\/script")  # guard against early close
        html = html.replace(EMBED_EMPTY, f'<script id="embedded-transcript" type="text/plain">{text}</script>')

    out_path = TOOL / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the offline single-file Transkript-Checker")
    ap.add_argument("--embed", default=None, help="Path to a transcript .txt to embed")
    ap.add_argument("--out", default="dist/transcript-checker.html", help="Output path (relative to tool/)")
    args = ap.parse_args()

    out = build(args.embed, args.out)
    print(f"Built {out.relative_to(TOOL)}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
