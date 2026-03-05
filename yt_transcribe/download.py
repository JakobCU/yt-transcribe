"""Download audio from YouTube using yt-dlp."""

from pathlib import Path

import yt_dlp


def download_audio(url: str, output_dir: str = ".") -> str:
    """Download audio from a YouTube URL and return the path to the file."""
    output_template = str(Path(output_dir) / "%(title)s.%(ext)s")

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        print(f"Downloading audio from: {url}")
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        print(f"  Saved: {filename}")
        return filename
