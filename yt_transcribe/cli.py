"""CLI entry point for yt-transcribe."""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def is_url(s: str) -> bool:
    """Check if a string looks like a YouTube URL."""
    return bool(re.match(r"https?://", s)) or s.startswith("youtu.be/")


def main():
    parser = argparse.ArgumentParser(
        prog="yt-transcribe",
        description="Transcribe YouTube videos or local audio files with speaker diarization",
    )
    parser.add_argument(
        "source",
        help="YouTube URL or path to local audio file",
    )
    parser.add_argument(
        "--model", default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Language code (e.g. 'en'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization (transcription only)",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token for pyannote (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to use: 'cuda' or 'cpu' (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output base filename (without extension). Defaults to input filename.",
    )
    args = parser.parse_args()

    # Download if URL, otherwise use local file
    if is_url(args.source):
        from yt_transcribe.download import download_audio
        audio_path = download_audio(args.source)
    else:
        audio_path = args.source
        if not os.path.isfile(audio_path):
            print(f"Error: file not found: {audio_path}")
            sys.exit(1)

    # Resolve HF token for diarization
    hf_token = None
    if not args.no_diarize:
        from yt_transcribe.setup import resolve_hf_token
        hf_token = resolve_hf_token(args.hf_token)
        if not hf_token:
            print("No token available. Continuing without diarization.\n")

    # Run transcription
    from yt_transcribe.transcribe import transcribe
    transcribe(
        audio_path=audio_path,
        model=args.model,
        language=args.language,
        no_diarize=args.no_diarize or not hf_token,
        hf_token=hf_token,
        device=args.device,
        output=args.output,
    )
