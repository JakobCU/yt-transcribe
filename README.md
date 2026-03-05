# yt-transcribe

Transcribe YouTube videos (or local audio files) with speaker diarization using Whisper + pyannote.audio.

## Install

```bash
# Create venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install with CUDA support (recommended)
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

**Requirements:** FFmpeg must be installed and on your PATH (used for audio conversion).

## Usage

```bash
# Transcribe a YouTube video
yt-transcribe https://www.youtube.com/watch?v=VIDEO_ID

# Transcribe a local file
yt-transcribe recording.m4a

# Skip diarization (faster, no HF token needed)
yt-transcribe recording.m4a --no-diarize

# Use a specific model/language
yt-transcribe recording.m4a --model medium --language en

# Force CPU
yt-transcribe recording.m4a --device cpu
```

## First run with diarization

Speaker diarization uses gated pyannote models that require a Hugging Face token. On first run, the tool will:

1. Prompt you for your token (get one at https://huggingface.co/settings/tokens)
2. Check access to the required models
3. Open the acceptance pages if needed
4. Save the token to `.env` for future runs

You also need to accept the terms for these models:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `large-v3` | Whisper model: tiny, base, small, medium, large, large-v2, large-v3 |
| `--language` | auto-detect | Language code (e.g. `en`, `de`, `ja`) |
| `--no-diarize` | off | Skip speaker diarization |
| `--hf-token` | from .env | Hugging Face token |
| `--device` | auto | `cuda` or `cpu` |
| `--output` | input filename | Output base path (without extension) |

## Output

Produces two files:
- `filename.txt` — timestamped text with speaker labels
- `filename.srt` — subtitle file with speaker labels
