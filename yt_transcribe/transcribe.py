"""Transcribe audio with speaker diarization using Whisper + pyannote.audio."""

import subprocess
import time
from pathlib import Path

import torch
import torchaudio

# Patch torchaudio for compatibility with speechbrain (removed in torchaudio 2.10)
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]

import whisper
from tqdm import tqdm


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_short(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format for text output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert_to_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV for maximum compatibility."""
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        return audio_path
    wav_path = str(path.with_suffix(".wav"))
    print(f"  Converting {path.name} to WAV...")
    subprocess.run(
        ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path, "-y"],
        capture_output=True,
        check=True,
    )
    return wav_path


def get_speaker_for_segment(seg_start, seg_end, diarization):
    """Find the dominant speaker for a given time segment."""
    speaker_durations = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(seg_start, turn.start)
        overlap_end = min(seg_end, turn.end)
        if overlap_start < overlap_end:
            duration = overlap_end - overlap_start
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
    if not speaker_durations:
        return "UNKNOWN"
    return max(speaker_durations, key=speaker_durations.get)


def run_diarization(audio_path: str, hf_token: str, device: str):
    """Run pyannote speaker diarization."""
    import soundfile
    from pyannote.audio import Pipeline

    print("\n[2/3] Running speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device(device))

    # Load audio via soundfile to bypass broken torchcodec on Windows
    data, sample_rate = soundfile.read(audio_path)
    waveform = torch.from_numpy(data).float().unsqueeze(0)  # (1, time)
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.x returns DiarizeOutput; extract the Annotation
    diarization = getattr(output, "speaker_diarization", output)

    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    print(f"  Detected {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")

    return diarization


def run_transcription(audio_path: str, model_name: str, device: str, language: str | None):
    """Run Whisper transcription."""
    print(f"\n[1/3] Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name, device=device)

    print(f"  Transcribing (this may take a while for long audio)...")
    transcribe_opts = {"verbose": False, "fp16": device == "cuda"}
    if language:
        transcribe_opts["language"] = language

    result = model.transcribe(audio_path, **transcribe_opts)

    print(f"  Transcription complete: {len(result['segments'])} segments")
    if result.get("language"):
        print(f"  Detected language: {result['language']}")

    return result


def merge_and_save(result, diarization, output_base: str):
    """Merge whisper segments with speaker labels and save output files."""
    print("\n[3/3] Merging transcription with speaker labels...")

    segments = result["segments"]
    txt_path = output_base + ".txt"
    srt_path = output_base + ".srt"

    txt_lines = []
    srt_lines = []

    for i, seg in enumerate(tqdm(segments, desc="  Merging")):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        if diarization is not None:
            speaker = get_speaker_for_segment(start, end, diarization)
        else:
            speaker = "SPEAKER"

        txt_lines.append(f"[{format_timestamp_short(start)}] {speaker}: {text}")

        srt_lines.append(str(i + 1))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(f"[{speaker}] {text}")
        srt_lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"  Saved: {txt_path}")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"  Saved: {srt_path}")


def transcribe(
    audio_path: str,
    model: str = "large-v3",
    language: str | None = None,
    no_diarize: bool = False,
    hf_token: str | None = None,
    device: str | None = None,
    output: str | None = None,
):
    """Main transcription pipeline."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    output_base = output or str(Path(audio_path).with_suffix(""))

    # Convert to WAV for compatibility (pyannote torchcodec broken on Windows)
    wav_path = convert_to_wav(audio_path)

    start_time = time.time()

    # Step 1: Transcription
    result = run_transcription(wav_path, model, device, language)

    # Save transcription immediately (so it's not lost if diarization fails)
    merge_and_save(result, None, output_base)

    # Step 2: Diarization
    if not no_diarize and hf_token:
        diarization = run_diarization(wav_path, hf_token, device)
        # Re-save with speaker labels
        merge_and_save(result, diarization, output_base)

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\nDone! Total time: {h:02d}:{m:02d}:{s:02d}")
