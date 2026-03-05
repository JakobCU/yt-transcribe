"""First-run setup flow for Hugging Face token and model access."""

import os
import webbrowser
from pathlib import Path


GATED_MODELS = [
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
]


def check_model_access(token: str) -> list[str]:
    """Check which gated models the token can't access. Returns list of inaccessible URLs."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []  # can't check, let it fail later with a clear error

    api = HfApi(token=token)
    inaccessible = []
    for url in GATED_MODELS:
        repo_id = url.replace("https://huggingface.co/", "")
        try:
            api.model_info(repo_id, token=token)
        except Exception:
            inaccessible.append(url)
    return inaccessible


def resolve_hf_token(hf_token: str | None, env_path: Path | None = None) -> str | None:
    """Resolve HF token from arg, env var, or .env file. Prompt if missing."""
    # 1. Already provided via CLI
    if hf_token:
        return hf_token

    # 2. Environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # 3. Prompt user
    print("\nNo Hugging Face token found.")
    print("A token is required for speaker diarization (pyannote gated models).")
    print("Get one at: https://huggingface.co/settings/tokens\n")

    token = input("Paste your token here (or press Enter to skip diarization): ").strip()
    if not token:
        return None

    # 4. Check model access
    print("\nChecking model access...")
    inaccessible = check_model_access(token)
    if inaccessible:
        print("\nYou need to accept the model terms at:")
        for url in inaccessible:
            print(f"  -> {url}")

        answer = input("\nOpen these URLs in your browser? [Y/n] ").strip().lower()
        if answer != "n":
            for url in inaccessible:
                webbrowser.open(url)

        input("\nPress Enter once you've accepted all models...")

        # Re-check
        still_inaccessible = check_model_access(token)
        if still_inaccessible:
            print("\nStill can't access these models:")
            for url in still_inaccessible:
                print(f"  -> {url}")
            print("Continuing without diarization.\n")
            return None

    # 5. Save to .env
    if env_path is None:
        env_path = Path.cwd() / ".env"

    print(f"\nSaving token to {env_path}")
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        lines = [l for l in lines if not l.startswith("HF_TOKEN=")]
    lines.append(f"HF_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n")

    return token
