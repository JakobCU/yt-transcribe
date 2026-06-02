"""Minimal Tkinter GUI for yt-transcribe.

Pick an audio file, choose model / diarization / device, hit Run.
Spawns a subprocess so the heavy ML deps load lazily and output streams live.
"""

import os
import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Resolve the conda env python so the subprocess always has the deps,
# regardless of which interpreter launched the GUI.
DEFAULT_PYTHON = Path(
    r"C:\Users\TX.Lab\miniconda3\envs\yt-transcribe\python.exe"
)
PROJECT_ROOT = Path(__file__).resolve().parent

MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
AUDIO_EXTS = [
    ("Audio/Video", "*.mp3 *.wav *.m4a *.mp4 *.aac *.flac *.ogg *.opus *.webm *.mkv"),
    ("All files", "*.*"),
]


class TranscribeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("yt-transcribe")
        self.root.geometry("720x560")

        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self.root.after(100, self._drain_log_queue)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # File picker
        file_frame = ttk.Frame(self.root)
        file_frame.pack(fill="x", **pad)
        ttk.Label(file_frame, text="Audio file:").pack(side="left")
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(file_frame, text="Browse…", command=self._pick_file).pack(side="left")

        # Options row
        opts = ttk.Frame(self.root)
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="medium")
        ttk.Combobox(
            opts, textvariable=self.model_var, values=MODELS,
            state="readonly", width=12,
        ).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(opts, text="Device:").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.device_var = tk.StringVar(value="cuda")
        for i, dev in enumerate(["cuda", "cpu", "auto"]):
            ttk.Radiobutton(
                opts, text=dev.upper(), variable=self.device_var, value=dev,
            ).grid(row=0, column=3 + i, sticky="w", padx=4)

        self.diarize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="Speaker diarization", variable=self.diarize_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(opts, text="Language (optional, e.g. de, en):").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        self.lang_var = tk.StringVar(value="")
        ttk.Entry(opts, textvariable=self.lang_var, width=8).grid(
            row=2, column=3, sticky="w", pady=(8, 0)
        )

        # Buttons
        btns = ttk.Frame(self.root)
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="Run", command=self._start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            btns, text="Stop", command=self._stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Clear log", command=self._clear_log).pack(side="left")
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(btns, textvariable=self.status_var, foreground="#555").pack(
            side="right"
        )

        # Log
        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, wrap="none", height=20, font=("Consolas", 9))
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    # ---------- actions ----------
    def _pick_file(self):
        # Default to the parent project (audio files often land there, not the worktree)
        initial = PROJECT_ROOT.parent.parent.parent if (
            PROJECT_ROOT.name != "yt-transcribe"
        ) else PROJECT_ROOT
        path = filedialog.askopenfilename(
            title="Pick audio file",
            initialdir=str(initial),
            filetypes=AUDIO_EXTS,
        )
        if path:
            self.file_var.set(path)

    def _build_command(self) -> list[str] | None:
        audio = self.file_var.get().strip()
        if not audio or not Path(audio).is_file():
            messagebox.showerror("Missing file", "Pick a valid audio file first.")
            return None

        py = str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable

        # cli.py defines main() but has no `if __name__ == "__main__"` block,
        # so `python -m yt_transcribe.cli` is a no-op. Call main() directly.
        argv = ["yt-transcribe", audio, "--model", self.model_var.get()]
        device = self.device_var.get()
        if device != "auto":
            argv += ["--device", device]
        if not self.diarize_var.get():
            argv += ["--no-diarize"]
        lang = self.lang_var.get().strip()
        if lang:
            argv += ["--language", lang]

        argv_repr = ", ".join(repr(a) for a in argv)
        code = (
            f"import sys; sys.argv=[{argv_repr}]; "
            "from yt_transcribe.cli import main; main()"
        )
        return [py, "-u", "-c", code]

    def _start(self):
        if self.proc is not None:
            return
        cmd = self._build_command()
        if cmd is None:
            return

        self._append_log(f"$ {' '.join(shlex.quote(c) for c in cmd[:3])} …\n\n")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running…")

        # Run from the project root so relative paths resolve cleanly.
        # PROJECT_ROOT is the worktree; cwd to the parent project where audio usually lives.
        cwd = str(Path(self.file_var.get()).parent) or str(PROJECT_ROOT)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # The package isn't pip-installed; it's imported from the repo root.
        # Since cwd is the audio file's folder, put the repo root on PYTHONPATH
        # so `from yt_transcribe.cli import main` resolves regardless of cwd.
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{PROJECT_ROOT}{os.pathsep}{existing_pp}" if existing_pp
            else str(PROJECT_ROOT)
        )

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except Exception as e:
            messagebox.showerror("Failed to launch", str(e))
            self._reset_buttons()
            return

        threading.Thread(target=self._reader, daemon=True).start()

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.status_var.set("Stopping…")

    def _reader(self):
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.log_queue.put(line)
        self.proc.wait()
        rc = self.proc.returncode
        self.log_queue.put(f"\n[exit code {rc}]\n")
        self.proc = None
        self.root.after(0, self._reset_buttons, rc)

    def _reset_buttons(self, rc: int | None = None):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if rc is None:
            self.status_var.set("Idle.")
        elif rc == 0:
            self.status_var.set("Done.")
        else:
            self.status_var.set(f"Failed (exit {rc}).")

    def _drain_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _append_log(self, text: str):
        # Whisper's tqdm uses \r to overwrite the same line — collapse those
        # so the log doesn't fill with thousands of progress lines.
        if "\r" in text and "\n" not in text:
            # in-place progress update: replace last line
            self.log.delete("end-2l", "end-1l")
            text = text.replace("\r", "") + "\n"
        self.log.insert("end", text)
        self.log.see("end")

    def _clear_log(self):
        self.log.delete("1.0", "end")


def main():
    root = tk.Tk()
    TranscribeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
