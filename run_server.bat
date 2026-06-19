@echo off
REM Startet den Transkript-Studio-Server (liefert Frontend + Pipeline + LLM).
REM Frontend braucht NICHT separat gestartet zu werden -- alles unter einer URL.
cd /d "%~dp0"

set PY=C:\Users\TX.Lab\miniconda3\envs\yt-transcribe\python.exe

REM Reduce CUDA memory fragmentation on small GPUs (6 GB) so the Whisper<->pyannote
REM hand-off has contiguous room. (The pipeline also offloads one model at a time.)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Optional zum Testen ohne GPU / ohne API-Key (Zeilen einkommentieren):
REM set TRANSCRIBE_FAKE=1
REM set LLM_FAKE=1

echo.
echo   transkript::studio  ->  http://127.0.0.1:8000/
echo   (Strg+C zum Beenden)
echo.
"%PY%" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
