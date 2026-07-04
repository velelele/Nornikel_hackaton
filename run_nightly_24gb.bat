@echo off
setlocal
REM Usage: run_nightly_24gb.bat D:\path\to\documents
REM Runs Stage 1 + Stage 2 compressed_graph. Requires the local OpenAI-compatible LLM server from config.nightly.24gb.yaml.
if "%~1"=="" (
  echo Usage: %~nx0 ^<documents_root^>
  exit /b 2
)
python scripts\nightly_two_stage.py "%~1" --config config.nightly.24gb.yaml
