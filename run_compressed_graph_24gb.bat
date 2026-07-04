@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_compressed_graph_24gb.bat D:\path\to\documents
  exit /b 2
)
python scripts\nightly_two_stage.py "%~1" --config config.nightly.24gb.yaml --stage2-profile nightly_compressed_answerable_24gb --force-stage2
