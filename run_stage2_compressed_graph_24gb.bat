@echo off
setlocal
python scripts\stage2_full_build.py --config config.nightly.24gb.yaml --profile nightly_compressed_answerable_24gb --force
