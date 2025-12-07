#!/bin/bash
export OPENAI_API_KEY=$(cat key.txt)
export PYTHONPATH=.
uv run python scripts/debug_llm_api.py 2>&1 | tail -60
