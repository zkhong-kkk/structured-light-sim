#!/usr/bin/env bash

# Shared, portable initialization for the recorded experiment pipelines.
# Override these when the repository or virtual environment lives elsewhere:
#   STRUCTURED_LIGHT_PROJECT=/path/to/repo
#   STRUCTURED_LIGHT_VENV=/path/to/venv
#   STRUCTURED_LIGHT_GPU=0

PIPELINE_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT="${STRUCTURED_LIGHT_PROJECT:-$(cd "$PIPELINE_DIR/.." && pwd)}"
GPU_ID="${STRUCTURED_LIGHT_GPU:-0}"

if [[ -n "${STRUCTURED_LIGHT_VENV:-}" ]]; then
    source "$STRUCTURED_LIGHT_VENV/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: no virtual environment is active." >&2
    echo "Set STRUCTURED_LIGHT_VENV or activate the environment first." >&2
fi

cd "$PROJECT"
