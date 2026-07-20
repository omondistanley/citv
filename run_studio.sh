#!/bin/bash
# CITV Studio — quick launcher
# Usage: bash run_studio.sh
set -e
cd "$(dirname "$0")"

# Activate venv if present
if [ -f venv/bin/python ]; then
    PYTHON=venv/bin/python
else
    PYTHON=python3
fi

# Install gradio if missing
$PYTHON -c "import gradio" 2>/dev/null || {
    echo "Installing gradio..."
    $PYTHON -m pip install gradio --quiet
}

echo "Starting CITV Studio at http://localhost:7860"
$PYTHON app.py
