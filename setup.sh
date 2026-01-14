#!/bin/bash
# Quick setup script for RealtimeSTT-ONNX using UV

echo "========================================"
echo "RealtimeSTT-ONNX Setup Script"
echo "========================================"
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
    echo "Please restart your terminal and run this script again."
    exit 1
fi

echo "Found UV:"
uv --version
echo ""

echo "Creating virtual environment..."
uv venv
echo ""

echo "Activating virtual environment..."
source .venv/bin/activate
echo ""

echo "Installing package in editable mode..."
uv pip install -e .
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run examples:"
echo "  python examples/basic_transcription.py"
echo ""
echo "For GPU support:"
echo "  uv pip install -e '.[gpu]'"
echo ""
echo "For development tools:"
echo "  uv pip install -e '.[dev]'"
echo ""
