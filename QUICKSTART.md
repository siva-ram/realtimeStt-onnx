# RealtimeSTT-ONNX - Quick Start

## 🚀 Quick Setup

### Windows
```bash
# Run the setup script
setup.bat
```

### Linux/macOS
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh
```

### Manual Setup
```bash
# Install UV
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
uv pip install -e .
```

## 📝 Quick Test

```bash
# Activate venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Run basic example
python examples/basic_transcription.py

# Or use uv run (no activation needed)
uv run python examples/basic_transcription.py
```

## 💡 VS Code Debugging

1. Open project in VS Code
2. Press **F5**
3. Select **"Basic Transcription"** from dropdown
4. Speak into your microphone!

## 📦 Installation Options

```bash
# CPU only (default)
uv pip install -e .

# With GPU support
uv pip install -e ".[gpu]"

# With development tools
uv pip install -e ".[dev]"

# Everything
uv pip install -e ".[gpu,dev]"
```

## 🔧 Common Commands

```bash
# Run tests
uv run pytest

# Run with callbacks
uv run python examples/with_callbacks.py

# Run with custom VAD
uv run python examples/custom_vad.py

# Compare models
uv run python examples/test_models.py
```

For detailed documentation, see [DEVELOPMENT.md](DEVELOPMENT.md) or [README.md](README.md).
