# RealtimeSTT-ONNX Development Guide

## Installation with UV

### Prerequisites

Install UV if you haven't already:
```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup Development Environment

1. **Clone and navigate to the project:**
   ```bash
   cd d:/code/realtimeStt-onnx
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   # Create venv and install package in editable mode
   uv venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   
   # Install package with dependencies
   uv pip install -e .
   ```

3. **For GPU support:**
   ```bash
   uv pip install -e ".[gpu]"
   ```

4. **For development (includes pytest, black, ruff):**
   ```bash
   uv pip install -e ".[dev]"
   ```

### Quick Install Commands

```bash
# CPU only (default)
uv pip install -e .

# GPU support
uv pip install -e ".[gpu]"

# Development tools
uv pip install -e ".[dev]"

# Everything
uv pip install -e ".[gpu,dev]"
```

### Running Examples

After installation:
```bash
# Basic transcription
python examples/basic_transcription.py

# With callbacks
python examples/with_callbacks.py

# Custom VAD
python examples/custom_vad.py
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=realtimestt_onnx

# Run specific test file
uv run pytest tests/test_audio_buffer.py -v
```

### Using UV Run

UV can run scripts without activating the venv:

```bash
# Run example directly
uv run python examples/basic_transcription.py

# Run tests directly
uv run pytest

# Run with specific Python version
uv run --python 3.11 python examples/basic_transcription.py
```

### Syncing Dependencies

Update dependencies from pyproject.toml:
```bash
uv pip sync
```

### Adding New Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Add GPU dependency
uv add --optional gpu package-name
```

## Package Management Commands

```bash
# Install from pyproject.toml
uv pip install -e .

# Update all dependencies
uv pip install --upgrade -e .

# List installed packages
uv pip list

# Show package info
uv pip show realtimestt-onnx

# Uninstall package
uv pip uninstall realtimestt-onnx
```

## Building the Package

```bash
# Build wheel and sdist
uv build

# Install from built wheel
uv pip install dist/realtimestt_onnx-0.1.0-py3-none-any.whl
```

## VS Code Integration

The project includes `.vscode/launch.json` for debugging. Simply:
1. Open project in VS Code
2. Ensure UV virtual environment is activated
3. Press F5 and select a debug configuration
4. Start speaking when prompted

## Troubleshooting

### PyAudio Installation Issues

On Windows with UV:
```bash
# UV will handle PyAudio installation automatically
# If issues occur, try:
uv pip install --find-links https://www.lfd.uci.edu/~gohlke/pythonlibs/ PyAudio
```

### Virtual Environment Not Activated

Make sure to activate the venv:
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS  
source .venv/bin/activate
```

Or use `uv run` to run commands without activation.

### Dependency Conflicts

UV handles dependency resolution automatically. If conflicts occur:
```bash
# Clear cache and reinstall
uv cache clean
uv pip install -e . --force-reinstall
```
