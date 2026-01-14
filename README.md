# RealtimeSTT-ONNX

A realtime speech-to-text library using ONNX models with advanced voice activity detection and wake word support.

## Overview

RealtimeSTT-ONNX combines the realtime audio processing capabilities of [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) with ONNX-based speech recognition from [onnx-asr](https://github.com/istupakov/onnx-asr). This provides:

- **Realtime audio capture and buffering** from microphone
- **Advanced voice activity detection** using WebRTC VAD and Silero VAD
- **ONNX-based transcription** supporting multiple model architectures
- **Callback-based architecture** for events and partial results
- **GPU acceleration** support via ONNX Runtime

## Features

- ✅ Voice Activity Detection: Two-stage VAD (WebRTC → Silero) for accurate speech detection
- ✅ Realtime Transcription: Fast ONNX model inference with GPU support
- ✅ Multiple Model Support: Whisper, Parakeet, Conformer, and more
- ✅ Callback System: Hooks for recording start/stop, transcription events
- ✅ Configurable VAD: Adjust sensitivity for different environments
- ✅ Speaker Audio Capture: WASAPI loopback support for transcribing system audio (Windows)
- ⏳ Wake Word Activation: Coming soon!

## Installation

### Using UV (Recommended)

UV is a fast Python package manager. Install it first:

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the package:

```bash
# Basic installation (CPU)
cd d:/code/realtimeStt-onnx
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
uv pip install -e .
```

### GPU Support

```bash
uv pip install -e ".[gpu]"
```

### Development Installation

```bash
uv pip install -e ".[dev]"
```

### Alternative: Using pip

```bash
pip install -r requirements.txt
```

### Installing PyAudio on Windows

UV handles PyAudio automatically. If using pip and encountering issues:

```bash
pip install pipwin
pipwin install pyaudio
```

## Quick Start

### Basic Transcription

```python
from realtimestt_onnx import AudioToTextRecorder

def on_text(text):
    print(f"Transcribed: {text}")

recorder = AudioToTextRecorder()

while True:
    recorder.text(on_text)
```

### With Callbacks

```python
from realtimestt_onnx import AudioToTextRecorder

recorder = AudioToTextRecorder(
    model="nemo-parakeet-tdt-0.6b-v3",
    on_recording_start=lambda: print("🎤 Recording..."),
    on_recording_stop=lambda: print("⏸️  Stopped"),
    on_transcription_start=lambda: print("🔄 Transcribing...")
)

recorder.text(lambda text: print(f"✅ {text}"))
```

### Realtime Transcription

Get transcription updates during speech:

```python
from realtimestt_onnx import AudioToTextRecorder

def on_realtime_update(text):
    print(f"\r🎤 {text}", end="", flush=True)

def on_final(text):
    print(f"\n✅ Final: {text}\n")

recorder = AudioToTextRecorder(
    enable_realtime_transcription=True,
    on_realtime_transcription_update=on_realtime_update,
    spinner=True  # Shows spinner during final transcription
)

recorder.text(on_final)
```

### Speaker Audio Transcription (Windows)

Transcribe audio playing from your speakers using WASAPI loopback:

```python
from realtimestt_onnx import AudioToTextRecorder
from realtimestt_onnx.audio_devices import get_default_speaker_loopback

# Get speaker loopback device
speaker_index = get_default_speaker_loopback()

# Transcribe from speakers
recorder = AudioToTextRecorder(
    input_device_type="speaker",  # Use speaker instead of microphone
    input_device_index=speaker_index
)

recorder.text(lambda text: print(f"🔊 {text}"))
```

### Multi-Source (Mic + Speaker Simultaneously)

Transcribe both microphone and speaker with a **single shared ONNX model**:

```python
from realtimestt_onnx import MultiSourceRecorder

def on_text(text, source):
    emoji = "🎤" if "Mic" in source else "🔊"
    print(f"{emoji} [{source}]: {text}")

# One model for both sources!
multi = MultiSourceRecorder(model="nemo-parakeet-tdt-0.6b-v3")

# Add microphone
multi.add_source("microphone", source_name="Mic", on_text=on_text)

# Add speaker
multi.add_source("speaker", source_name="Speaker", on_text=on_text)

# Start both simultaneously
multi.record_continuously()
```

## Supported ONNX Models

The library supports various ONNX ASR models:

### Recommended Models

- **`nemo-parakeet-tdt-0.6b-v3`** - Multilingual, good balance (default)
- **`nemo-parakeet-ctc-0.6b`** - English, fast
- **`whisper-base`** - English, OpenAI Whisper
- **`onnx-community/whisper-small`** - Optimized Whisper

### All Supported Models

- Nvidia NeMo: Parakeet, Canary, FastConformer
- OpenAI Whisper (ONNX versions)
- GigaChat GigaAM v2/v3
- Alpha Cephei Vosk models
- T-Tech T-one

See [onnx-asr documentation](https://github.com/istupakov/onnx-asr#supported-model-names) for the complete list.

## Configuration

### AudioToTextRecorder Parameters

```python
recorder = AudioToTextRecorder(
    # Model settings
    model="nemo-parakeet-tdt-0.6b-v3",  # ONNX model name or path
    quantization=None,                   # "int8", "fp16", or None
    device="cpu",                        # "cpu" or "cuda"
    
    # Audio settings
    sample_rate=16000,                   # Sample rate in Hz
    input_device_index=None,             # Microphone index (None = default)
    
    # VAD settings
    webrtc_aggressiveness=3,             # 0-3, higher = more aggressive
    silero_threshold=0.5,                # 0-1, higher = more confident required
    use_silero_vad=True,                 # Enable Silero VAD verification
    
    # Speech detection
    min_speech_duration_ms=250,          # Minimum speech duration
    post_speech_silence_duration=0.5,    # Silence to end recording (seconds)
    
    # Callbacks
    on_recording_start=None,             # Called when speech detected
    on_recording_stop=None,              # Called when silence detected
    on_transcription_start=None,         # Called before transcription
)
```

### VAD Configuration Tips

**For Noisy Environments:**
```python
recorder = AudioToTextRecorder(
    webrtc_aggressiveness=3,    # Maximum filtering
    silero_threshold=0.6,       # Higher confidence required
    post_speech_silence_duration=0.8
)
```

**For Quiet Environments:**
```python
recorder = AudioToTextRecorder(
    webrtc_aggressiveness=1,    # Less aggressive
    silero_threshold=0.3,       # More sensitive
    post_speech_silence_duration=0.4
)
```

## Examples

See the `examples/` directory for complete examples:

- [`basic_transcription.py`](examples/basic_transcription.py) - Simple transcription
- [`with_callbacks.py`](examples/with_callbacks.py) - Using all callbacks
- [`custom_vad.py`](examples/custom_vad.py) - Custom VAD configuration
- [`realtime_transcription.py`](examples/realtime_transcription.py) - Realtime updates during speech
- [`speaker_transcription.py`](examples/speaker_transcription.py) - Transcribe speaker audio (Windows)
- [`multi_source_transcription.py`](examples/multi_source_transcription.py) - Mic + Speaker with shared model

## Architecture

```
┌─────────────────────────────────────────────┐
│        AudioToTextRecorder (Main)           │
├─────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ PyAudio  │→ │   VAD    │→ │  Buffer   │ │
│  │ Stream   │  │ Handler  │  │           │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│                      ↓                       │
│              ┌──────────────┐               │
│              │ ONNX Engine  │               │
│              │ (onnx-asr)   │               │
│              └──────────────┘               │
└─────────────────────────────────────────────┘
```

### Components

- **AudioBuffer**: Thread-safe circular buffer for audio chunks
- **VADHandler**: Two-stage VAD (WebRTC + Silero) for speech detection
- **ONNXEngine**: ONNX model wrapper with preprocessing and transcription
- **AudioToTextRecorder**: Main orchestrator coordinating all components

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for much faster transcription
2. **Model Size**: Smaller models (base, tiny) are faster but less accurate
3. **Quantization**: Use `quantization="int8"` for faster inference
4. **Sample Rate**: 16kHz is optimal for most models
5. **Chunk Size**: Default (30ms) works well for most cases

## Comparison with Original RealtimeSTT

| Feature | RealtimeSTT | RealtimeSTT-ONNX |
|---------|-------------|-------------------|
| Backend | Faster Whisper | ONNX Runtime |
| Models | Whisper only | Multiple architectures |
| VAD | WebRTC + Silero | WebRTC + Silero |
| GPU Support | ✅ CUDA | ✅ CUDA, TensorRT |
| Wake Words | ✅ | ⏳ Coming soon |
| Performance | Fast | Very Fast (ONNX optimized) |

## Troubleshooting

### PyAudio Installation Issues

On Windows, use pipwin:
```bash
pip install pipwin
pipwin install pyaudio
```

On Linux:
```bash
sudo apt-get install portaudio19-dev
pip install PyAudio
```

### Model Loading Errors

Ensure you have internet connection for first-time model downloads. Models are cached in `~/.cache/huggingface/`.

### GPU Not Working

Install ONNX Runtime GPU version:
```bash
pip install onnx-asr[gpu,hub]
pip install onnxruntime-gpu
```

## License

MIT License - see LICENSE file for details.

## Credits

This library builds upon:
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) - Audio processing architecture
- [onnx-asr](https://github.com/istupakov/onnx-asr) - ONNX model inference
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) - Initial VAD

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
