"""
Example demonstrating all callback features.
"""
from realtimestt_onnx import AudioToTextRecorder
import logging


def on_recording_start():
    """Called when recording starts (speech detected)."""
    print("\n🎤 Recording started...")


def on_recording_stop():
    """Called when recording stops (silence detected)."""
    print("⏸️  Recording stopped")


def on_transcription_start():
    """Called when transcription begins."""
    print("🔄 Transcribing...")


def on_text(text):
    """Called when transcription is complete."""
    print(f"✅ Transcribed: {text}\n")


def main():
    print("=== Realtime STT with Callbacks ===")
    print("This example shows all available callbacks.")
    print("Speak into your microphone. Press Ctrl+C to exit.\n")
    
    # Create recorder with all callbacks enabled
    recorder = AudioToTextRecorder(
        model="nemo-parakeet-tdt-0.6b-v3",
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_transcription_start=on_transcription_start,
        level=logging.INFO  # Show info logs
    )
    
    try:
        while True:
            recorder.text(on_text)
    except KeyboardInterrupt:
        print("\nStopping...")
        recorder.stop()


if __name__ == "__main__":
    main()
