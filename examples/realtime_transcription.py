"""
Example demonstrating realtime transcription callbacks.
"""
from realtimestt_onnx import AudioToTextRecorder


def on_realtime_update(text):
    """Called during speech for realtime updates."""
    print(f"\r🎤 Realtime: {text}", end="", flush=True)


def on_realtime_stabilized(text):
    """Called when realtime transcription stabilizes."""
    print(f"\n📝 Stabilized: {text}")


def on_final_transcription(text):
    """Called when final transcription is complete."""
    print(f"\n✅ Final: {text}\n")


def main():
    print("=== Realtime Transcription Demo ===")
    print("This example shows realtime transcription updates during speech.")
    print("Speak into your microphone. Press Ctrl+C to exit.\n")
    
    # Create recorder with realtime transcription enabled
    recorder = AudioToTextRecorder(
        model="nemo-parakeet-tdt-0.6b-v3",
        enable_realtime_transcription=True,
        on_realtime_transcription_update=on_realtime_update,
        on_realtime_transcription_stabilized=on_realtime_stabilized,
        spinner=True,  # Show spinner during final transcription
    )
    
    try:
        while True:
            recorder.text(on_final_transcription)
    except KeyboardInterrupt:
        print("\nStopping...")
        recorder.stop()


if __name__ == "__main__":
    main()
