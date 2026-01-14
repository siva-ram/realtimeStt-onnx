"""
Simple example demonstrating basic realtime transcription.
"""
from realtimestt_onnx import AudioToTextRecorder


def on_text(text):
    """Callback function to handle transcribed text."""
    print(f"Transcribed: {text}")


def main():
    print("Initializing realtime speech-to-text...")
    print("Wait until ready, then speak into your microphone.")
    print("Press Ctrl+C to exit.\n")
    
    # Create recorder with default settings
    # Uses nemo-parakeet-tdt-0.6b-v3 model by default
    recorder = AudioToTextRecorder()
    
    try:
        while True:
            # Block until speech is detected, recorded, and transcribed
            recorder.text(on_text)
    except KeyboardInterrupt:
        print("\nStopping...")
        recorder.stop()


if __name__ == "__main__":
    main()
