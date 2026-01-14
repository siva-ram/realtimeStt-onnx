"""
Example demonstrating speaker audio transcription using WASAPI loopback.
"""
from realtimestt_onnx import AudioToTextRecorder
from realtimestt_onnx.audio_devices import print_audio_devices, get_default_speaker_loopback


def on_text(text):
    """Callback function to handle transcribed text."""
    print(f"🔊 Speaker: {text}\n")


def main():
    print("=== Speaker Audio Transcription (WASAPI Loopback) ===")
    print("This example transcribes audio playing from your speakers.")
    print("Note: This works on Windows with WASAPI support.\n")
    
    # Show available devices
    print_audio_devices()
    
    try:
        # Get default speaker loopback device
        speaker_index = get_default_speaker_loopback()
        print(f"Using speaker loopback device index: {speaker_index}\n")
    except Exception as e:
        print(f"Error: {e}")
        print("WASAPI loopback may not be available on this system.")
        return
    
    print("Starting transcription of speaker audio...")
    print("Play some audio/video and it will be transcribed.")
    print("Press Ctrl+C to exit.\n")
    
    # Create recorder with speaker input
    recorder = AudioToTextRecorder(
        model="nemo-parakeet-tdt-0.6b-v3",
        input_device_type="speaker",  # Capture from speakers instead of mic
        input_device_index=speaker_index,
        on_recording_start=lambda: print("🎵 Audio detected..."),
        on_recording_stop=lambda: print("⏸️  Audio stopped"),
    )
    
    try:
        while True:
            recorder.text(on_text)
    except KeyboardInterrupt:
        print("\nStopping...")
        recorder.stop()


if __name__ == "__main__":
    main()
