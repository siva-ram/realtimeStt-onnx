"""
Example: Simultaneous microphone and speaker transcription with a single ONNX model.
"""
from realtimestt_onnx import MultiSourceRecorder
from realtimestt_onnx.audio_devices import get_default_speaker_loopback


def on_transcription(text: str, source: str):
    """
    Callback for transcribed text.
    
    Args:
        text: The transcribed text
        source: Which source it came from ("Microphone" or "Speakers")
    """
    # Use different emojis for different sources
    emoji = "🎤" if "Microphone" in source else "🔊"
    print(f"{emoji} [{source}]: {text}")


def main():
    print("=== Simultaneous Mic + Speaker Transcription ===")
    print("This example transcribes BOTH your microphone AND speaker audio")
    print("using a SINGLE shared ONNX model (memory efficient!).\n")
    
    # Create multi-source recorder with shared ONNX engine
    multi_recorder = MultiSourceRecorder(
        model="nemo-parakeet-tdt-0.6b-v3",
        device="cpu",  # Change to "cuda" for GPU
        spinner=True
    )
    
    # Add microphone source
    multi_recorder.add_source(
        input_device_type="microphone",
        source_name="Microphone",
        on_text=on_transcription,
        on_recording_start=lambda: print("🎤 Microphone: Recording...")
    )
    
    # Add speaker source (WASAPI loopback)
    try:
        speaker_index = get_default_speaker_loopback()
        multi_recorder.add_source(
            input_device_type="speaker",
            input_device_index=speaker_index,
            source_name="Speakers",
            on_text=on_transcription,
            on_recording_start=lambda: print("🔊 Speakers: Audio detected...")
        )
        print("✅ Both microphone and speaker sources configured!\n")
    except Exception as e:
        print(f"⚠️  Speaker loopback not available: {e}")
        print("Will only transcribe microphone.\n")
    
    print("🚀 Starting transcription...")
    print("Speak into your microphone OR play audio from speakers.")
    print("Press Ctrl+C to stop.\n")
    
    # Start recording from all sources
    try:
        multi_recorder.record_continuously()
    except KeyboardInterrupt:
        print("\n\nStopping...")
        multi_recorder.stop_all()
        print("✅ Done!")


if __name__ == "__main__":
    main()
