"""
Example showing custom VAD configuration.
"""
from realtimestt_onnx import AudioToTextRecorder


def on_text(text):
    print(f"Transcribed: {text}")


def main():
    print("=== Custom VAD Configuration ===")
    print("This example uses custom VAD settings for different environments.\n")
    
    # Create recorder with custom VAD settings
    # Higher webrtc_aggressiveness for noisy environments
    # Lower silero_threshold for more sensitive detection
    recorder = AudioToTextRecorder(
        model="nemo-parakeet-tdt-0.6b-v3",
        
        # VAD settings
        webrtc_aggressiveness=3,    # 0-3, higher = more aggressive filtering
        silero_threshold=0.3,        # 0-1, lower = more sensitive
        use_silero_vad=True,         # Use two-stage VAD
        
        # Speech detection timing
        min_speech_duration_ms=250,              # Minimum speech duration
        post_speech_silence_duration=0.7,        # Silence before ending (seconds)
    )
    
    print("Configuration:")
    print("  - WebRTC Aggressiveness: 3 (high)")
    print("  - Silero Threshold: 0.3 (sensitive)")
    print("  - Silence duration: 0.7s")
    print("\nSpeak now...\n")
    
    try:
        while True:
            recorder.text(on_text)
    except KeyboardInterrupt:
        print("\nStopping...")
        recorder.stop()


if __name__ == "__main__":
    main()
