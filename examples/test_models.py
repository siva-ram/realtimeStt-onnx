"""
Example showing how to use different ONNX models.
"""
from realtimestt_onnx import AudioToTextRecorder


def test_model(model_name, description):
    """Test a specific ONNX model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    print("Speak now...\n")
    
    try:
        recorder = AudioToTextRecorder(model=model_name)
        
        def on_text(text):
            print(f"✅ Transcribed: {text}\n")
            print("Stopping test...")
        
        # Record one utterance
        recorder.text(on_text)
        recorder.stop()
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def main():
    print("=== ONNX Model Comparison ===")
    print("This example tests different ONNX models.")
    print("You'll be prompted to speak for each model.\n")
    
    models = [
        ("nemo-parakeet-ctc-0.6b", "Fast Parakeet CTC (English)"),
        ("whisper-base", "OpenAI Whisper Base"),
        # Add more models as needed
    ]
    
    for model_name, description in models:
        input(f"\nPress Enter to test {model_name}...")
        test_model(model_name, description)
    
    print("\n✨ All tests complete!")


if __name__ == "__main__":
    main()
