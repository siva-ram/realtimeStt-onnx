"""
Tests for the audio buffer module.
"""
import unittest
import numpy as np
from realtimestt_onnx.audio_buffer import AudioBuffer


class TestAudioBuffer(unittest.TestCase):
    """Test cases for AudioBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = AudioBuffer(sample_rate=16000, max_duration=2.0)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.sample_rate, 16000)
        self.assertEqual(self.buffer.max_duration, 2.0)
        self.assertEqual(len(self.buffer), 0)
    
    def test_append_audio(self):
        """Test appending audio to buffer."""
        # Create 1 second of audio
        audio = np.random.randn(16000).astype(np.float32)
        self.buffer.append(audio)
        
        self.assertGreater(len(self.buffer), 0)
        self.assertAlmostEqual(self.buffer.duration(), 1.0, places=2)
    
    def test_get_audio(self):
        """Test retrieving audio from buffer."""
        # Add some audio
        audio = np.random.randn(16000).astype(np.float32)
        self.buffer.append(audio)
        
        # Get all audio
        retrieved = self.buffer.get_audio()
        self.assertEqual(len(retrieved), len(audio))
    
    def test_get_audio_duration(self):
        """Test retrieving specific duration."""
        # Add 2 seconds of audio
        audio = np.random.randn(32000).astype(np.float32)
        self.buffer.append(audio)
        
        # Get last 0.5 seconds
        retrieved = self.buffer.get_audio(duration=0.5)
        self.assertAlmostEqual(len(retrieved), 8000, delta=100)
    
    def test_clear(self):
        """Test clearing the buffer."""
        audio = np.random.randn(16000).astype(np.float32)
        self.buffer.append(audio)
        
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.duration(), 0.0)
    
    def test_int16_to_float32_conversion(self):
        """Test audio format conversion."""
        # Create int16 audio
        audio_int16 = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        self.buffer.append(audio_int16)
        
        # Retrieved audio should be float32
        retrieved = self.buffer.get_audio()
        self.assertEqual(retrieved.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(retrieved)), 1.0)
    
    def test_overflow_handling(self):
        """Test buffer overflow with max_duration."""
        buffer = AudioBuffer(sample_rate=16000, max_duration=1.0)
        
        # Add 2 seconds of audio (should keep only last 1 second)
        audio = np.random.randn(32000).astype(np.float32)
        buffer.append(audio)
        
        # Buffer should only have max_duration worth
        self.assertLessEqual(buffer.duration(), 1.1)  # Small tolerance


if __name__ == '__main__':
    unittest.main()
