"""
Thread-safe audio buffer for managing recorded audio chunks.
"""
import threading
import numpy as np
from collections import deque
from typing import Optional


class AudioBuffer:
    """
    A thread-safe circular buffer for audio data.
    
    This buffer handles audio chunks efficiently with automatic overflow
    management and format normalization.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        dtype: type = np.float32
    ):
        """
        Initialize the audio buffer.
        
        Args:
            sample_rate: Sample rate of the audio in Hz
            max_duration: Maximum buffer duration in seconds
            dtype: Data type for audio samples (np.float32 or np.int16)
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.dtype = dtype
        self.max_samples = int(sample_rate * max_duration)
        
        self._buffer = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
        
    def append(self, audio_chunk: np.ndarray) -> None:
        """
        Append audio chunk to the buffer.
        
        Args:
            audio_chunk: Audio samples as numpy array
        """
        with self._lock:
            # Normalize to target dtype
            normalized = self._normalize_audio(audio_chunk)
            
            # Add samples to buffer
            for sample in normalized:
                self._buffer.append(sample)
    
    def get_audio(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Get audio from the buffer.
        
        Args:
            duration: Duration of audio to retrieve in seconds.
                     If None, returns all buffered audio.
        
        Returns:
            Audio samples as numpy array
        """
        with self._lock:
            if duration is None:
                # Return all buffered audio
                return np.array(list(self._buffer), dtype=self.dtype)
            else:
                # Return last N seconds
                num_samples = int(self.sample_rate * duration)
                num_samples = min(num_samples, len(self._buffer))
                
                if num_samples == 0:
                    return np.array([], dtype=self.dtype)
                
                # Get last num_samples from buffer
                samples = list(self._buffer)[-num_samples:]
                return np.array(samples, dtype=self.dtype)
    
    def clear(self) -> None:
        """Clear all audio from the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        """Get number of samples in buffer."""
        with self._lock:
            return len(self._buffer)
    
    def duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target dtype.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        # Flatten if needed
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Convert to target dtype
        if self.dtype == np.float32:
            if audio.dtype == np.int16:
                # Convert int16 to float32 in range [-1.0, 1.0]
                return audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                return audio.astype(np.float32) / 2147483648.0
            else:
                return audio.astype(np.float32)
        elif self.dtype == np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Convert float to int16
                return (audio * 32768.0).astype(np.int16)
            else:
                return audio.astype(np.int16)
        else:
            return audio.astype(self.dtype)
