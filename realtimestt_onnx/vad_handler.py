"""
Voice Activity Detection (VAD) handler combining WebRTC VAD and Silero VAD.
"""
import numpy as np
import webrtcvad
import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VADHandler:
    """
    Voice Activity Detection handler using two-stage detection:
    1. WebRTC VAD for fast initial detection
    2. Silero VAD for accurate verification
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        webrtc_aggressiveness: int = 3,
        silero_threshold: float = 0.5,
        use_silero: bool = True,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500
    ):
        """
        Initialize the VAD handler.
        
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000 for WebRTC)
            webrtc_aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
            silero_threshold: Silero VAD threshold (0-1, higher = more confident speech required)
            use_silero: Whether to use Silero VAD for verification
            min_speech_duration_ms: Minimum speech duration to consider as valid speech
            min_silence_duration_ms: Minimum silence duration to end speech
        """
        self.sample_rate = sample_rate
        self.use_silero = use_silero
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # Initialize WebRTC VAD
        self.webrtc_vad = webrtcvad.Vad(webrtc_aggressiveness)
        
        # Initialize Silero VAD if enabled
        self.silero_model = None
        self.silero_threshold = silero_threshold
        if use_silero:
            try:
                # Load Silero VAD model
                self.silero_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.silero_get_speech_timestamps = utils[0]
                logger.info("Silero VAD model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Silero VAD: {e}. Using WebRTC VAD only.")
                self.use_silero = False
        
        # State tracking
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speech_active = False
    
    def is_speech(self, audio_chunk: bytes, duration_ms: int = 30) -> bool:
        """
        Check if audio chunk contains speech using WebRTC VAD.
        
        Args:
            audio_chunk: Raw audio bytes (must be 10, 20, or 30 ms at supported sample rate)
            duration_ms: Duration of the chunk in milliseconds
            
        Returns:
            True if speech is detected, False otherwise
        """
        try:
            return self.webrtc_vad.is_speech(audio_chunk, self.sample_rate)
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return False
    
    def verify_speech_silero(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Verify speech using Silero VAD.
        
        Args:
            audio: Audio as float32 numpy array
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        if not self.use_silero or self.silero_model is None:
            return True, 1.0  # Assume speech if Silero is not available
        
        try:
            # Ensure audio is float32 and in correct shape
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Silero expects audio normalized to [-1, 1]
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / 32768.0
            
            # Check minimum audio length to prevent "Input audio chunk is too short" error
            # Silero VAD requires at least 512 samples (0.032 seconds at 16kHz)
            min_samples = max(512, int(self.sample_rate * 0.032))
            if len(audio) < min_samples:
                logger.debug(f"Audio too short for Silero VAD ({len(audio)} samples), skipping verification")
                return True, 1.0  # Assume speech if too short to verify
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio)
            
            # Get speech probability
            with torch.no_grad():
                speech_prob = self.silero_model(audio_tensor, self.sample_rate).item()
            
            is_speech = speech_prob >= self.silero_threshold
            return is_speech, speech_prob
            
        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return True, 1.0  # Default to speech on error
    
    def process_audio(
        self,
        audio_chunk: bytes,
        audio_array: Optional[np.ndarray] = None,
        duration_ms: int = 30
    ) -> dict:
        """
        Process audio chunk and return VAD state.
        
        Args:
            audio_chunk: Raw audio bytes for WebRTC VAD
            audio_array: Audio as numpy array for Silero VAD (optional)
            duration_ms: Duration of chunk in milliseconds
            
        Returns:
            Dictionary with VAD state information
        """
        # Fast WebRTC check
        webrtc_speech = self.is_speech(audio_chunk, duration_ms)
        
        result = {
            'webrtc_speech': webrtc_speech,
            'silero_speech': None,
            'silero_confidence': None,
            'final_speech': webrtc_speech,
            'speech_started': False,
            'speech_ended': False
        }
        
        # Verify with Silero if enabled and audio_array provided
        if self.use_silero and audio_array is not None and webrtc_speech:
            silero_speech, confidence = self.verify_speech_silero(audio_array)
            result['silero_speech'] = silero_speech
            result['silero_confidence'] = confidence
            result['final_speech'] = silero_speech
        
        return result
    
    def reset(self) -> None:
        """Reset VAD state."""
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speech_active = False
