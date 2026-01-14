"""
ONNX-based speech recognition engine.
"""
import numpy as np
import onnx_asr
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class ONNXEngine:
    """
    Speech recognition engine using ONNX models.
    
    This engine wraps onnx-asr functionality and handles:
    - Model loading from HuggingFace or local paths
    - Audio preprocessing and normalization
    - VAD-based chunking for long audio
    - Batch transcription
    """
    
    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: Optional[str] = None,
        use_vad: bool = True,
        vad_threshold: float = 0.5,
        device: str = "cpu",
        providers: Optional[List] = None
    ):
        """
        Initialize the ONNX engine.
        
        Args:
            model_name: Name of the ONNX model to load (e.g., "nemo-parakeet-tdt-0.6b-v3")
                       or path to local ONNX model directory
            quantization: Optional quantization ("int8", "fp16", None)
            use_vad: Whether to use VAD for audio segmentation
            vad_threshold: VAD threshold for speech detection
            device: Device to use ("cpu" or "cuda")
            providers: Optional ONNX Runtime providers list
        """
        self.model_name = model_name
        self.quantization = quantization
        self.use_vad = use_vad
        self.vad_threshold = vad_threshold
        
        # Set up providers based on device
        if providers is None:
            if device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        
        self.providers = providers
        
        # Load the model
        logger.info(f"Loading ONNX model: {model_name}")
        try:
            self.model = onnx_asr.load_model(
                model_name,
                quantization=quantization,
                providers=providers
            )
            
            # Add VAD if requested
            if use_vad:
                try:
                    vad = onnx_asr.load_vad("silero")
                    self.model = self.model.with_vad(vad, threshold=vad_threshold)
                    logger.info("VAD enabled for audio segmentation")
                except Exception as e:
                    logger.warning(f"Failed to load VAD: {e}. Proceeding without VAD.")
                    self.use_vad = False
            
            logger.info(f"Model loaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: int = 16000
    ) -> Union[str, List[str]]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio file path or numpy array of audio samples
            sample_rate: Sample rate of the audio (if audio is numpy array)
            
        Returns:
            Transcribed text. If VAD is enabled and audio is long,
            returns a string (concatenated segments) or list of transcriptions
        """
        try:
            if isinstance(audio, np.ndarray):
                # Ensure audio is in correct format
                audio = self._preprocess_audio(audio, sample_rate)
                
                # Recognize with sample rate
                result = self.model.recognize(audio, sample_rate=sample_rate)
            else:
                # Audio is a file path
                result = self.model.recognize(audio)
            
            # Handle different return types
            if isinstance(result, list):
                # VAD segmented the audio, concatenate results
                return " ".join(result)
            else:
                return result
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def transcribe_batch(
        self,
        audio_list: List[Union[str, np.ndarray]],
        sample_rate: int = 16000
    ) -> List[str]:
        """
        Transcribe multiple audio files/arrays in batch.
        
        Args:
            audio_list: List of audio file paths or numpy arrays
            sample_rate: Sample rate (for numpy arrays)
            
        Returns:
            List of transcribed texts
        """
        try:
            # Process each audio
            processed_audio = []
            for audio in audio_list:
                if isinstance(audio, np.ndarray):
                    processed_audio.append(
                        self._preprocess_audio(audio, sample_rate)
                    )
                else:
                    processed_audio.append(audio)
            
            # Batch recognize
            results = self.model.recognize(processed_audio)
            
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]
            
            # Handle VAD-segmented results
            final_results = []
            for result in results:
                if isinstance(result, list):
                    final_results.append(" ".join(result))
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            return [""] * len(audio_list)
    
    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Preprocess audio for ONNX model.
        
        Args:
            audio: Audio numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Preprocessed audio array
        """
        # Ensure float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
        
        # Ensure mono (flatten if stereo)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Normalize to [-1, 1] if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'quantization': self.quantization,
            'use_vad': self.use_vad,
            'providers': self.providers
        }
