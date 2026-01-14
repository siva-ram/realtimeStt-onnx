"""
Multi-source audio recorder for simultaneous transcription of multiple audio sources.
"""
import threading
import time
import logging
from typing import Optional, Callable, List
from queue import Queue

from .audio_recorder import AudioToTextRecorder
from .onnx_engine import ONNXEngine

logger = logging.getLogger(__name__)


class MultiSourceRecorder:
    """
    Manage multiple audio sources with a single shared ONNX engine.
    
    Perfect for transcribing microphone and speaker audio simultaneously
    without loading the model twice.
    """
    
    def __init__(
        self,
        model: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: Optional[str] = None,
        device: str = "cpu",
        sample_rate: int = 16000,
        spinner: bool = True
    ):
        """
        Initialize multi-source recorder with shared ONNX engine.
        
        Args:
            model: ONNX model name or path
            quantization: Model quantization ("int8", "fp16", or None)
            device: Device to use ("cpu" or "cuda")
            sample_rate: Audio sample rate in Hz
            spinner: Whether to show spinner
        """
        # Create shared ONNX engine
        self.onnx_engine = ONNXEngine(
            model_name=model,
            quantization=quantization,
            use_vad=False,
            device=device
        )
        
        self.sample_rate = sample_rate
        self.spinner = spinner
        self.recorders: List[AudioToTextRecorder] = []
        self.running = False
        
        logger.info("MultiSourceRecorder initialized with shared ONNX engine")
    
    def add_source(
        self,
        input_device_type: str,
        input_device_index: Optional[int] = None,
        on_text: Optional[Callable[[str, str], None]] = None,
        source_name: Optional[str] = None,
        **kwargs
    ) -> AudioToTextRecorder:
        """
        Add an audio source (microphone or speaker).
        
        Args:
            input_device_type: "microphone" or "speaker"
            input_device_index: Device index (None for default)
            on_text: Callback(text, source_name) when text is transcribed
            source_name: Name for this source (e.g., "Microphone", "Speakers")
            **kwargs: Additional AudioToTextRecorder parameters
            
        Returns:
            AudioToTextRecorder instance for this source
        """
        # Generate source name if not provided
        if source_name is None:
            source_name = f"{input_device_type.capitalize()}-{len(self.recorders)}"
        
        # Create callback wrapper to include source name
        def wrapped_callback(text):
            if on_text:
                on_text(text, source_name)
        
        # Create recorder with shared engine
        recorder = AudioToTextRecorder(
            input_device_type=input_device_type,
            input_device_index=input_device_index,
            sample_rate=self.sample_rate,
            onnx_engine=self.onnx_engine,  # Share the engine!
            spinner=False,  # Disable individual spinners
            **kwargs
        )
        
        # Store the callback and source name
        recorder._source_name = source_name
        recorder._on_text_callback = wrapped_callback
        
        self.recorders.append(recorder)
        logger.info(f"Added source: {source_name}")
        
        return recorder
    
    def start_all(self):
        """Start recording from all sources simultaneously."""
        logger.info("Starting all audio sources...")
        for recorder in self.recorders:
            recorder.start()
        self.running = True
        logger.info(f"All {len(self.recorders)} sources started")
    
    def stop_all(self):
        """Stop recording from all sources."""
        logger.info("Stopping all audio sources...")
        for recorder in self.recorders:
            recorder.stop()
        self.running = False
        logger.info("All sources stopped")
    
    def record_continuously(self):
        """
        Continuously record and transcribe from all sources.
        
        Run in separate threads for each source. Press Ctrl+C to stop.
        """
        if not self.running:
            self.start_all()
        
        # Create thread for each source
        threads = []
        for recorder in self.recorders:
            thread = threading.Thread(
                target=self._record_loop,
                args=(recorder,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop_all()
    
    def _record_loop(self, recorder: AudioToTextRecorder):
        """Recording loop for a single source."""
        while self.running:
            try:
                recorder.text(recorder._on_text_callback)
            except Exception as e:
                logger.error(f"Error in {recorder._source_name}: {e}")
                time.sleep(0.1)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all()
