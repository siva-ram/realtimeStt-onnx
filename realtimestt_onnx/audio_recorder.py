"""
Main AudioToTextRecorder class for realtime speech-to-text.
"""
import pyaudiowpatch as pyaudio
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable
from queue import Queue

from .audio_buffer import AudioBuffer
from .vad_handler import VADHandler
from .onnx_engine import ONNXEngine

logger = logging.getLogger(__name__)


class AudioToTextRecorder:
    """
    Realtime audio-to-text recorder using ONNX models.
    
    This class handles:
    - Audio capture from microphone
    - Voice activity detection
    - Realtime transcription using ONNX models
    - Callback-based event system
    """
    
    def __init__(
        self,
        model: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: Optional[str] = None,
        device: str = "cpu",
        input_device_index: Optional[int] = None,
        input_device_type: str = "microphone",  # "microphone" or "speaker"
        sample_rate: int = 16000,
        chunk_size: int = 480,  # 30ms at 16kHz
        
        # VAD parameters
        webrtc_aggressiveness: int = 3,
        silero_threshold: float = 0.5,
        use_silero_vad: bool = True,
        
        # Speech detection parameters
        min_speech_duration_ms: int = 250,
        post_speech_silence_duration: float = 0.5,
        
        # Callbacks
        on_recording_start: Optional[Callable] = None,
        on_recording_stop: Optional[Callable] = None,
        on_transcription_start: Optional[Callable] = None,
        on_realtime_transcription_update: Optional[Callable[[str], None]] = None,
        on_realtime_transcription_stabilized: Optional[Callable[[str], None]] = None,
        
        # Other settings
        level: int = logging.WARNING,
        use_microphone: bool = True,
        spinner: bool = True,
        enable_realtime_transcription: bool = False,
        onnx_engine: Optional['ONNXEngine'] = None  # Allow external engine
    ):
        """
        Initialize the AudioToTextRecorder.
        
        Args:
            model: ONNX model name or path
            quantization: Model quantization ("int8", "fp16", or None)
            device: Device to use ("cpu" or "cuda")
            input_device_index: Microphone/Speaker device index (None for default)
            input_device_type: Input source type ("microphone" or "speaker" for loopback)
            sample_rate: Audio sample rate in Hz
            chunk_size: Audio chunk size in samples
            enable_realtime_transcription: Enable realtime partial transcriptions
            webrtc_aggressiveness: WebRTC VAD aggressiveness (0-3)
            silero_threshold: Silero VAD threshold (0-1)
            use_silero_vad: Whether to use Silero VAD
            min_speech_duration_ms: Minimum speech duration in ms
            post_speech_silence_duration: Silence duration to end recording (seconds)
            on_recording_start: Callback when recording starts
            on_recording_stop: Callback when recording stops
            on_transcription_start: Callback when transcription starts
            on_realtime_transcription_update: Callback for realtime transcription updates
            on_realtime_transcription_stabilized: Callback for stabilized realtime transcription
            level: Logging level
            use_microphone: Whether to use microphone (False to feed audio manually)
            spinner: Whether to show spinner during transcription
            enable_realtime_transcription: Enable realtime partial transcriptions during speech
        """
        # Set up logging
        logging.basicConfig(level=level)
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.use_microphone = use_microphone
        self.input_device_index = input_device_index
        self.input_device_type = input_device_type
        
        # VAD and speech detection
        self.post_speech_silence_duration = post_speech_silence_duration
        self.min_speech_duration_ms = min_speech_duration_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        
        # Callbacks
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_transcription_start = on_transcription_start
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.on_realtime_transcription_stabilized = on_realtime_transcription_stabilized
        
        # Spinner
        self.spinner = None
        self.use_spinner = spinner
        if spinner:
            try:
                from halo import Halo
                self.spinner = Halo(spinner='dots')
            except ImportError:
                logger.warning("halo not installed. Install with: pip install halo")
        
        # Initialize components
        if onnx_engine is not None:
            # Use shared engine
            self.onnx_engine = onnx_engine
            self._owns_engine = False
            logger.info("Using shared ONNX engine")
        else:
            # Create new engine
            self.onnx_engine = ONNXEngine(
                model_name=model,
                quantization=quantization,
                use_vad=False,  # We handle VAD separately
                device=device
            )
            self._owns_engine = True
        
        self.vad_handler = VADHandler(
            sample_rate=sample_rate,
            webrtc_aggressiveness=webrtc_aggressiveness,
            silero_threshold=silero_threshold,
            use_silero=use_silero_vad,
            min_speech_duration_ms=min_speech_duration_ms
        )
        
        self.audio_buffer = AudioBuffer(
            sample_rate=sample_rate,
            max_duration=30.0,
            dtype=np.float32
        )
        
        # Audio stream
        self.audio = None
        self.stream = None
        
        # State
        self.is_recording = False
        self.is_running = False
        self.speech_detected = False
        self.silence_start = None
        
        # Threading
        self.audio_queue = Queue()
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        logger.info("AudioToTextRecorder initialized")
    
    def start(self) -> None:
        """Start the audio recorder."""
        if self.is_running:
            logger.warning("Recorder is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        if self.use_microphone:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Determine device and input mode
            device_index = self.input_device_index
            if self.input_device_type == "speaker":
                # Use WASAPI loopback for speaker audio on Windows
                if device_index is None:
                    # Get default loopback device
                    try:
                        wasapi_info = self.audio.get_host_api_info_by_type(pyaudio.paWASAPI)
                        default_speakers = self.audio.get_device_info_by_index(
                            wasapi_info["defaultOutputDevice"]
                        )
                        
                        # Check if loopback is available
                        if default_speakers["maxInputChannels"] > 0:
                            device_index = wasapi_info["defaultOutputDevice"]
                            logger.info(f"Using default WASAPI loopback: {default_speakers['name']}")
                        else:
                            logger.error("WASAPI loopback not available on this device")
                            raise RuntimeError("Speaker loopback not supported")
                    except Exception as e:
                        logger.error(f"Failed to get WASAPI loopback device: {e}")
                        raise
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            source_type = "speaker" if self.input_device_type == "speaker" else "microphone"
            logger.info(f"{source_type.capitalize()} stream started")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("AudioToTextRecorder started")
    
    def stop(self) -> None:
        """Stop the audio recorder."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop stream
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
        
        # Wait for processing thread
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("AudioToTextRecorder stopped")
    
    def text(self, callback: Callable[[str], None]) -> None:
        """
        Record audio and transcribe to text.
        
        This method blocks until speech is detected, recorded, and transcribed.
        
        Args:
            callback: Callback function that receives the transcribed text
        """
        if not self.is_running:
            self.start()
        
        # Reset state
        self.audio_buffer.clear()
        self.speech_detected = False
        self.silence_start = None
        self.vad_handler.reset()
        
        # Wait for speech to be detected and recorded
        while self.is_running and not self.speech_detected:
            time.sleep(0.1)
        
        # Handle realtime transcription if enabled
        if self.enable_realtime_transcription and self.on_realtime_transcription_update:
            last_transcription = ""
            while self.is_running and self.is_recording:
                # Get current audio buffer
                audio = self.audio_buffer.get_audio()
                if len(audio) > self.sample_rate * 0.5:  # At least 0.5 seconds
                    # Quick transcription for realtime feedback
                    text = self.onnx_engine.transcribe(audio, self.sample_rate)
                    if text and text != last_transcription:
                        self.on_realtime_transcription_update(text)
                        last_transcription = text
                time.sleep(0.2)  # Update every 200ms
        else:
            # Wait for recording to complete
            while self.is_running and self.is_recording:
                time.sleep(0.1)
        
        # Get recorded audio
        audio = self.audio_buffer.get_audio()
        
        if len(audio) > 0:
            # Start spinner if enabled
            if self.spinner:
                self.spinner.start("Transcribing...")
            
            # Trigger transcription callback
            if self.on_transcription_start:
                self.on_transcription_start()
            
            # Transcribe
            text = self.onnx_engine.transcribe(audio, self.sample_rate)
            
            # Stop spinner
            if self.spinner:
                self.spinner.succeed("Transcription complete")
            
            # Call user callback with result
            if text and callback:
                callback(text)
        
        # Reset for next recording
        self.audio_buffer.clear()
        self.speech_detected = False
    
    def feed_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Feed audio chunk manually (when use_microphone=False).
        
        Args:
            audio_chunk: Audio samples as numpy array
        """
        if self.use_microphone:
            logger.warning("Cannot feed audio manually when use_microphone=True")
            return
        
        self.audio_queue.put(audio_chunk)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to queue for processing
        self.audio_queue.put((in_data, audio_chunk))
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self) -> None:
        """Process audio chunks from queue."""
        while not self.stop_event.is_set():
            try:
                # Get audio chunk from queue (with timeout)
                audio_data = self.audio_queue.get(timeout=0.1)
                
                if audio_data is None:
                    continue
                
                # Unpack audio data
                if isinstance(audio_data, tuple):
                    raw_bytes, audio_array = audio_data
                else:
                    # Manual feed
                    audio_array = audio_data
                    raw_bytes = audio_array.tobytes()
                
                # Convert to float32
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # VAD check
                vad_result = self.vad_handler.process_audio(
                    raw_bytes,
                    audio_float,
                    duration_ms=int(len(audio_array) / self.sample_rate * 1000)
                )
                
                is_speech = vad_result['final_speech']
                
                if is_speech:
                    # Speech detected
                    if not self.speech_detected:
                        self.speech_detected = True
                        self.is_recording = True
                        self.silence_start = None
                        if self.on_recording_start:
                            self.on_recording_start()
                        logger.debug("Speech started")
                    
                    # Add to buffer
                    self.audio_buffer.append(audio_float)
                
                elif self.speech_detected:
                    # Silence during/after speech
                    if self.silence_start is None:
                        self.silence_start = time.time()
                    
                    # Check if silence duration exceeded
                    silence_duration = time.time() - self.silence_start
                    if silence_duration >= self.post_speech_silence_duration:
                        # End of speech
                        self.is_recording = False
                        if self.on_recording_stop:
                            self.on_recording_stop()
                        logger.debug(f"Speech ended (silence: {silence_duration:.2f}s)")
                        self.silence_start = None
                    else:
                        # Still adding silence to buffer
                        self.audio_buffer.append(audio_float)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error processing audio: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
