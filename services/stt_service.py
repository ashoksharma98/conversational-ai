# services/stt_service.py
import logging
import whisper
import torch
import numpy as np
from io import BytesIO
import wave

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service using Whisper"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"STT Service initialized with device: {self.device}")
    
    async def load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper {self.model_name} model...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: WAV file as bytes
            
        Returns:
            str: Transcribed text
        """
        if self.model is None:
            await self.load_model()
        
        try:
            # Read WAV file from bytes
            with BytesIO(audio_data) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wav_file:
                    # Get audio parameters
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    # Read audio frames
                    audio_frames = wav_file.readframes(n_frames)
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_frames, dtype=np.int16)
                    
                    # Convert to float32 and normalize to [-1, 1]
                    audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            logger.info("Transcribing audio...")
            result = self.model.transcribe(audio_float, language="en")
            transcribed_text = result["text"].strip()
            
            logger.info(f"Transcription: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
