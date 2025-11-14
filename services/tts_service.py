# services/tts_service.py
import logging
import edge_tts
from io import BytesIO

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using Edge TTS"""
    
    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
        logger.info(f"TTS Service initialized with Edge TTS, voice: {voice}")
    
    async def load_model(self):
        """No model loading needed for Edge TTS"""
        logger.info("Edge TTS ready")
    
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech using Edge TTS
        
        Args:
            text: Input text to synthesize
            
        Returns:
            bytes: Audio data (MP3 format)
        """
        try:
            logger.info(f"Synthesizing speech for: {text}")
            
            # Create communicator
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Generate speech and collect bytes
            audio_data = BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
            
            audio_bytes = audio_data.getvalue()
            logger.info(f"Speech synthesized successfully, size: {len(audio_bytes)} bytes")
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error during speech synthesis: {e}")
            raise