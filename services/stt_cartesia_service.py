# services/stt_cartesia_service.py
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from cartesia import AsyncCartesia
import requests

logger = logging.getLogger(__name__)


class CartesiaService:
    def __init__(self, api_key: str):
        self._stt_url = 'https://api.cartesia.ai/stt'
        self._tts_url = 'https://api.cartesia.ai/tts/bytes'
        self._headers = { 'Authorization': f'Bearer {api_key}', 'Cartesia-Version': '2025-04-16' }

    def transcribe(self, wav_bytes: bytes) -> dict:
        try:
            files = {
                "file": ("audio.wav", wav_bytes, "audio/wav")
            }

            payload = {
                "model": "ink-whisper",
                "language": "en",
                "timestamp_granularities[]": "word"
            }

            response = requests.post(
                self._stt_url,
                headers=self._headers,
                data=payload,
                files=files
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error transcribing audio: {e}")

    def synthesize(self, transcript: str) -> bytes:
        self._headers['Content-Type'] = 'application/json'
        payload = {
            "model_id": "sonic-3",
            "transcript": f"{transcript}",
            "voice": {
                "mode": "id",
                "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02"
            },
            "language": "en",
            "generation_config": {
                "volume": 1,
                "speed": 1,
                "emotion": "neutral"
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 16000
            },
            "save": False,
            "pronunciation_dict_id": None,
            "speed": "normal"
        }
        try:
            resp = requests.post(self._tts_url, json=payload, headers=self._headers)
            resp.raise_for_status()
            return resp.content
        except requests.ConnectTimeout as timeout_error:
            raise Exception(f"Timeout error: {timeout_error}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error synthesizing audio: {e}")
        except Exception as err:
            raise Exception(f"Error synthesizing audio: {err}")


# class CartesiaSTTService:
#     """
#     Cartesia Ink Whisper STT Service - Real-time streaming speech-to-text
#     Uses official Cartesia Python SDK
#
#     Features:
#     - WebSocket-based streaming via Cartesia SDK
#     - Built-in VAD (Voice Activity Detection)
#     - Interim and final transcriptions
#     """
#
#     def __init__(
#             self,
#             api_key: str,
#             model: str = "ink-whisper",
#             language: str = "en",
#             sample_rate: int = 16000,
#             encoding: str = "pcm_s16le",
#             min_volume: float = 0.1,
#             max_silence_duration_secs: float = 1.5
#     ):
#         """
#         Initialize Cartesia STT Service
#
#         Args:
#             api_key: Cartesia API key
#             model: Model to use (default: ink-whisper)
#             language: Language code (default: en)
#             sample_rate: Audio sample rate in Hz (default: 16000)
#             encoding: Audio encoding (default: pcm_s16le)
#             min_volume: Minimum volume threshold for VAD (0.0-1.0)
#             max_silence_duration_secs: Max silence before ending utterance
#         """
#         self.api_key = api_key
#         self.model = model
#         self.language = language
#         self.sample_rate = sample_rate
#         self.encoding = encoding
#         self.min_volume = min_volume
#         self.max_silence_duration_secs = max_silence_duration_secs
#
#         # Cartesia client
#         self.client: Optional[AsyncCartesia] = None
#         self.websocket = None
#         self.connected = False
#
#         logger.info(f"Initialized CartesiaSTTService with model={model}, language={language}")
#
#     async def connect(self) -> bool:
#         """
#         Establish WebSocket connection to Cartesia STT API using SDK
#
#         Returns:
#             bool: True if connected successfully
#         """
#         try:
#             logger.info(f"Connecting to Cartesia STT using SDK...")
#
#             # Initialize Cartesia client
#             self.client = AsyncCartesia(api_key=self.api_key)
#
#             # Create WebSocket connection with parameters
#             self.websocket = await self.client.stt.websocket(
#                 model=self.model,
#                 language=self.language,
#                 encoding=self.encoding,
#                 sample_rate=self.sample_rate,
#                 min_volume=self.min_volume,
#                 max_silence_duration_secs=self.max_silence_duration_secs
#             )
#
#             self.connected = True
#             logger.info("✅ Connected to Cartesia STT WebSocket")
#
#             return True
#
#         except Exception as e:
#             logger.error(f"❌ Failed to connect to Cartesia STT: {e}")
#             self.connected = False
#             return False
#
#     async def send_audio(self, audio_chunk: bytes):
#         """
#         Send audio chunk to Cartesia STT
#
#         Args:
#             audio_chunk: Raw PCM audio bytes (pcm_s16le, 16kHz)
#         """
#         if not self.connected or not self.websocket:
#             logger.warning("⚠️ Not connected, cannot send audio")
#             return
#
#         try:
#             await self.websocket.send(audio_chunk)
#         except Exception as e:
#             logger.error(f"❌ Error sending audio: {e}")
#
#     async def receive_transcriptions(self) -> AsyncGenerator[Dict[str, Any], None]:
#         """
#         Async generator that yields transcription events
#
#         Yields:
#             dict: Transcription event with keys:
#                 - type: "interim", "final", "done", "error"
#                 - text: Transcribed text (for interim/final)
#                 - is_final: Boolean (for final events)
#
#         Example:
#             async for event in stt.receive_transcriptions():
#                 if event['type'] == 'final':
#                     print(f"Final transcription: {event['text']}")
#         """
#         if not self.connected or not self.websocket:
#             logger.error("❌ Not connected, cannot receive transcriptions")
#             return
#
#         try:
#             logger.info("📝 Starting to receive transcriptions...")
#
#             async for result in self.websocket.receive():
#                 event = {}
#
#                 # Extract text if available
#                 if hasattr(result, 'text') and result.text:
#                     event['text'] = result.text
#
#                 # Determine event type
#                 if hasattr(result, 'is_final'):
#                     event['is_final'] = result.is_final
#                     event['type'] = 'final' if result.is_final else 'interim'
#
#                     # Log appropriately
#                     if result.is_final:
#                         logger.info(f"✅ Final: {result.text}")
#                     else:
#                         logger.debug(f"📝 Interim: {result.text[:50]}...")
#
#                 # Check for special event types
#                 if hasattr(result, 'type'):
#                     event['type'] = result.type
#
#                     if result.type == "done":
#                         logger.info("🏁 Utterance done")
#                         event['done'] = True
#                     elif result.type == "error":
#                         error_msg = getattr(result, 'message', 'Unknown error')
#                         logger.error(f"❌ STT Error: {error_msg}")
#                         event['message'] = error_msg
#                     elif result.type == "flush_done":
#                         logger.info("🚽 Flush done")
#
#                 # Yield the event
#                 if event:  # Only yield if we have data
#                     yield event
#
#         except Exception as e:
#             logger.error(f"❌ Error receiving transcriptions: {e}")
#             self.connected = False
#
#     async def flush(self):
#         """
#         Flush/finalize current utterance
#         """
#         if not self.connected or not self.websocket:
#             return
#
#         try:
#             # Many SDK websocket wrappers implement an async flush() — try to await it.
#             flush_method = getattr(self.websocket, "flush", None)
#             if flush_method is None:
#                 logger.warning("⚠️ STT websocket has no flush() method")
#                 return
#
#             # If flush is coroutineable, await it; else call it synchronously.
#             import inspect
#             if inspect.iscoroutinefunction(flush_method) or inspect.isawaitable(flush_method):
#                 await flush_method()
#             else:
#                 # might be a normal function; call it
#                 flush_method()
#
#             logger.info("🚽 Flushed STT buffer")
#         except Exception as e:
#             logger.error(f"❌ Error flushing STT: {e}")
#
#     async def close(self):
#         """
#         Close connection and cleanup resources
#         """
#         logger.info("🔌 Closing Cartesia STT connection...")
#
#         self.connected = False
#
#         # Close WebSocket
#         if self.websocket:
#             try:
#                 await self.websocket.close()
#             except:
#                 pass
#             self.websocket = None
#
#         # Close client
#         if self.client:
#             try:
#                 await self.client.close()
#             except:
#                 pass
#             self.client = None
#
#         logger.info("✅ Cartesia STT connection closed")
#
#     async def __aenter__(self):
#         """Async context manager entry"""
#         await self.connect()
#         return self
#
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         """Async context manager exit"""
#         await self.close()