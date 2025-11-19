# services/stt_cartesia_service.py
import asyncio
import websockets
import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from cartesia import AsyncCartesia
import requests
import base64
import uuid

logger = logging.getLogger(__name__)


class CartesiaSTTService:
    """
    Bare minimum Cartesia STT streaming service
    Accepts audio chunks and yields transcription results
    """

    def __init__(
            self,
            api_key: str,
            model: str = "ink-whisper",
            language: str = "en",
            encoding: str = "pcm_s16le",
            sample_rate: int = 16000,
            min_volume: float = 0.1,
            max_silence_duration_secs: float = 2.0
    ):
        self.api_key = api_key
        self.wss_url = "wss://api.cartesia.ai/stt/websocket"

        # Build query parameters
        self.params = {
            "model": model,
            "language": language,
            "encoding": encoding,
            "sample_rate": str(sample_rate),
            "min_volume": str(min_volume),
            "max_silence_duration_secs": str(max_silence_duration_secs)
        }

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

    async def connect(self):
        """Establish WebSocket connection to Cartesia STT"""
        try:
            # Build URL with query parameters
            params_str = "&".join([f"{k}={v}" for k, v in self.params.items()])
            url = f"{self.wss_url}?{params_str}"

            self.websocket = await websockets.connect(
                url,
                additional_headers={
                    "X-API-Key": self.api_key,
                    "Cartesia-Version": "2024-06-10"
                },
                max_size=10485760
            )

            self.connected = True
            logger.info("✅ Connected to Cartesia STT")

        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise

    async def send_audio(self, audio_chunk: bytes):
        """Send audio chunk to Cartesia"""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to Cartesia STT")

        await self.websocket.send(audio_chunk)

    async def receive_transcriptions(self) -> AsyncGenerator[dict, None]:
        """
        Receive transcription results from Cartesia

        Yields:
            dict: Transcription result with keys:
                - type: "transcript", "done", "flush_done", "error"
                - text: Transcribed text (for transcript type)
                - is_final: Boolean (for transcript type)
                - words: List of word-level timestamps (optional)
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to Cartesia STT")

        try:
            async for message in self.websocket:
                result = json.loads(message)
                yield result

        except Exception as e:
            logger.error(f"❌ Error receiving transcriptions: {e}")
            raise

    async def finalize(self):
        """Send finalize command to flush remaining audio"""
        if self.websocket:
            await self.websocket.send("finalize")

    async def close(self, send_done: bool = True):
        if self.websocket:
            try:
                if send_done:
                    await self.websocket.send("done")
                await self.websocket.close()
            except:
                pass
            finally:
                self.websocket = None
                self.connected = False


class CartesiaTTSService:
    """
    Bare minimum Cartesia TTS streaming service
    Accepts text chunks and yields audio chunks
    """

    def __init__(
            self,
            api_key: str,
            model_id: str = "sonic-3",
            voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
            language: str = "en",
            sample_rate: int = 16000,
            encoding: str = "pcm_s16le"
    ):
        self.api_key = api_key
        self.wss_url = "wss://api.cartesia.ai/tts/websocket"
        self.model_id = model_id
        self.voice_id = voice_id
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.context_id = None

    async def connect(self):
        """Establish WebSocket connection to Cartesia TTS"""
        try:
            self.websocket = await websockets.connect(
                self.wss_url,
                additional_headers={
                    "X-API-Key": self.api_key,
                    "Cartesia-Version": "2024-06-10"
                },
                max_size=10485760  # 10MB
            )

            self.connected = True
            logger.info("✅ Connected to Cartesia TTS")

        except Exception as e:
            logger.error(f"❌ Failed to connect to TTS: {e}")
            raise

    async def send_text(self, text: str, continue_: bool = True):
        """
        Send text chunk to Cartesia for synthesis

        Args:
            text: Text to synthesize
            continue_: True if more text coming, False for final chunk
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to Cartesia TTS")

        # Generate context_id on first send
        if self.context_id is None:
            self.context_id = str(uuid.uuid4())

        message = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id
            },
            "language": self.language,
            "context_id": self.context_id,
            "output_format": {
                "container": "raw",
                "encoding": self.encoding,
                "sample_rate": self.sample_rate
            },
            "continue": continue_
        }

        await self.websocket.send(json.dumps(message))
        logger.debug(f"📤 Sent text: {text[:50]}... (continue={continue_})")

    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Receive audio chunks from Cartesia

        Yields:
            bytes: Raw PCM audio data
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to Cartesia TTS")

        try:
            async for message in self.websocket:
                result = json.loads(message)
                msg_type = result.get("type")

                if msg_type == "chunk":
                    # Decode base64 audio data
                    audio_b64 = result.get("data", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    yield audio_bytes

                elif msg_type == "done":
                    logger.info("✅ TTS generation complete")
                    break

                elif msg_type == "error":
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"❌ TTS Error: {error_msg}")
                    break

                # Ignore timestamps, flush_done, etc.

        except Exception as e:
            logger.error(f"❌ Error receiving audio: {e}")
            raise

    async def stream_synthesis(
            self,
            text_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        High-level streaming: take text chunks, yield audio chunks

        Args:
            text_generator: Async generator yielding text chunks (from LLM)

        Yields:
            bytes: Audio chunks as they're generated
        """

        # Task to send text chunks
        async def send_text_chunks():
            try:
                buffer = ""
                async for text_chunk in text_generator:
                    buffer += text_chunk

                    # Send every few words or on punctuation
                    if len(buffer.split()) >= 5 or any(p in buffer for p in '.!?,;'):
                        await self.send_text(buffer, continue_=True)
                        buffer = ""

                # Send final chunk
                if buffer:
                    await self.send_text(buffer, continue_=False)
                else:
                    # Send empty final chunk to signal completion
                    await self.send_text("", continue_=False)

                logger.info("🏁 Finished sending text")

            except Exception as e:
                logger.error(f"❌ Error sending text chunks: {e}")

        # Start sending text in background
        send_task = asyncio.create_task(send_text_chunks())

        try:
            # Yield audio chunks as they arrive
            async for audio_chunk in self.receive_audio():
                yield audio_chunk
        finally:
            # Wait for sending to complete
            await send_task

    async def synthesize_complete(self, text: str) -> bytes:
        """
        Synthesize complete text and return all audio

        Args:
            text: Complete text to synthesize

        Returns:
            bytes: Complete audio data
        """
        await self.send_text(text, continue_=False)

        audio_chunks = []
        async for chunk in self.receive_audio():
            audio_chunks.append(chunk)

        return b"".join(audio_chunks)

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            finally:
                self.websocket = None
                self.connected = False
                self.context_id = None

        logger.info("✅ Cartesia TTS connection closed")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# class CartesiaService:
#     def __init__(self, api_key: str):
#         self._stt_url = 'https://api.cartesia.ai/stt'
#         self._tts_url = 'https://api.cartesia.ai/tts/bytes'
#         self._headers = { 'Authorization': f'Bearer {api_key}', 'Cartesia-Version': '2025-04-16' }
#
#     def transcribe(self, wav_bytes: bytes) -> dict:
#         try:
#             files = {
#                 "file": ("audio.wav", wav_bytes, "audio/wav")
#             }
#
#             payload = {
#                 "model": "ink-whisper",
#                 "language": "en",
#                 "timestamp_granularities[]": "word"
#             }
#
#             response = requests.post(
#                 self._stt_url,
#                 headers=self._headers,
#                 data=payload,
#                 files=files
#             )
#
#             response.raise_for_status()
#             return response.json()
#
#         except requests.exceptions.RequestException as e:
#             raise Exception(f"Error transcribing audio: {e}")
#
#     def synthesize(self, transcript: str) -> bytes:
#         self._headers['Content-Type'] = 'application/json'
#         payload = {
#             "model_id": "sonic-3",
#             "transcript": f"{transcript}",
#             "voice": {
#                 "mode": "id",
#                 "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02"
#             },
#             "language": "en",
#             "generation_config": {
#                 "volume": 1,
#                 "speed": 1,
#                 "emotion": "neutral"
#             },
#             "output_format": {
#                 "container": "wav",
#                 "encoding": "pcm_s16le",
#                 "sample_rate": 16000
#             },
#             "save": False,
#             "pronunciation_dict_id": None,
#             "speed": "normal"
#         }
#         try:
#             resp = requests.post(self._tts_url, json=payload, headers=self._headers)
#             resp.raise_for_status()
#             return resp.content
#         except requests.ConnectTimeout as timeout_error:
#             raise Exception(f"Timeout error: {timeout_error}")
#         except requests.exceptions.RequestException as e:
#             raise Exception(f"Error synthesizing audio: {e}")
#         except Exception as err:
#             raise Exception(f"Error synthesizing audio: {err}")


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