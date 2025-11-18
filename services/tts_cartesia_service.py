# services/tts_cartesia_service.py
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from cartesia import AsyncCartesia
import base64

logger = logging.getLogger(__name__)


class CartesiaTTSService:
    """
    Cartesia Sonic TTS Service - Real-time streaming text-to-speech
    Uses official Cartesia Python SDK

    Features:
    - WebSocket-based streaming
    - Supports streaming text input (LLM tokens)
    - Streams audio chunks as they're generated
    - Context-based conversation continuity
    - Low latency audio generation
    """

    def __init__(
            self,
            api_key: str,
            model_id: str = "sonic-3",
            voice_id: str = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
            output_format: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Cartesia TTS Service

        Args:
            api_key: Cartesia API key
            model_id: Model to use (default: sonic-3)
            voice_id: Voice ID for TTS
            output_format: Audio output configuration
        """
        self.api_key = api_key
        self.model_id = model_id
        self.voice_id = voice_id

        # Default output format optimized for streaming
        self.output_format = output_format or {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 16000
        }

        # Cartesia client
        self.client: Optional[AsyncCartesia] = None
        self.websocket = None
        self.connected = False

        # Audio queue
        self.audio_queue: asyncio.Queue = asyncio.Queue()

        # Tasks
        self.receiver_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized CartesiaTTSService with model={model_id}, voice={voice_id}")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Cartesia TTS API using SDK

        Returns:
            bool: True if connected successfully
        """
        try:
            logger.info(f"Connecting to Cartesia TTS using SDK...")

            # Initialize Cartesia client
            self.client = AsyncCartesia(api_key=self.api_key)

            # Create WebSocket connection (no parameters here)
            self.websocket = await self.client.tts.websocket()

            self.connected = True
            logger.info("✅ Connected to Cartesia TTS WebSocket")

            # Start receiver task
            self.receiver_task = asyncio.create_task(self._audio_receiver())

            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to Cartesia TTS: {e}")
            self.connected = False
            return False

    async def synthesize_complete(
            self,
            text: str,
            context_id: Optional[str] = None
    ) -> bytes:
        """
        Synthesize complete text and return full audio

        Args:
            text: Complete text to synthesize
            context_id: Optional context ID

        Returns:
            bytes: Complete audio data
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to Cartesia TTS")

        try:
            logger.info(f"🎵 Synthesizing: {text[:50]}...")

            # Send to Cartesia
            response = await self.websocket.send(
                model_id=self.model_id,
                transcript=text,
                voice={"mode": "id", "id": self.voice_id},
                output_format=self.output_format,
                context_id=context_id or "default"
            )

            # Collect all audio chunks
            audio_chunks = []

            async for message in response:
                if hasattr(message, 'audio') and message.audio:
                    # Audio chunk received
                    audio_chunks.append(message.audio)
                    logger.debug(f"🔊 Received audio chunk: {len(message.audio)} bytes")

                if hasattr(message, 'done') and message.done:
                    logger.info("✅ TTS generation complete")
                    break

                if hasattr(message, 'type') and message.type == "error":
                    error_msg = getattr(message, 'message', 'Unknown error')
                    raise Exception(f"TTS Error: {error_msg}")

            return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"❌ Error in TTS synthesis: {e}")
            raise

    async def synthesize_streaming(
            self,
            text_generator: AsyncGenerator[str, None],
            context_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        High-level streaming synthesis from async text generator

        Args:
            text_generator: Async generator yielding text chunks (e.g., from LLM)
            context_id: Optional context ID

        Yields:
            bytes: Audio chunks
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to Cartesia TTS")

        ctx_id = context_id or "default"

        # Send text chunks as they arrive
        async def send_text_chunks():
            try:
                chunk_count = 0
                async for text_chunk in text_generator:
                    response = await self.websocket.send(
                        model_id=self.model_id,
                        transcript=text_chunk,
                        voice={"mode": "id", "id": self.voice_id},
                        output_format=self.output_format,
                        context_id=ctx_id,
                        continue_=True
                    )
                    chunk_count += 1
                    logger.debug(f"📤 Sent text chunk {chunk_count}: {text_chunk[:30]}...")

                    # Yield audio from this chunk's response
                    async for message in response:
                        if hasattr(message, 'audio') and message.audio:
                            yield message.audio

                        if hasattr(message, 'done') and message.done:
                            break

                logger.info("🏁 Completed streaming synthesis")

            except Exception as e:
                logger.error(f"❌ Error sending text chunks: {e}")

        try:
            # Yield audio chunks as they arrive from streaming
            async for audio_chunk in send_text_chunks():
                yield audio_chunk
        finally:
            pass

    async def _audio_receiver(self):
        """
        Background task: Receive audio chunks from WebSocket
        (Not actively used but kept for compatibility)
        """
        logger.info("🔊 Audio receiver task started")

        try:
            # This task is passive - actual receiving happens in synthesize methods
            while self.connected:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"❌ Audio receiver task failed: {e}")
        finally:
            logger.info("🔇 Audio receiver task stopped")

    async def cancel_generation(self, context_id: str = "default"):
        """
        Cancel current TTS generation

        Args:
            context_id: Context ID to cancel
        """
        if not self.connected or not self.websocket:
            return

        try:
            await self.websocket.cancel(context_id=context_id)
            logger.info(f"🛑 Cancelled TTS generation for context: {context_id}")
        except Exception as e:
            logger.error(f"❌ Error cancelling TTS: {e}")

    async def close(self):
        """
        Close connection and cleanup resources
        """
        logger.info("🔌 Closing Cartesia TTS connection...")

        self.connected = False

        # Cancel receiver task
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        # Close client
        if self.client:
            try:
                await self.client.close()
            except:
                pass
            self.client = None

        logger.info("✅ Cartesia TTS connection closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()