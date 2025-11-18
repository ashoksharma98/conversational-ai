# client_cartesia.py
import asyncio
import websockets
import json
import pyaudio
import logging
from typing import Optional
from collections import deque
import wave
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio parameters (matches Cartesia requirements)
CHUNK_SIZE = 1600  # 50ms at 16kHz (1600 samples = 3200 bytes)
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
RATE = 16000  # 16kHz sample rate


class CartesiaConversationalClient:
    """
    Real-time streaming conversational AI client using Cartesia

    Features:
    - Continuous audio streaming to server
    - Real-time transcription feedback
    - Streaming audio playback
    - Simple push-to-talk interface
    """

    def __init__(self, websocket_url: str = "ws://localhost:8001/ws/conversation/cartesia"):
        self.websocket_url = websocket_url
        self.audio = pyaudio.PyAudio()
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

        # State
        self.is_recording = False
        self.is_playing = False
        self.should_stop = False

        # Audio buffers
        self.playback_buffer = deque()

        logger.info("🎙️ Cartesia Conversational Client initialized")

    async def connect(self) -> bool:
        """Connect to WebSocket server"""
        try:
            logger.info(f"🔌 Connecting to {self.websocket_url}...")

            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10,
                max_size=16777216  # 16MB
            )

            # Wait for ready message
            ready_msg = await self.websocket.recv()
            if isinstance(ready_msg, str):
                status = json.loads(ready_msg)
                if status.get("status") == "ready":
                    logger.info("✅ Connected! Server is ready")
                    return True

            logger.error("❌ Server not ready")
            return False

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    async def stream_audio_to_server(self):
        """
        Continuously stream audio chunks to server while recording
        """
        logger.info("🎤 Starting audio streaming...")

        # Open microphone stream
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        try:
            while self.is_recording and not self.should_stop:
                # Read audio chunk
                audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Send to server
                if self.websocket:
                    await self.websocket.send(audio_chunk)

                # Small delay to prevent CPU overuse
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"❌ Error streaming audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            logger.info("🔇 Audio streaming stopped")

    async def receive_server_messages(self):
        """
        Receive and process messages from server
        """
        logger.info("📡 Listening for server messages...")

        try:
            while not self.should_stop and self.websocket:
                message = await self.websocket.recv()

                # Handle JSON status messages
                if isinstance(message, str):
                    try:
                        status = json.loads(message)
                        await self.handle_status_message(status)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Invalid JSON: {message[:100]}")

                # Handle binary audio data
                elif isinstance(message, bytes):
                    await self.handle_audio_data(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("📡 Server connection closed")
        except Exception as e:
            logger.error(f"❌ Error receiving messages: {e}")

    async def handle_status_message(self, status: dict):
        """Process status messages from server"""
        status_type = status.get("status")

        if status_type == "transcribing":
            # Interim transcription
            text = status.get("text", "")
            is_final = status.get("is_final", False)

            if not is_final:
                # Show interim (live transcription)
                print(f"\r📝 Transcribing: {text}", end="", flush=True)
            else:
                print()  # New line after interim

        elif status_type == "transcribed":
            # Final transcription
            text = status.get("text", "")
            logger.info(f"✅ You said: {text}")

        elif status_type == "response_generated":
            # LLM response
            text = status.get("text", "")
            logger.info(f"💬 AI: {text}")

        elif status_type == "processing":
            # Processing stage
            stage = status.get("stage", "")
            if not status.get("keepalive"):
                logger.info(f"⏳ {stage.replace('_', ' ').title()}...")

        elif status_type == "audio_ready":
            # Audio metadata
            audio_size = status.get("audio_size", 0)
            logger.info(f"🎵 Audio ready ({audio_size} bytes)")
            self.is_playing = True

        elif status_type == "completed":
            # Turn complete
            logger.info("✅ Turn completed!")
            self.is_playing = False

            # Stop recording for this turn
            self.is_recording = False

        elif status_type == "error":
            # Error occurred
            stage = status.get("stage", "unknown")
            message = status.get("message", "Unknown error")
            logger.error(f"❌ Error at {stage}: {message}")

    async def handle_audio_data(self, audio_data: bytes):
        """Handle incoming audio data"""
        # Add to playback buffer
        self.playback_buffer.append(audio_data)

    def play_audio_buffer(self):
        """
        Play all audio from buffer (blocking)
        Called in separate thread/after receiving all audio
        """
        if not self.playback_buffer:
            return

        logger.info("🔊 Playing audio response...")

        try:
            # Combine all chunks
            complete_audio = b"".join(self.playback_buffer)
            self.playback_buffer.clear()

            # Create WAV in memory for playback
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(complete_audio)

            wav_buffer.seek(0)

            # Play using PyAudio
            with wave.open(wav_buffer, 'rb') as wf:
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )

                # Read and play in chunks
                data = wf.readframes(4096)
                while data:
                    stream.write(data)
                    data = wf.readframes(4096)

                stream.stop_stream()
                stream.close()

            logger.info("✅ Playback finished!")

        except Exception as e:
            logger.error(f"❌ Error playing audio: {e}")

    async def conversation_turn(self):
        """
        Execute one conversation turn:
        1. Start recording
        2. Stream audio to server
        3. Receive transcription and response
        4. Play audio response
        """
        logger.info("\n" + "=" * 60)
        logger.info("🎙️  Speak now... (will auto-stop on silence)")
        logger.info("=" * 60)

        # Start recording immediately
        self.is_recording = True

        # Create tasks for streaming and receiving (run concurrently)
        stream_task = asyncio.create_task(self.stream_audio_to_server())
        receive_task = asyncio.create_task(self.receive_server_messages())

        # Wait for both tasks to run
        # receive_task will keep running and handle server messages
        # stream_task will stream audio while is_recording is True

        try:
            # Wait for recording to stop (triggered by server's completed message)
            while self.is_recording:
                await asyncio.sleep(0.1)

            logger.info("🔇 Recording stopped")

            # Cancel streaming task since we're done recording
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

            # Continue receiving until audio playback is done
            logger.info("⏳ Waiting for response...")
            while self.is_playing or not self.playback_buffer:
                await asyncio.sleep(0.1)

                # Timeout after 30 seconds
                if not self.is_playing and not self.playback_buffer:
                    # Check if we've been waiting too long
                    await asyncio.sleep(0.1)

            # Cancel receive task
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

            # Play the audio response
            if self.playback_buffer:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.play_audio_buffer
                )

        except Exception as e:
            logger.error(f"❌ Error in conversation turn: {e}")

            # Cleanup tasks
            stream_task.cancel()
            receive_task.cancel()

            try:
                await stream_task
            except:
                pass

            try:
                await receive_task
            except:
                pass

    async def run(self):
        """Main conversation loop"""
        # Connect to server
        if not await self.connect():
            logger.error("Failed to connect to server")
            return

        logger.info("\n" + "=" * 60)
        logger.info("🤖 Cartesia Conversational AI Client")
        logger.info("=" * 60)
        logger.info("Commands:")
        logger.info("  - Press ENTER to start speaking")
        logger.info("  - Type 'q' or 'quit' to exit")
        logger.info("=" * 60 + "\n")

        try:
            while not self.should_stop:
                # Get user input
                user_input = input("Press ENTER to speak (or 'q' to quit): ")

                if user_input.lower() in ['q', 'quit', 'exit']:
                    logger.info("👋 Goodbye!")
                    break

                # Execute conversation turn
                await self.conversation_turn()

        except KeyboardInterrupt:
            logger.info("\n👋 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in conversation loop: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up...")

        self.should_stop = True

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()

        # Terminate PyAudio
        self.audio.terminate()

        logger.info("✅ Cleanup complete")


async def main():
    """Main entry point"""
    client = CartesiaConversationalClient()

    try:
        await client.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())