# server_cartesia.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import json
import os
from typing import Optional
from dotenv import load_dotenv

from services.stt_cartesia_service import CartesiaSTTService
from services.tts_cartesia_service import CartesiaTTSService
from services.llm_service import LLMService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Cartesia Conversational AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM service (reusing existing Gemini service)
llm_service = LLMService(api_key=os.getenv("GEMINI_API_KEY"))


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Cartesia Conversational AI Server...")
    llm_service.load_model()
    logger.info("✅ Server ready!")


@app.get("/")
async def root():
    return {
        "message": "Cartesia Conversational AI API is running",
        "endpoints": {
            "health": "/health",
            "websocket": "/ws/conversation/cartesia"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "provider": "cartesia"}


@app.websocket("/ws/conversation/cartesia")
async def websocket_cartesia_conversation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time conversational AI using Cartesia

    Flow:
    1. Client connects and streams audio chunks
    2. Server uses Cartesia STT to transcribe in real-time
    3. Final transcription → Gemini LLM
    4. LLM response → Cartesia TTS (streaming)
    5. Audio chunks streamed back to client

    Client sends: Binary audio data (PCM, 16kHz, mono)
    Server sends: JSON status updates + binary audio chunks
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"🔌 WebSocket connection established [Client: {client_id}]")

    # Initialize Cartesia services
    stt_service: Optional[CartesiaSTTService] = None
    tts_service: Optional[CartesiaTTSService] = None

    # State tracking
    current_transcription = ""
    is_processing = False

    try:
        # Get API key from environment
        cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        if not cartesia_api_key:
            await websocket.send_json({
                "status": "error",
                "message": "CARTESIA_API_KEY not configured"
            })
            await websocket.close()
            return

        # Initialize STT service
        logger.info("🎤 Initializing Cartesia STT service...")
        stt_service = CartesiaSTTService(
            api_key=cartesia_api_key,
            min_volume=0.1,
            max_silence_duration_secs=1.5
        )
        await stt_service.connect()

        # Initialize TTS service
        logger.info("🔊 Initializing Cartesia TTS service...")
        tts_service = CartesiaTTSService(
            api_key=cartesia_api_key,
            voice_id="6ccbfb76-1fc6-48f7-b71d-91ac6298247b"  # Default voice
        )
        await tts_service.connect()

        await websocket.send_json({
            "status": "ready",
            "message": "Cartesia services initialized"
        })
        logger.info("✅ Cartesia services ready")

        # Create tasks for concurrent operations
        async def handle_client_audio():
            """Receive audio from client and forward to STT"""
            nonlocal is_processing

            try:
                while True:
                    # Use the low-level receive() so we can inspect text vs bytes
                    message = await websocket.receive()

                    # message is a dict: either {'type': 'websocket.receive', 'bytes': b'...'}
                    # or {'type': 'websocket.receive', 'text': '...'}
                    # or {'type': 'websocket.disconnect'}
                    if message is None:
                        continue

                    # WebSocket disconnect (handshake closed)
                    if message.get("type") == "websocket.disconnect":
                        logger.info("📡 Client disconnected (disconnect message)")
                        break

                    # Binary audio frame
                    if "bytes" in message and message["bytes"] is not None:
                        audio_data = message["bytes"]

                        # If currently processing a response, ignore new audio (or buffer if you want)
                        if is_processing:
                            logger.debug("⚠️ Ignoring audio - currently processing response")
                            continue

                        # Forward to Cartesia STT
                        await stt_service.send_audio(audio_data)
                        continue

                    # Text message (JSON control)
                    if "text" in message and message["text"] is not None:
                        try:
                            data = json.loads(message["text"])
                        except Exception:
                            # Not JSON — ignore or log
                            logger.debug("⚠️ Received non-json text message from client")
                            continue

                        # Handle explicit client-end signal for utterance flush
                        msg_type = data.get("type")
                        if msg_type == "end_utterance":
                            logger.info("📨 Received end_utterance from client -> flushing STT")
                            # ask the STT service to flush / finalize current utterance
                            try:
                                await stt_service.flush()
                            except Exception as e:
                                logger.error(f"❌ Error flushing STT: {e}")
                                await websocket.send_json({
                                    "status": "error",
                                    "stage": "stt",
                                    "message": str(e)
                                })
                            continue

                        # If you want to support other control messages from client, handle here:
                        # e.g., {"type":"cancel"} or {"type":"ping"}
                        logger.debug(f"📩 Received control message: {data}")
                        continue

            except WebSocketDisconnect:
                logger.info("📡 Client disconnected")
            except Exception as e:
                logger.error(f"❌ Error handling client audio: {e}")

        async def handle_stt_transcriptions():
            """Process transcriptions from STT and trigger LLM + TTS"""
            nonlocal current_transcription, is_processing

            try:
                async for event in stt_service.receive_transcriptions():
                    event_type = event.get("type")

                    if event_type == "interim":
                        # Send interim transcription to client (optional)
                        interim_text = event.get("text", "")
                        await websocket.send_json({
                            "status": "transcribing",
                            "text": interim_text,
                            "is_final": False
                        })
                        logger.debug(f"📝 Interim: {interim_text[:50]}...")

                    elif event_type == "final":
                        # Final transcription received
                        current_transcription = event.get("text", "")
                        logger.info(f"✅ Final transcription: {current_transcription}")

                        await websocket.send_json({
                            "status": "transcribed",
                            "text": current_transcription,
                            "is_final": True
                        })

                        # Process with LLM and TTS
                        if current_transcription.strip():
                            is_processing = True
                            await process_llm_and_tts(current_transcription)
                            is_processing = False

                            # Start new utterance for next turn
                            await stt_service.start_new_utterance()

                    elif event_type == "done":
                        logger.info("🏁 STT utterance complete")

                    elif event_type == "error":
                        error_msg = event.get("message", "Unknown error")
                        logger.error(f"❌ STT Error: {error_msg}")
                        await websocket.send_json({
                            "status": "error",
                            "stage": "stt",
                            "message": error_msg
                        })

            except Exception as e:
                logger.error(f"❌ Error in STT handler: {e}")

        async def process_llm_and_tts(transcription: str):
            """Process transcription through LLM and synthesize response"""
            try:
                # Step 1: Generate LLM response
                await websocket.send_json({
                    "status": "processing",
                    "stage": "generating_response"
                })

                logger.info("🤖 Generating LLM response...")

                # Progress callback for LLM
                async def llm_progress():
                    await websocket.send_json({
                        "status": "processing",
                        "stage": "generating_response",
                        "keepalive": True
                    })

                llm_response = await llm_service.generate_response(
                    transcription,
                    progress_callback=llm_progress
                )

                logger.info(f"💬 LLM Response: {llm_response}")

                await websocket.send_json({
                    "status": "response_generated",
                    "text": llm_response
                })

                # Step 2: Synthesize with TTS (streaming)
                await websocket.send_json({
                    "status": "processing",
                    "stage": "synthesizing_speech"
                })

                logger.info("🎵 Synthesizing speech...")

                # Generate audio using Cartesia TTS
                audio_data = await tts_service.synthesize_complete(llm_response)

                logger.info(f"✅ TTS complete: {len(audio_data)} bytes")

                # Send audio metadata
                await websocket.send_json({
                    "status": "audio_ready",
                    "audio_size": len(audio_data),
                    "format": "pcm_s16le",
                    "sample_rate": 16000
                })

                # Send audio data
                # Chunk if necessary (512KB chunks)
                CHUNK_SIZE = 512 * 1024

                if len(audio_data) > CHUNK_SIZE:
                    num_chunks = (len(audio_data) + CHUNK_SIZE - 1) // CHUNK_SIZE

                    for i in range(0, len(audio_data), CHUNK_SIZE):
                        chunk = audio_data[i:i + CHUNK_SIZE]
                        await websocket.send_bytes(chunk)
                        logger.debug(f"📤 Sent audio chunk {i // CHUNK_SIZE + 1}/{num_chunks}")
                else:
                    await websocket.send_bytes(audio_data)

                # Send completion status
                await websocket.send_json({
                    "status": "completed"
                })

                logger.info("✅ Conversation turn completed")

            except Exception as e:
                logger.error(f"❌ Error in LLM/TTS processing: {e}")
                await websocket.send_json({
                    "status": "error",
                    "stage": "llm_tts",
                    "message": str(e)
                })

        # Run both handlers concurrently
        audio_task = asyncio.create_task(handle_client_audio())
        stt_task = asyncio.create_task(handle_stt_transcriptions())

        # Wait for either task to complete (or fail)
        done, pending = await asyncio.wait(
            [audio_task, stt_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check if any task raised an exception
        for task in done:
            if task.exception():
                raise task.exception()

    except WebSocketDisconnect:
        logger.info(f"📡 Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Cleanup Cartesia services
        logger.info("🧹 Cleaning up Cartesia services...")

        if stt_service:
            await stt_service.close()

        if tts_service:
            await tts_service.close()

        logger.info(f"✅ Cleanup complete [Client: {client_id}]")


if __name__ == "__main__":
    import uvicorn

    # Check for required environment variables
    if not os.getenv("CARTESIA_API_KEY"):
        logger.error("❌ CARTESIA_API_KEY not found in environment")
        exit(1)

    if not os.getenv("GEMINI_API_KEY"):
        logger.error("❌ GEMINI_API_KEY not found in environment")
        exit(1)

    logger.info("🚀 Starting Cartesia Conversational AI Server...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port from your existing server
        ws_max_size=16777216  # 16MB limit for WebSocket messages
    )