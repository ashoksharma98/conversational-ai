# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import json
import tempfile
import os
from services.stt_service import STTService
from services.llm_service import LLMService
from services.tts_service import TTSService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Conversational AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (you'll need to provide API keys)
stt_service = STTService(model_name="base")
llm_service = LLMService(api_key="AIzaSyBmkjkJTuwmixHyOTsHXLzNF9-c5ZnpIwo")
tts_service = TTSService(voice="en-US-AriaNeural")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    await stt_service.load_model()
    llm_service.load_model()
    await tts_service.load_model()
    logger.info("All models loaded successfully")

@app.get("/")
async def root():
    return {"message": "Conversational AI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time conversational AI
    
    Client sends: Binary audio data (WAV format)
    Server sends: JSON status updates and binary audio responses
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Add periodic ping to keep connection alive
    async def send_ping():
        while True:
            try:
                await asyncio.sleep(20)  # Send ping every 20 seconds
                await websocket.send_json({"type": "ping"})
            except:
                break
    
    ping_task = asyncio.create_task(send_ping())
    
    try:
        while True:
            # Receive audio data from client
            audio_data = await websocket.receive_bytes()
            logger.info(f"Received audio data: {len(audio_data)} bytes")
            
            # Send status update
            await websocket.send_json({
                "status": "processing",
                "stage": "transcribing"
            })
            
            # Step 1: Convert audio to temp WAV file and transcribe
            try:
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav.write(audio_data)
                    temp_wav_path = temp_wav.name
                
                # Read the WAV file for transcription
                with open(temp_wav_path, 'rb') as f:
                    wav_bytes = f.read()
                
                # Transcribe using STT
                transcription = await stt_service.transcribe(wav_bytes)
                logger.info(f"Transcription: {transcription}")
                
                # Cleanup temp file
                os.unlink(temp_wav_path)
                
                # Send transcription to client
                await websocket.send_json({
                    "status": "transcribed",
                    "transcription": transcription
                })
                
            except Exception as e:
                logger.error(f"STT Error: {e}")
                await websocket.send_json({
                    "status": "error",
                    "stage": "transcription",
                    "message": str(e)
                })
                continue
            
            # Step 2: Generate LLM response
            try:
                await websocket.send_json({
                    "status": "processing",
                    "stage": "generating_response"
                })
                
                llm_response = await llm_service.generate_response(transcription)
                logger.info(f"LLM Response: {llm_response}")
                
                # Send LLM response to client
                await websocket.send_json({
                    "status": "response_generated",
                    "response": llm_response
                })
                
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                await websocket.send_json({
                    "status": "error",
                    "stage": "llm",
                    "message": str(e)
                })
                continue
            
            # Step 3: Convert response to speech
            try:
                await websocket.send_json({
                    "status": "processing",
                    "stage": "synthesizing_speech"
                })

                # Create a progress callback to send keepalive messages during TTS
                last_update = asyncio.get_event_loop().time()

                async def tts_progress_callback():
                    nonlocal last_update
                    current_time = asyncio.get_event_loop().time()
                    # Send update every 3 seconds during synthesis
                    if current_time - last_update > 3:
                        await websocket.send_json({
                            "status": "processing",
                            "stage": "synthesizing_speech",
                            "keepalive": True
                        })
                        last_update = current_time
                            
                audio_response = await tts_service.synthesize(llm_response, progress_callback=tts_progress_callback)
                logger.info(f"TTS generated: {len(audio_response)} bytes")
                
                # Send audio response to client
                await websocket.send_json({
                    "status": "audio_ready",
                    "audio_size": len(audio_response)
                })
                
                # Stream audio back to client
                await websocket.send_bytes(audio_response)
                
                # Send completion status
                await websocket.send_json({
                    "status": "completed"
                })
                
            except Exception as e:
                logger.error(f"TTS Error: {e}")
                await websocket.send_json({
                    "status": "error",
                    "stage": "tts",
                    "message": str(e)
                })
                continue
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        ping_task.cancel()
        logger.info("Cleaning up WebSocket connection")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
⏳ Processing: synthesizing_speech
⏳ Processing: synthesizing_speech
⏳ Processing: synthesizing_speech
⏳ Processing: synthesizing_speech
⏳ Processing: synthesizing_speech
🎵 Audio ready (size: 1249344 bytes)
❌ Error receiving message: sent 1009 (message too big) frame with 1101757 bytes exceeds limit of 1048576 bytes; no close frame received
❌ Error receiving message: received 1011 (internal error) keepalive ping timeout; then sent 1011 (internal error) keepalive ping timeout
"""
