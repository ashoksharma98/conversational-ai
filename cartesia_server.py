from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import json
import tempfile
import os
from contextlib import asynccontextmanager
from services.stt_cartesia_service import CartesiaService
from services.llm_service import LLMService

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = LLMService(api_key=os.getenv('GEMINI_API_KEY'))
    llm.load_model()
    yield

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan, title="Conversational AI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/conversation/cartesia")
async def websocket_conversation(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    cartesia_service = CartesiaService()
    llm = LLMService(api_key=os.getenv('GEMINI_API_KEY'))

    while True:
        try:
            audio_data = await websocket.receive_bytes()
            logger.info(f"Received audio {len(audio_data)} bytes")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(audio_data)
                temp_path = temp_wav.name

            with open(temp_path, 'rb') as f:
                wav_bytes = f.read()

            transcription = cartesia_service.transcribe(wav_bytes)['text']
            logger.info(f"Transcription: {transcription}")

            llm_resp = await llm.generate_response(transcription)
            logger.info(f"LLM Response: {llm_resp}")

            tts_resp = cartesia_service.synthesize(llm_resp)
            logger.info("TTS Response generated...")

            os.unlink(temp_path)

            await websocket.send_bytes(tts_resp)

            await websocket.send_json({
                "status": "transcribed",
                "transcription": transcription,
                "llm_response": llm_resp
            })

        except WebSocketDisconnect:
            logger.info("Client disconnected")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            await websocket.send_json({"status": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_max_size=16777216  # 16MB limit for WebSocket messages
    )
