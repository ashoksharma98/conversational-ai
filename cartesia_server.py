from fastapi import FastAPI, WebSocket, WebSocketDisconnect, responses
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import json
import tempfile
import os
from contextlib import asynccontextmanager
from services.stt_cartesia_service import CartesiaSTTService, CartesiaTTSService
from services.llm_service import LLMService, GeminiService

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stt = CartesiaSTTService(api_key=os.getenv('CARTESIA_API_KEY'))
llm = GeminiService(api_key=os.getenv('GEMINI_API_KEY'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # l_ = LLMService(api_key=os.getenv('GEMINI_API_KEY'))
    # l_._ensure_model()
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


async def consume_cartesia_transcriptions(stt: CartesiaSTTService, websocket: WebSocket):
    """Single long-running task that consumes all transcription events"""
    try:
        async for result in stt.receive_transcriptions():
            rtype = result.get("type")

            if rtype == "transcript":
                text = result.get("text", "")
                is_final = result.get("is_final", False)

                await websocket.send_json({
                    "status": "partial" if not is_final else "final",
                    "transcription": text,
                    "is_final": is_final
                })

                if is_final and text:
                    logger.info(f"User: {text}")

                    # Create TTS service per turn (recommended)
                    async with CartesiaTTSService(api_key=os.getenv('CARTESIA_API_KEY')) as tts:
                        # Bridge: LLM tokens → TTS text generator → audio
                        async def llm_to_text_chunks():
                            async for token in llm.stream_response(text):
                                yield token

                        # This is the magic line: stream audio as soon as it arrives
                        try:
                            async for audio_chunk in tts.stream_synthesis(llm_to_text_chunks()):
                                await websocket.send_bytes(audio_chunk)

                            # Optional: send signal when assistant is done speaking
                            await websocket.send_json({
                                "status": "assistant_done_speaking",
                                "message": "turn_complete"
                            })

                        except Exception as e:
                            logger.error(f"TTS streaming failed: {e}")
                            await websocket.send_json({
                                "status": "error",
                                "message": "Failed to generate speech"
                            })

            elif rtype == "done":
                logger.info("Cartesia sent 'done'")
                break

            elif rtype == "flush_done":
                logger.info("Flush completed")

            elif rtype == "error":
                error = result.get("message", "Unknown error")
                logger.error(f"Cartesia error: {error}")
                await websocket.send_json({"status": "error", "message": error})

    except Exception as e:
        logger.error(f"Error in transcription consumer: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except:
            pass


@app.get("/record")
async def get_recorder():
    RECORDING_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Cartesia Real-Time STT Demo</title>
      <style>
        body { font-family: system-ui; padding: 2rem; background: #0d1117; color: #c9d1d9; }
        #output { margin-top: 1rem; padding: 1rem; background: #161b22; border-radius: 8px; min-height: 100px; }
        button { padding: 12px 24px; font-size: 1.1rem; margin: 0.5rem; }
        .partial { opacity: 0.7; }
        .final { font-weight: bold; color: #8bffb3; }
      </style>
    </head>
    <body>
      <h1>Cartesia Real-Time STT (Whisper Ink)</h1>
      <button id="start">Start Recording</button>
      <button id="stop" disabled>Stop Recording</button>
      <div id="status">Ready</div>
      <div id="output"></div>

    <script>
    let ws;
    let mediaRecorder;
    let downsamplerNode;
    
    let audioContext = null;
    let nextStartTime = 0;
    
    function initAudioContext() {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 16000
        });
        nextStartTime = audioContext.currentTime;
      }
    }

    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');

    async function startRecording() {
      if (ws) ws.close();

      ws = new WebSocket(`ws://${window.location.host}/ws/conversation/cartesia`);

      ws.onopen = () => {
        statusEl.textContent = "Connected – Speak now!";
        startBtn.disabled = true;
        stopBtn.disabled = false;
      };

      ws.onmessage = async (event) => {
          if (event.data instanceof Blob) {
            initAudioContext();
        
            const arrayBuffer = await event.data.arrayBuffer();
            const int16Array = new Int16Array(arrayBuffer);
        
            // Convert Int16Array → Float32Array (-1.0 to +1.0)
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
              float32Array[i] = int16Array[i] / 32768;
            }
        
            const audioBuffer = audioContext.createBuffer(1, float32Array.length, 16000);
            audioBuffer.getChannelData(0).set(float32Array);
        
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(nextStartTime);
            nextStartTime = Math.max(nextStartTime + audioBuffer.duration, audioContext.currentTime);
          } 
          else {
            const data = JSON.parse(event.data);
            if (data.status === "final") {
              outputEl.innerHTML += `<div class="final">You: ${data.transcription}</div>`;
            }
            if (data.status === "assistant_done") {
              console.log("Assistant finished speaking");
            }
          }
        };

      ws.onclose = () => {
        statusEl.textContent = "Disconnected";
        startBtn.disabled = false;
        stopBtn.disabled = true;
      };

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);

      // Downsample to 16kHz + convert to 16-bit PCM
      const scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
      scriptNode.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const buffer = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          buffer[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
        }
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(buffer.buffer);
        }
      };

      source.connect(scriptNode);
      scriptNode.connect(audioContext.destination);

      downsamplerNode = scriptNode;
    }

    function stopRecording() {
      if (downsamplerNode) downsamplerNode.disconnect();
      if (audioContext) audioContext.close();
      if (ws) ws.close();
    }

    startBtn.onclick = startRecording;
    stopBtn.onclick = stopRecording;
    </script>
    </body>
    </html>
    """
    return responses.HTMLResponse(RECORDING_HTML)


@app.websocket("/ws/conversation/cartesia")
async def websocket_conversation(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected (mic streaming)")

    # Reuse one service instance per connection (or singleton if you prefer)
    # stt = CartesiaSTTService(api_key=os.getenv('CARTESIA_API_KEY'))

    try:
        await stt.connect()
        logger.info("Connected to Cartesia STT")

        # Single background task to consume all transcription events
        consumer_task = asyncio.create_task(
            consume_cartesia_transcriptions(stt, websocket)
        )

        # Main loop: only receive audio from browser
        while True:
            try:
                # Receive raw PCM bytes from browser
                audio_chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                await stt.send_audio(audio_chunk)

            except asyncio.TimeoutError:
                logger.debug("Heartbeat timeout – keeping connection alive")
                try:
                    await websocket.send_json({"status": "heartbeat"})
                except:
                    break
                continue

            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except:
            pass
    finally:
        # Clean shutdown
        if 'consumer_task' in locals():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

        try:
            await stt.finalize()
            await asyncio.sleep(0.3)  # Let flush complete
            await stt.close(send_done=True)  # Make sure to add this param if you kept it
        except:
            pass

        logger.info("WebSocket closed cleanly")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_max_size=16777216  # 16MB limit for WebSocket messages
    )
