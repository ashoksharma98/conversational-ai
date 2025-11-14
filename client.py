# client.py
import asyncio
import websockets
import json
import pyaudio
import wave
import tempfile
import os
from io import BytesIO

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Record for 5 seconds

class ConversationalAIClient:
    def __init__(self, websocket_url: str = "ws://localhost:8000/ws/conversation"):
        self.websocket_url = websocket_url
        self.audio = pyaudio.PyAudio()
        
    def record_audio(self) -> bytes:
        """Record audio from microphone and return WAV bytes"""
        print("\n🎤 Recording... Speak now!")
        
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("✅ Recording finished!")
        
        stream.stop_stream()
        stream.close()
        
        # Convert to WAV format in memory
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        return wav_buffer.getvalue()
    
    def play_audio(self, audio_data: bytes):
        """Play audio response from bytes using PyAudio"""
        print("\n🔊 Playing response...")
        
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # Convert MP3 to WAV using pydub, then play with PyAudio
            try:
                from pydub import AudioSegment
                
                # Load MP3 and convert to WAV
                audio = AudioSegment.from_mp3(temp_audio_path)
                
                # Export to WAV in memory
                wav_buffer = BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                
                # Play using PyAudio
                with wave.open(wav_buffer, 'rb') as wf:
                    stream = self.audio.open(
                        format=self.audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )
                    
                    # Read and play audio in chunks
                    data = wf.readframes(CHUNK)
                    while data:
                        stream.write(data)
                        data = wf.readframes(CHUNK)
                    
                    stream.stop_stream()
                    stream.close()
                
            except ImportError:
                # Fallback: use pygame for MP3 playback
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(temp_audio_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                pygame.mixer.quit()
            
            # Cleanup
            os.unlink(temp_audio_path)
            print("✅ Playback finished!")
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            print("💡 Tip: Install pydub and ffmpeg, or pygame for audio playback")
    
    async def run_conversation(self):
        """Main conversation loop"""
        print(f"🔗 Connecting to {self.websocket_url}...")
        
        async with websockets.connect(self.websocket_url) as websocket:
            print("✅ Connected to server!\n")
            
            while True:
                # Ask user if they want to speak
                user_input = input("Press ENTER to speak (or 'q' to quit): ")
                if user_input.lower() == 'q':
                    print("👋 Goodbye!")
                    break
                
                # Record audio
                audio_data = self.record_audio()
                print(f"📤 Sending audio data ({len(audio_data)} bytes)...")
                
                # Send audio to server
                await websocket.send(audio_data)
                
                # Receive responses
                audio_response = None
                audio_chunks = []
                is_chunked = False
                num_chunks = 0
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        
                        # Check if it's JSON or binary
                        if isinstance(message, bytes):
                            # This is audio data
                            if is_chunked:
                                audio_chunks.append(message)
                                print(f"📥 Received audio chunk {len(audio_chunks)}/{num_chunks}")

                                # Check if we've received all chunks
                                if len(audio_chunks) == num_chunks:
                                    audio_response = b''.join(audio_chunks)
                                    print(f"✅ All audio chunks received ({len(audio_response)} bytes)")
                            else:
                                audio_response = message
                                print(f"📥 Received audio response ({len(audio_response)} bytes)")
                            
                        else:
                            # This is a JSON status message
                            status = json.loads(message)
                            
                            # Ignore ping messages
                            if status.get("type") == "ping":
                                continue
                            
                            if status["status"] == "transcribed":
                                print(f"📝 You said: {status['transcription']}")
                                
                            elif status["status"] == "response_generated":
                                print(f"💬 AI response: {status['response']}")
                                
                            elif status["status"] == "processing":
                                stage = status.get("stage", "")
                                if not status.get("keepalive"):  # Don't spam keepalive messages
                                    print(f"⏳ Processing: {stage}")
                                
                            elif status["status"] == "audio_ready":
                                audio_size = status['audio_size']
                                is_chunked = status.get('chunked', False)
                                num_chunks = status.get('num_chunks', 0)

                                if is_chunked:
                                    print(f"🎵 Audio ready (size: {audio_size} bytes, {num_chunks} chunks)")
                                    audio_chunks = []  # Reset chunks list
                                else:
                                    print(f"🎵 Audio ready (size: {audio_size} bytes)")
                                
                            elif status["status"] == "completed":
                                print("✅ Conversation turn completed!")
                                break
                                
                            elif status["status"] == "error":
                                print(f"❌ Error at {status.get('stage', 'unknown')}: {status.get('message', 'Unknown error')}")
                                break
                    
                    except asyncio.TimeoutError:
                        print("⏰ Timeout waiting for response")
                        break
                    except Exception as e:
                        print(f"❌ Error receiving message: {e}")
                        break
                
                # Play audio response if received
                if audio_response:
                    self.play_audio(audio_response)
                
                print("\n" + "="*50 + "\n")
    
    def cleanup(self):
        """Cleanup resources"""
        self.audio.terminate()

async def main():
    client = ConversationalAIClient()
    
    try:
        await client.run_conversation()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())