# client.py
import collections
import asyncio
import websockets
import json
import pyaudio
import wave
import tempfile
import os
from io import BytesIO
import webrtcvad

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Record for 5 seconds


class VADRecorder:
    """Voice Activity Detection based audio recorder"""

    def __init__(
            self,
            sample_rate: int = 16000,
            frame_duration_ms: int = 30,
            vad_aggressiveness: int = 2,
            silence_duration: float = 1.5,
            min_speech_duration: float = 0.5,
            max_recording_duration: float = 10.0
    ):
        """
        Initialize VAD recorder

        Args:
            sample_rate: Audio sample rate (8000, 16000, or 32000)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            vad_aggressiveness: VAD aggressiveness level (0-3, higher = more aggressive)
            silence_duration: Seconds of silence before stopping
            min_speech_duration: Minimum recording duration
            max_recording_duration: Maximum recording duration (safety timeout)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_recording_duration = max_recording_duration

        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_size * 2  # 2 bytes per sample (int16)

        # State tracking
        self.reset()

    def reset(self):
        """Reset state for new recording"""
        self.state = "WAITING_FOR_SPEECH"
        self.speech_frames = 0
        self.silence_frames = 0
        self.total_frames = 0
        self.frames_buffer = collections.deque()

    def process_frame(self, frame: bytes) -> tuple[bool, str]:
        """
        Process audio frame through VAD

        Args:
            frame: Audio frame bytes

        Returns:
            tuple: (should_continue_recording, current_state)
        """
        self.total_frames += 1
        self.frames_buffer.append(frame)

        # Calculate durations
        frame_duration_s = self.frame_duration_ms / 1000.0
        total_duration = self.total_frames * frame_duration_s
        silence_duration_current = self.silence_frames * frame_duration_s
        speech_duration = self.speech_frames * frame_duration_s

        # Safety timeout
        if total_duration >= self.max_recording_duration:
            return False, "MAX_DURATION_REACHED"

        # Check if frame contains speech
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            # If VAD fails, assume it's speech to be safe
            is_speech = True

        # State machine
        if self.state == "WAITING_FOR_SPEECH":
            if is_speech:
                self.state = "RECORDING_SPEECH"
                self.speech_frames += 1
                print("🎤 Speech detected, recording...")

        elif self.state == "RECORDING_SPEECH":
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                self.silence_frames += 1

                # Check if we should stop
                if silence_duration_current >= self.silence_duration:
                    if speech_duration >= self.min_speech_duration:
                        return False, "SILENCE_DETECTED"
                    else:
                        # Not enough speech yet, continue
                        pass

        return True, self.state

    def get_audio_data(self) -> bytes:
        """Get all recorded audio frames as bytes"""
        return b''.join(self.frames_buffer)


class ConversationalAIClient:
    def __init__(self, websocket_url: str = "ws://localhost:8000/ws/conversation"):
        self.websocket_url = websocket_url
        self.audio = pyaudio.PyAudio()
        
    def record_audio(self) -> bytes:
        """Record audio from microphone using VAD and return WAV bytes"""
        print("\n🎤 Listening... Start speaking!")

        # Initialize VAD recorder
        vad_recorder = VADRecorder(
            sample_rate=RATE,
            frame_duration_ms=30,
            vad_aggressiveness=2,
            silence_duration=1.5,
            min_speech_duration=0.5,
            max_recording_duration=10.0
        )
        
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=vad_recorder.frame_size
        )

        recording = True
        frames = []

        try:
            while recording:
                # Read frame
                frame = stream.read(vad_recorder.frame_size, exception_on_overflow=False)
                frames.append(frame)

                # Process through VAD
                should_continue, state = vad_recorder.process_frame(frame)

                if not should_continue:
                    if state == "SILENCE_DETECTED":
                        print("🔇 Silence detected, stopping recording...")
                    elif state == "MAX_DURATION_REACHED":
                        print("⏱️ Maximum duration reached, stopping recording...")
                    recording = False

        except KeyboardInterrupt:
            print("\n⚠️ Recording interrupted by user")
        finally:
            stream.stop_stream()
            stream.close()

        print("✅ Recording finished!")

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
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                print(f"🔗 Connecting to {self.websocket_url}...")

                async with websockets.connect(
                        self.websocket_url,
                        ping_interval=20,  # Send ping every 20 seconds
                        ping_timeout=10,  # Wait 10 seconds for pong
                ) as websocket:
                    print("✅ Connected to server!\n")
                    retry_count = 0  # Reset retry count on successful connection

                    while True:
                        # Ask user if they want to speak
                        user_input = input("Press ENTER to speak (or 'q' to quit): ")
                        if user_input.lower() == 'q':
                            print("👋 Goodbye!")
                            return

                        try:
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
                                    message = await asyncio.wait_for(websocket.recv(), timeout=60.0)

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
                                            print(
                                                f"❌ Error at {status.get('stage', 'unknown')}: {status.get('message', 'Unknown error')}")
                                            break

                                except asyncio.TimeoutError:
                                    print("⏰ Timeout waiting for response")
                                    break

                            # Play audio response if received
                            if audio_response:
                                self.play_audio(audio_response)

                            print("\n" + "=" * 50 + "\n")

                        except websockets.exceptions.ConnectionClosed:
                            print("⚠️ Connection closed, reconnecting...")
                            raise  # Trigger reconnection

            except websockets.exceptions.ConnectionClosed:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 Reconnection attempt {retry_count}/{max_retries}...")
                    await asyncio.sleep(2)
                else:
                    print("❌ Max reconnection attempts reached")
                    raise
            except Exception as e:
                print(f"❌ Connection error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 Reconnection attempt {retry_count}/{max_retries}...")
                    await asyncio.sleep(2)
                else:
                    raise
    
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