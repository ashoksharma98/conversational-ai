# Conversational AI – Voice-to-Voice Pipeline

This project is an **end-to-end conversational AI system** that takes **user speech as input**, processes it through an **LLM**, and responds back in **AI-generated speech**.  
Everything is built using **open-source components** except the LLM layer.

The goal is to create a **low-latency, real-time, voice-based assistant** that works over a WebSocket connection.

## 🚀 Features

- **Full Voice-to-Voice Conversation**
  - Speech → Text → LLM → Text → Speech
- **Open-Source STT & TTS**
  - **STT:** Whisper  
  - **TTS:** Edge TTS
- **Python Client for Local Testing**
- **WebSocket-Based Architecture**
- **Modular Pipeline**

## 🧩 Project Structure

```
conversational-ai/
│
├── server.py         # FastAPI WebSocket server
├── client.py         # Local client for testing
├── stt/              # Speech-to-text logic
├── tts/              # Text-to-speech logic
├── llm/              # Language model layer
├── utils/            # Helpers & common utilities
└── requirements.txt
```

## 📦 Setup & Installation

### 1. Clone the repository
git clone https://github.com/ashoksharma98/conversational-ai
cd conversational-ai

### 2. Switch to ongoing-work branch
git checkout ongoing-work

### 3. Install dependencies
pip install -r requirements.txt

### 4. Install ffmpeg
sudo apt install ffmpeg

## ▶️ How to Run

### Start server
python server.py

### Run client
python client.py

Press Enter and start speaking.

## ⚙️ Current Limitations

- Latency (~15s) due to CPU processing
- No native streaming in STT/TTS
- XTTS too heavy on CPU

## 🔧 Work in Progress

- Streaming TTS
- Possible streaming STT
- Better context handling
- GPU integration
- Language auto-detection
