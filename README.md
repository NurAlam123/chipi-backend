# Chipi - Backend

[Chipi](https://github.com/nuralam123/chipi.git) is a local chatbot app powered by HuggingFace models and PyTorch. This is the backend API service that powers the chat interface with features like streaming responses, managing conversations, and more.

## 🚀 Tech Stack

- **Python**
- **FastAPI**
- **PyTorch**
- **HuggingFace Transformers**

## 📡 API Endpoints

> All endpoints are prefixed with `/api`

### GET `/api/conversations`

Retrieve a list of all stored conversations.

### POST `/api/conversations/new`

Create a new conversation session.

### GET `/api/conversations/{session_id}`

Fetch the complete conversation history including metadata and messages.

### DELETE `/api/conversations/{session_id}`

Delete a specific conversation session.

### POST `/api/conversations/{session_id}/message`

Send a message to the LLM under a session. The model responds contextually.

### GET `/api/conversations/{session_id}/stream`

Stream the LLM response token-by-token as they’re generated. Useful for real-time UI updates.

### PATCH `/api/conversations/{session_id}/title`

Update the title of a specific conversation.

## Project Structure

```
chipi-backend/
├── database/
│   └── conversations        # PostgreSQL data for conversations
├── datasets/                # Store datasets if needed
├── models/                  # Local models go here
├── piper-models/            # TTS models for Piper (if using TTS)
├── whisper-base-en/         # Whisper model files (if using STT)
├── pyproject.toml
├── README.md
└── src/
    ├── __init__.py
    ├── api.py               # FastAPI entrypoint
    ├── bot.py               # LLM handling logic
    ├── config.py            # Configurations and environment
    ├── db.py                # Database logic
    ├── stt.py               # Whisper STT support
    └── tts.py               # Piper TTS support
```

## 🚀 Run Locally

1. **Edit the configuration**
   Open `config.py` and update the settings as per your environment (e.g., model path, database URL, etc.).

2. **Install all dependencies**

   ```bash
   uv install
   ```

   > Requires [uv](https://github.com/astral-sh/uv). If you don’t have it yet:
   >
   > ```bash
   > pip install uv
   > ```

3. **Start PostgreSQL (if using a local database)**
   Make sure your PostgreSQL service is running and accessible.

4. **Change directory to `src`**

   ```bash
   cd src
   ```

5. **Run the FastAPI server**

   ```bash
   uvicorn api:app --reload
   ```

## 📜 License

[MIT](./LICENSE)
