from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import multiprocessing as mp
from queue import Queue
import uuid
from typing import Dict

import subprocess


from bot import Bot
from db import ConversationDB
from config import (
    DATABASE_ABS_PATH,
    DATABASE_HOST,
    DATABASE_NAME,
    DATABASE_PASS,
    DATABASE_PORT,
    DATABASE_USER,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database Connection
    # Run the pg server if it's not running
    try:
        print("INITIALIZING POSTGRESQL SERVER...")
        subprocess.run(
            [
                "pg_ctl",
                "-D",
                str(DATABASE_ABS_PATH / DATABASE_NAME),
                "-l",
                str(DATABASE_ABS_PATH / "logfile"),
                "start",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print(
            f"POSTGRESQL SERVER IS RUNNING IN: {DATABASE_PORT}\nDATABASE: {DATABASE_NAME}"
        )
    except subprocess.CalledProcessError as e:
        if "another server might be running" in e.stderr.lower():
            print("ANOTHER SERVER IS RUNNING ON PORT", DATABASE_PORT)
        else:
            print("SERVER FAILED TO START\nERROR::", e.stderr)
            exit()
    app.state.db = ConversationDB(app.state.db_config)
    yield

    # Cleanup
    bot.unload_model()
    app.state.db.close()


app = FastAPI(lifespan=lifespan)
origins = ["http://localhost:5173", "http://192.168.0.105:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
app.state.db_config = {
    "host": DATABASE_HOST,
    "user": DATABASE_USER,
    "password": DATABASE_PASS,
    "database": DATABASE_NAME,
    "port": DATABASE_PORT,
}

# Initialize bot with database
bot = Bot(app.state.db_config)


def worker_process(
    session_id: str, prompt: str, thinking: bool, output_queue: mp.Queue
):
    try:
        # Load model if not loaded
        if not bot.model or not bot.tokenizer:
            bot.load_model()

        # Set the current session and prompt
        bot.session_id = session_id
        bot.prompt(prompt)

        # Create a queue for streaming
        stream_queue = Queue()

        # Generate stream
        for chunk in bot.generate_stream(stream_queue, thinking):
            output_queue.put(chunk)
        output_queue.put(None)  # Signal done
    except Exception as e:
        output_queue.put(f"data: ERROR: {str(e)}\n\n")
        output_queue.put(None)
        raise


@app.get("/api/ping")
async def ping():
    return {"response": "pong"}


@app.get("/api/conversations")
async def get_conversations(limit: int = 20, offset: int = 0):
    """Get list of all conversations with metadata"""
    try:
        conversations = app.state.db.list_conversations(limit=limit, offset=offset)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations/new")
async def new_conversation(title: str | None = None) -> Dict:
    """Create a new conversation session"""
    if not title:
        title = "New Conversation"
    try:
        session_id = app.state.db.create_conversation(initial_prompt=title)
        return {"session_id": session_id, "message": "New conversation created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str) -> Dict:
    """Get complete conversation with metadata and messages"""
    try:
        # Validate UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        conversation, messages = app.state.db.get_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"conversation": conversation, "messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/conversations/{session_id}/title")
async def update_conversation_title(session_id: str, title: str) -> Dict:
    """Update conversation title"""
    try:
        # Validate UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        success = app.state.db.update_conversation_title(session_id, title)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "session_id": session_id,
            "title": title,
            "message": "Title updated successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str) -> Dict:
    """Delete a conversation and all its messages"""
    try:
        # Validate UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        success = app.state.db.delete_conversation(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "session_id": session_id,
            "message": "Conversation deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations/{session_id}/message")
async def add_message(session_id: str, role: str, content: str) -> Dict:
    """Add a message to a conversation"""
    try:
        # Validate UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        if role not in ["user", "assistant", "system"]:
            raise HTTPException(status_code=400, detail="Invalid role specified")

        success = app.state.db.add_message(session_id, role, content)
        if not success:
            raise HTTPException(status_code=404, detail="Failed to add message")

        return {"session_id": session_id, "message": "Message added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{session_id}/stream")
async def stream_response(session_id: str, prompt: str, thinking: bool):
    """Stream LLM response for a given prompt in a conversation"""
    try:
        # Validate UUID format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session ID format")

        # Use multiprocessing Manager for better queue handling
        manager = mp.Manager()
        output_queue = manager.Queue()

        # Start the worker process
        p = mp.Process(
            target=worker_process,
            args=(session_id, prompt, thinking, output_queue),
            daemon=True,
        )
        p.start()

        # Headers for SSE
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }

        async def event_generator():
            try:
                while True:
                    if not p.is_alive() and output_queue.empty():
                        break

                    try:
                        chunk = output_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        response = f"{chunk}\\n\\n\n\n"
                        yield response
                    except Exception:  # Queue.Empty or others
                        continue

                p.join()  # Clean up the process
            finally:
                if p.is_alive():
                    p.terminate()
                manager.shutdown()

        return StreamingResponse(
            event_generator(), media_type="text/event-stream", headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
