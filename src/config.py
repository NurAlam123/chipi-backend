from pathlib import Path

ABS_PATH = Path(__file__).resolve().parent.parent

# LLM CONFIG
LLM_MAX_TOKEN = 32768
LLM_MODEL_NAME = "Qwen/Qwen3-0.6B"
LLM_MODEL_FOLDER_NAME = "qwen3-0.6B"
LLM_MODEL_PATH = ABS_PATH / "models" / LLM_MODEL_FOLDER_NAME

# TTS CONFIG
TTS_MODEL = ABS_PATH / "piper-models/en_US-hfc_female-medium.onnx"
TTS_CONFIG = ABS_PATH / "piper-models/en_US-hfc_female-medium.onnx.json"

# STT CONFIG
SD_SAMPLE_RATE = 44100
SD_CHANNELS = 1

WHISPER_MODEL_PATH = ABS_PATH / "whisper-base-en/"

# DATABASE CONFIG
DATABASE_ABS_PATH = ABS_PATH / "database"
DATABASE_NAME = "conversations"
DATABASE_PORT = "5434"
DATABASE_HOST = "localhost"
DATABASE_USER = ""
DATABASE_PASS = ""
