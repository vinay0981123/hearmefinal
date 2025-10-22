import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
    DEVICE: str = os.getenv("DEVICE", "auto")
    NUM_SPEAKERS = os.getenv("NUM_SPEAKERS")
    MIN_SPEAKERS = os.getenv("MIN_SPEAKERS")
    MAX_SPEAKERS = os.getenv("MAX_SPEAKERS")

settings = Settings()
