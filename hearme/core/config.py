import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
    DEVICE_RAW: str = os.getenv("DEVICE", "auto")
    DEVICE: str = DEVICE_RAW.lower()
    # Optional: allow forcing CPU for CTranslate2 via env
    CT2_FORCE_CPU: str | None = os.getenv("CT2_FORCE_CPU")

settings = Settings()
