import os
import torch
from faster_whisper import WhisperModel
from hearme.core.config import settings

_model_cache = None

def _get_device_and_compute():
    # If user explicitly sets cuda and it's available, use it
    if settings.DEVICE == "cuda" and torch.cuda.is_available():
        # ensure CT2 doesn't force CPU
        return "cuda", "float16"
    # If auto: prefer cuda when available
    if settings.DEVICE == "auto" and torch.cuda.is_available() and not os.getenv("CT2_FORCE_CPU"):
        return "cuda", "float16"
    # Otherwise CPU
    return "cpu", "int8"

def _apply_ct2_env(device: str):
    # CTranslate2 obeys CT2_FORCE_CPU=1; set it only when on CPU
    if device == "cpu":
        os.environ.setdefault("CT2_FORCE_CPU", "1")
    else:
        # Unset if present so GPU is used
        os.environ.pop("CT2_FORCE_CPU", None)

def _get_model():
    global _model_cache
    if _model_cache is None:
        device, compute_type = _get_device_and_compute()
        _apply_ct2_env(device)
        print(f"[STT] WHISPER_MODEL={settings.WHISPER_MODEL}, device={device}, compute_type={compute_type}")
        _model_cache = WhisperModel(
            settings.WHISPER_MODEL,
            device=device,
            compute_type=compute_type
        )
        print(f"[STT] model ready name={settings.WHISPER_MODEL}")
    return _model_cache

async def transcribe_segments(wav_path: str, language: str | None = None):
    model = _get_model()
    segments, info = model.transcribe(
        wav_path,
        vad_filter=False,
        word_timestamps=True,
        language="en",
        beam_size=1,
        temperature=0, 
        chunk_length=30,
    )
    out_segments = []
    for seg in segments:
        words = []
        if seg.words is not None:
            for w in seg.words:
                words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        out_segments.append({
            "text": seg.text.strip(),
            "start": float(seg.start),
            "end": float(seg.end),
            "words": words
        })
    return {"language": info.language, "duration": info.duration, "segments": out_segments}
