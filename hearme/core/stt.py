import torch
from faster_whisper import WhisperModel
from hearme.core.config import settings

_model_cache = None

def _get_model():
    global _model_cache
    if _model_cache is None:
        # Force CPU for now
        device = "cpu"
        compute_type = "int8"  # good speed/accuracy tradeoff on CPU
        if settings.DEVICE.lower() == "cuda":
            # only allow if explicitly set to cuda
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        _model_cache = WhisperModel(settings.WHISPER_MODEL, device=device, compute_type=compute_type)
    return _model_cache

async def transcribe_segments(wav_path: str, language: str | None = None):
    model = _get_model()
    segments, info = model.transcribe(
        wav_path,
        vad_filter=True,
        word_timestamps=True,
        language=language,
        beam_size=5
    )
    out_segments = []
    for seg in segments:
        words = []
        if seg.words is not None:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end)
                })
        out_segments.append({
            "text": seg.text.strip(),
            "start": float(seg.start),
            "end": float(seg.end),
            "words": words
        })
    return {
        "language": info.language,
        "duration": info.duration,
        "segments": out_segments
    }
