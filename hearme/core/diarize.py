import torch
from pyannote.audio import Pipeline
from hearme.core.config import settings

_pipeline_cache = None

def _gpu_enabled():
    if settings.DEVICE == "cuda" and torch.cuda.is_available():
        return True
    if settings.DEVICE == "auto" and torch.cuda.is_available():
        return True
    return False

def _get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HF_TOKEN
        )
        device = torch.device("cuda") if _gpu_enabled() else torch.device("cpu")
        _pipeline_cache = _pipeline_cache.to(device)
        print(f"[DIAR] device={device}, param_device={next(_pipeline_cache.model.parameters()).device}")

    return _pipeline_cache

async def diarize_audio_with_embeddings(
    wav_path: str,
    sample_rate: int,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None
):
    pipeline = _get_pipeline()
    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers
    if min_speakers is not None and max_speakers is not None:
        params["min_speakers"] = min_speakers
        params["max_speakers"] = max_speakers
    diar = pipeline(wav_path, **params)

    spk_embeds = {}
    for turn, _, speaker in diar.itertracks(yield_label=True):
        if speaker not in spk_embeds:
            spk_embeds[speaker] = []
    return diar, spk_embeds
