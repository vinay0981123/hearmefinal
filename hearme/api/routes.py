from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, AnyHttpUrl
from hearme.core.config import settings
from hearme.core.audio import fetch_and_prepare_audio
from hearme.core.stt import transcribe_segments
from hearme.core.diarize import diarize_audio_with_embeddings
from hearme.core.align import align_words_to_speakers
from hearme.core.mapping import build_stable_user_map


router = APIRouter()

class TranscribeBody(BaseModel):
    url: AnyHttpUrl
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    language: str | None = None

@router.get("/health")
async def health():
    return {"status": "ok"}


import logging
logger = logging.getLogger("hearme")


@router.post("/transcribe")
async def transcribe(body: TranscribeBody):
    try:
        wav_path, sr = await fetch_and_prepare_audio(str(body.url))
        diar, spk_embeds = await diarize_audio_with_embeddings(
            wav_path,
            sr,
            num_speakers=body.num_speakers,
            min_speakers=body.min_speakers,
            max_speakers=body.max_speakers
        )
        stt_result = await transcribe_segments(
            wav_path,
            language=body.language
        )
        items, speaker_turns = align_words_to_speakers(stt_result, diar)
        user_map = build_stable_user_map(speaker_turns, spk_embeds)
        lines = []
        for turn in speaker_turns:
            uid = user_map[turn["speaker"]]
            text = " ".join(w["word"] for w in turn["words"])
            lines.append(f"User {uid} said: {text}".strip())
        return {
            "segments": speaker_turns,
            "mapping": user_map,
            "transcript": "\n".join(lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
