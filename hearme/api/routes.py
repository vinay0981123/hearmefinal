from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, AnyHttpUrl, Field
from typing import Optional

from hearme.core.audio import fetch_and_prepare_audio
from hearme.core.stt import transcribe_segments
from hearme.core.diarize import diarize_audio_with_embeddings
from hearme.core.align import align_words_to_speakers
from hearme.core.mapping import build_stable_user_map

router = APIRouter()

class TranscribeBody(BaseModel):
    url: AnyHttpUrl = Field(
        default="https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3",
        description="Publicly accessible audio URL"
    )
    language: Optional[str] = Field(
        default="en",
        description="Language code to force decoding (e.g., 'en')."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3",
                "language": "en"
            }
        }

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/transcribe")
async def transcribe(body: TranscribeBody):
    try:
        # 1) Fetch + prepare audio
        wav_path, sr = await fetch_and_prepare_audio(str(body.url))

        # 2) Diarization (auto-infer speakers; removed num/min/max)
        diar, spk_embeds = await diarize_audio_with_embeddings(
            wav_path=wav_path,
            sample_rate=sr,
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
        )

        # 3) ASR with defaulted language "en"
        stt_result = await transcribe_segments(
            wav_path,
            language=body.language,  # defaults to "en"
        )

        # 4) Align + response
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
            "transcript": "\n".join(lines),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
from typing import List
from pydantic import Field, AnyHttpUrl
from hearme.core.batch_runner import process_many  # ensure file exists at hearme/core/batch_runner.py

class BatchBody(BaseModel):
    urls: List[AnyHttpUrl] = Field(
        default=[
            "https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3",
            "https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3"
        ],
        description="List of publicly accessible audio URLs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3",
                    "https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3"
                ]
            }
        }

@router.post("/transcribe_batch")
async def transcribe_batch(body: BatchBody):
    results = await process_many([str(u) for u in body.urls])
    return {"results": results}
