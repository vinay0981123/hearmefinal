# hearme/core/batch_runner.py
import asyncio
from typing import List, Dict

from hearme.core.audio import fetch_and_prepare_audio
from hearme.core.stt import transcribe_segments
from hearme.core.diarize import diarize_audio_with_embeddings
from hearme.core.align import align_words_to_speakers
from hearme.core.mapping import build_stable_user_map

# Concurrency cap: start at 2–4; raise if VRAM and CPU allow
SEM = asyncio.Semaphore(4)

async def process_one(url: str) -> Dict:
    async with SEM:
        wav_path, sr = await fetch_and_prepare_audio(url)
        # Diarization (kept as in your single-file endpoint; remove if ASR-only)
        diar, spk_embeds = await diarize_audio_with_embeddings(
            wav_path=wav_path,
            sample_rate=sr,
            num_speakers=None, min_speakers=None, max_speakers=None
        )
        stt = await transcribe_segments(wav_path, language="en")
        items, speaker_turns = align_words_to_speakers(stt, diar)
        user_map = build_stable_user_map(speaker_turns, spk_embeds)
        transcript = "\n".join(
            f"User {user_map[t['speaker']]} said: " + " ".join(w["word"] for w in t["words"])
            for t in speaker_turns
        )
        return {
            "url": url,
            "segments": speaker_turns,
            "mapping": user_map,
            "transcript": transcript
        }

async def process_many(urls: List[str]) -> List[Dict]:
    tasks = [asyncio.create_task(process_one(u)) for u in urls]
    return await asyncio.gather(*tasks, return_exceptions=False)
