import aiofiles
import asyncio
import os
import tempfile
import requests
import soundfile as sf
import librosa
import numpy as np

TARGET_SR = 16000

async def fetch_and_prepare_audio(url: str):
    suffix = ".mp3" if url.lower().endswith(".mp3") else ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(r.content)
    y, sr = librosa.load(tmp_path, sr=None, mono=True)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    wav_path = tmp_path if tmp_path.endswith(".wav") else tmp_path.replace(".mp3", ".wav")
    if not wav_path.endswith(".wav"):
        wav_path = tmp_path + ".wav"
    sf.write(wav_path, y.astype(np.float32), sr)
    if tmp_path != wav_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except:
            pass
    return wav_path, sr
