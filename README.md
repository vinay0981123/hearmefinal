Install system deps:

Ubuntu: 
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    wget llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev ffmpeg

Python:

pyenv global 3.10.12 or system Python 3.10.12 #currently this is working on 3.12 also so no need to change version.

python -m venv .venv && source .venv/bin/activate

pip install -U pip

pip install -r requirements.txt

Hugging Face:

Get HF token and accept pyannote community pipeline terms, add to .env as HF_TOKEN.​

Run:

bash uvicorn hearme.app:app --host 0.0.0.0 --port 8000 --reload --log-level info --proxy-headers --forwarded-allow-ips="*"

Test:

curl -X POST http://localhost:8000/api/v1/transcribe   -H "Content-Type: application/json"   -d '{"url":"https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3"}'

Response:

JSON includes segments with speaker labels, stable User mapping, and a human‑readable transcript lines “User N said: …”.​

Usage example
Request:

POST /api/v1/transcribe with body: { "url": "https://betcha.s3.us-east-2.amazonaws.com/audio-4.mp3" }​

Response:

transcript: plain text lines like “User 1 said: …” joined by newlines; mapping ensures same voice remains same User ID across the file.​

Notes and options
Speaker count: You can pass num_speakers, or min_speakers/max_speakers for better control when audio is known. Leaving blank lets the pipeline infer.​

GPU acceleration: enabled automatically if CUDA is present; faster-whisper compute_type is float16 on GPU and int8 on CPU for speed.​

Model choices: set WHISPER_MODEL to tiny/base/small/medium/large-v3; large-v3 recommended for best accuracy. English-only variants (.en) are faster if language is known.​

Docker: You can wrap this app similarly to common FastAPI + HF examples if deploying to spaces, ECR, or other platforms.​