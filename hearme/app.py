# hearme/app.py
from fastapi import FastAPI
from hearme.api.routes import router

app = FastAPI(title="HearMe STT + Diarization", debug=True)
app.include_router(router, prefix="/api/v1")
