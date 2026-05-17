import hmac
import os
import shutil
import tempfile
from typing import List

import sentry_sdk
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from starlette.background import BackgroundTask

from .processor import analyze_video, analyze_video_from_path, cleanup_file, process_video


SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=os.environ.get("ENVIRONMENT", "development"),
        traces_sample_rate=0.0,
        send_default_pii=False,
    )

app = FastAPI(title="Cartoon Face Filter AI Service")

EXPECTED_INTERNAL_AUTH = os.environ.get("AI_SERVICE_SECRET", "")


def require_internal_auth(x_internal_auth: str = Header(default="")) -> None:
    if not EXPECTED_INTERNAL_AUTH:
        # If no secret is configured, fail closed in production-style envs
        # so a misconfigured deploy can't accidentally accept anonymous traffic.
        raise HTTPException(status_code=503, detail="AI service not configured")
    if not hmac.compare_digest(x_internal_auth, EXPECTED_INTERNAL_AUTH):
        raise HTTPException(status_code=401, detail="Unauthorized")


class FilterAssignment(BaseModel):
    faceId: str
    overlayImageUrl: HttpUrl


class ProcessVideoRequest(BaseModel):
    videoUrl: HttpUrl
    detectedFaces: List[dict] = []
    filterAssignments: List[FilterAssignment]


class AnalyzeVideoRequest(BaseModel):
    videoUrl: HttpUrl


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", dependencies=[Depends(require_internal_auth)])
def analyze_endpoint(payload: AnalyzeVideoRequest):
    try:
        return analyze_video(video_url=str(payload.videoUrl))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/analyze-upload", dependencies=[Depends(require_internal_auth)])
async def analyze_upload_endpoint(video: UploadFile = File(...)):
    tmpdir = tempfile.mkdtemp()
    try:
        suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
        input_path = os.path.join(tmpdir, f"analysis-input{suffix}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        return analyze_video_from_path(input_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/process", dependencies=[Depends(require_internal_auth)])
def process_endpoint(payload: ProcessVideoRequest):
    try:
        output_path = process_video(
            video_url=str(payload.videoUrl),
            detected_faces=payload.detectedFaces,
            filter_assignments=[a.model_dump(mode="json") for a in payload.filterAssignments],
        )
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="processed.mp4",
            background=BackgroundTask(cleanup_file, output_path)
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
