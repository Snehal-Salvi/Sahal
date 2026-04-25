from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from starlette.background import BackgroundTask

from .processor import analyze_video, cleanup_file, process_video


app = FastAPI(title="Cartoon Face Filter AI Service")


class ProcessVideoRequest(BaseModel):
    videoUrl: HttpUrl
    detectedFaces: List[dict] = []
    filterAssignments: List[dict]


class AnalyzeVideoRequest(BaseModel):
    videoUrl: HttpUrl


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_endpoint(payload: AnalyzeVideoRequest):
    try:
        return analyze_video(video_url=str(payload.videoUrl))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/process")
def process_endpoint(payload: ProcessVideoRequest):
    try:
        output_path = process_video(
            video_url=str(payload.videoUrl),
            detected_faces=payload.detectedFaces,
            filter_assignments=payload.filterAssignments,
        )
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="processed.mp4",
            background=BackgroundTask(cleanup_file, output_path)
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
