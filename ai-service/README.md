---
title: Cartoon Face Filter AI
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Cartoon Face Filter — AI Service

FastAPI microservice for the Cartoon Face Filter app. Runs MediaPipe face mesh
and rebuilds video frames with overlays applied. Called by the project's Node
backend; not intended to be called directly.

## Endpoints

- `POST /analyze` — sample frames from a video URL, return face IDs + thumbnails.
- `POST /analyze-upload` — same, but accepts a multipart video upload.
- `POST /process` — apply per-face filter overlays and return the rendered MP4.

All endpoints require the `X-Internal-Auth` header to match `AI_SERVICE_SECRET`.
