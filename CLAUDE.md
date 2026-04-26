# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A three-service video pipeline called **Cartoon Face Filter**:

- **frontend** — React 18 + Vite app (no router, single component: `VideoUploader`)
- **backend** — Express API (`src/server.js`) + a separate BullMQ worker process (`src/worker.js`)
- **ai-service** — FastAPI microservice using MediaPipe + OpenCV for face analysis and FFmpeg for video rebuilding

## Running the Project

### Prerequisites

- Docker (for MongoDB and Redis)
- Node.js
- Python 3 with a virtual env at `ai-service/.venv`
- `ffmpeg` on the machine running the AI service

### Startup (four processes)

```bash
# 1. Start infrastructure
docker compose up -d mongo redis

# 2. All four services via concurrently (from repo root)
npm run dev
```

The root `npm run dev` starts: frontend (Vite, port 5173), backend API (port 5001), backend BullMQ worker, and AI service (uvicorn, port 8000).

### Individual services

```bash
# Backend API only
cd backend && npm run start       # production
cd backend && npm run dev         # watch mode

# Backend worker only
cd backend && npm run worker:start
cd backend && npm run worker      # watch mode

# AI service
cd ai-service
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
```

### First-time setup

```bash
# Root deps (concurrently)
npm install

# Backend
cp backend/.env.example backend/.env  # then fill in values
cd backend && npm install

# Frontend
cp frontend/.env.example frontend/.env
cd frontend && npm install

# AI service
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Architecture

### Request flow

1. React (`VideoUploader.jsx`) uploads raw video → Express → Cloudinary; metadata saved to MongoDB.
2. React calls `/api/videos/:id/analyze` → Express calls AI service `/analyze` → AI service samples 8 frames, runs MediaPipe face mesh, clusters by landmark descriptors to assign stable IDs (`face-1`, `face-2`, …), returns thumbnails and a representative frame.
3. User assigns one transparent PNG filter per detected face in the UI.
4. React calls `/api/videos/:id/process` → Express enqueues job in BullMQ (Redis).
5. Worker (`video.worker.js`) dequeues job, calls AI service `/process` with Cloudinary video URL + detected faces + filter assignments.
6. AI service: downloads video, runs MediaPipe per frame, extracts blink/smile/mouth/brow coefficients, mesh-warps each filter PNG onto each face, reassembles frames with FFmpeg + original audio, returns MP4 binary.
7. Worker uploads processed MP4 to Cloudinary, updates `Video.status` in MongoDB.
8. React polls `/api/videos/:id/status` every 4 s until `completed` or `failed`.

### Backend layout

```
backend/src/
  app.js          — Express app (CORS, routes, global error handler)
  server.js       — HTTP server entry point
  worker.js       — BullMQ worker entry point (separate process)
  config/         — DB connection, Cloudinary init
  controllers/    — video.controller.js (upload, analyze, process, status)
  models/         — Video.js (Mongoose schema)
  queues/         — videoQueue.js (BullMQ Queue + QueueEvents + IORedis)
  routes/         — video.routes.js
  services/       — ai.service.js (axios calls to FastAPI), cloudinary.service.js
  workers/        — video.worker.js (BullMQ Worker, concurrency 2)
  utils/
```

### AI service layout

```
ai-service/app/
  main.py       — FastAPI app with /analyze and /process endpoints
  processor.py  — MediaPipe face mesh, landmark math, mesh-warp, FFmpeg rebuild
```

Key constants in `processor.py`: `ANALYSIS_SAMPLE_FRAMES = 8`, `CANONICAL_CANVAS_SIZE = 512`, `MAX_ANALYSIS_FRAME_WIDTH = 960`.

### MongoDB Video model statuses

`uploaded` → `queued` → `processing` → `completed` | `failed`

### Environment variables (backend)

| Variable | Purpose |
|---|---|
| `PORT` | Express port (default 5001) |
| `MONGODB_URI` | MongoDB connection string |
| `REDIS_HOST` / `REDIS_PORT` | Redis for BullMQ |
| `AI_SERVICE_URL` | Base URL for the FastAPI service |
| `CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET` | Cloudinary credentials |
| `FRONTEND_URL` | Comma-separated allowed CORS origins |

### AI service tuning (ai-service/.env)

Coefficients controlling expression sensitivity: `CARTOON_BLINK_RATIO_MIN/MAX`, `CARTOON_MOUTH_OPEN_MIN/MAX`, `CARTOON_SMILE_*`, `CARTOON_BROW_RAISE_*`, `CARTOON_EXPRESSION_SMOOTHING`, `CARTOON_MAX_FACES`.

## Key Implementation Notes

- The backend and worker are **two separate Node processes** that share the same BullMQ queue in Redis. Both need to be running for end-to-end processing.
- `ai.service.js` uses `responseType: "arraybuffer"` and `timeout: 0` when calling `/process` because video processing can take a long time and returns binary MP4.
- The frontend validates PNG alpha channels by reading the IHDR `colorType` byte and scanning for `tRNS` chunks before sending a filter to the server.
- Face identity across frames is tracked via cosine similarity of MediaPipe descriptor keypoints, not deep embeddings.
- Processed video is streamed back from the AI service as a file response; the AI service deletes its temp file after the response is sent (`BackgroundTask(cleanup_file, ...)`).
