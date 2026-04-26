# Cartoon Face Filter

This repo contains a three-service video pipeline:

- `frontend`: React app for video upload, multi-face preview, filter assignment, and status polling
- `backend`: Express API, MongoDB metadata, Cloudinary storage, BullMQ queue
- `ai-service`: FastAPI microservice that analyzes faces, tracks identities, extracts expression coefficients, mesh-warps the cartoon face, and rebuilds the video with original audio

## Step-by-Step Backend Setup

1. Start infrastructure:

```bash
docker compose up -d mongo redis
```

2. Create Cloudinary credentials and copy backend env:

```bash
cp backend/.env.example backend/.env
```

3. Fill these values in `backend/.env`:

- `MONGODB_URI`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `REDIS_HOST`
- `REDIS_PORT`
- `AI_SERVICE_URL`
- `FRONTEND_URL`

4. Install backend packages:

```bash
cd backend
npm install
```

5. Start the Express API:

```bash
npm run dev
```

6. In a second terminal, start the BullMQ worker:

```bash
cd backend
npm run worker
```

7. Set up the AI service:

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Install `ffmpeg` on the machine running the AI service before starting it.

Optional AI-service tuning in `ai-service/.env`:

- `CARTOON_BLINK_RATIO_MIN` and `CARTOON_BLINK_RATIO_MAX`: lower values make blink detection trigger earlier
- `CARTOON_MOUTH_OPEN_MIN` and `CARTOON_MOUTH_OPEN_MAX`: controls how easily speech/open mouth animates
- `CARTOON_SMILE_WIDTH_MIN`, `CARTOON_SMILE_WIDTH_MAX`, `CARTOON_SMILE_LIFT_MIN`, `CARTOON_SMILE_LIFT_MAX`: tune smile sensitivity
- `CARTOON_BROW_RAISE_MIN` and `CARTOON_BROW_RAISE_MAX`: tune eyebrow motion sensitivity
- `CARTOON_EXPRESSION_SMOOTHING`: higher values react faster, lower values look steadier
- `CARTOON_MAX_FACES`: cap concurrent tracked faces

8. Start the frontend:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

## Simpler Startup

If you want one command instead of four terminals, use the repo root scripts:

1. Install the root helper dependency:

```bash
cd /Users/snehalashoksalvi/Documents/Codes/Filter
npm install
```

2. Start MongoDB and Redis:

```bash
npm run infra:up
```

3. Start the frontend, backend API, backend worker, and AI service together:

```bash
npm run dev
```

This keeps everything in one terminal and prefixes logs by service name.

## API Endpoints

### `POST /api/videos/upload`

Multipart upload endpoint.

- Form field: `video`
- Response: uploaded video document with Cloudinary URL

### `POST /api/videos/upload-overlay`

Multipart upload endpoint for the face overlay image.

- Form field: `overlay`
- Accepts: transparent PNG
- Response: uploaded overlay URL

### `POST /api/videos/:videoId/analyze`

Runs multi-face analysis on the uploaded video.

- Detects faces across sampled frames
- Assigns stable face IDs such as `face-1`, `face-2`
- Returns a representative frame plus selectable face thumbnails

### `POST /api/videos/:videoId/process`

Adds the job to BullMQ.

Request body:

```json
{
  "filterAssignments": [
    {
      "faceId": "face-1",
      "overlayImageUrl": "https://example.com/cartoon-mask-a.png"
    },
    {
      "faceId": "face-2",
      "overlayImageUrl": "https://example.com/cartoon-mask-b.png"
    }
  ]
}
```

### `GET /api/videos/:videoId/status`

Returns the current processing state.

Example response:

```json
{
  "video": {
    "_id": "6626d2...",
    "status": "completed",
    "originalUrl": "https://res.cloudinary.com/...",
    "processedUrl": "https://res.cloudinary.com/...",
    "overlayImageUrl": "https://...",
    "jobId": "12"
  }
}
```

## How the MVP Works

1. React uploads the raw video to the backend.
2. Express uploads the video buffer to Cloudinary and stores metadata in MongoDB.
3. React asks the AI service to analyze the video and returns labeled face thumbnails with stable face IDs.
4. The user assigns one uploaded PNG filter per detected face, or toggles a single filter for all faces.
5. BullMQ queues the video job in Redis with the stored face profiles and filter assignments.
6. The worker calls the FastAPI service with the Cloudinary video URL, face profiles, and per-face filter assignments.
7. FastAPI tracks the same faces across frames, extracts blink/smile/mouth/brow coefficients, mesh-warps each chosen filter, and rebuilds the video with FFmpeg while keeping original audio.
8. The worker uploads the processed video back to Cloudinary and updates the job status in MongoDB.

## Scaling and Optimization Suggestions

- Split API and worker into separate containers so uploads do not compete with processing jobs.
- Move from a single Redis queue to named queues by plan, region, or video length.
- Add signed upload support from React to Cloudinary to avoid sending large videos through Express.
- Store webhook or socket updates instead of polling for high job volume.
- Use FFmpeg GPU acceleration when available for encode-heavy workloads.
- Cache overlay assets locally in the AI service for repeated filters.
- Batch frame reads and use multiprocessing for longer videos.
- Add per-frame face smoothing and Kalman filtering to reduce jitter.
- Upgrade the coefficient extraction from heuristic landmark ratios to a dedicated face blendshape model if you want richer phoneme or emotion control.
