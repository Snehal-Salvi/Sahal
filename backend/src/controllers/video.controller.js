import multer from "multer";
import { Video } from "../models/Video.js";
import { videoQueue } from "../queues/videoQueue.js";
import { analyzeVideoBufferWithAI, analyzeVideoWithAI } from "../services/ai.service.js";
import { uploadBuffer } from "../services/cloudinary.service.js";

// Tracks Cloudinary uploads still in flight after the response was sent.
// queueVideoProcessing awaits the promise here if the user clicks Process
// before the upload finishes. Entries are deleted on resolve/reject.
const pendingCloudinaryUploads = new Map();

const storage = multer.memoryStorage();
const pngSignature = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
]);

function pngHasAlphaChannel(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 33) {
    return false;
  }

  if (!buffer.subarray(0, 8).equals(pngSignature)) {
    return false;
  }

  const ihdrLength = buffer.readUInt32BE(8);
  const ihdrType = buffer.toString("ascii", 12, 16);

  if (ihdrLength !== 13 || ihdrType !== "IHDR") {
    return false;
  }

  const colorType = buffer[25];
  if (colorType === 4 || colorType === 6) {
    return true;
  }

  let offset = 8;
  while (offset + 12 <= buffer.length) {
    const chunkLength = buffer.readUInt32BE(offset);
    const chunkType = buffer.toString("ascii", offset + 4, offset + 8);

    if (chunkType === "tRNS") {
      return true;
    }

    offset += chunkLength + 12;
  }

  return false;
}

const videoFileFilter = (req, file, cb) => {
  if (!file.mimetype.startsWith("video/")) {
    cb(new Error("Only video files are allowed"));
    return;
  }

  cb(null, true);
};

const overlayFileFilter = (req, file, cb) => {
  if (file.mimetype !== "image/png") {
    cb(new Error("Only transparent PNG overlay images are allowed"));
    return;
  }

  cb(null, true);
};

export const videoUpload = multer({
  storage,
  fileFilter: videoFileFilter,
  limits: {
    fileSize: 250 * 1024 * 1024
  }
});

export const overlayUpload = multer({
  storage,
  fileFilter: overlayFileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024
  }
});

export async function uploadVideo(req, res) {
  if (!req.file) {
    return res.status(400).json({ message: "Video file is required" });
  }

  // Kick off Cloudinary upload in parallel with face analysis. The user only
  // needs the originalUrl when they click Process, so we don't block the
  // response on it — we wait on the promise later if needed.
  const cloudinaryPromise = uploadBuffer(req.file.buffer, {
    folder: "cartoon-face-filter/originals",
    resource_type: "video",
  });

  const analysis = await analyzeVideoBufferWithAI({
    buffer: req.file.buffer,
    filename: req.file.originalname
  });

  const video = await Video.create({
    detectedFaces: (analysis.faces || []).map((face) => ({
      faceId: face.faceId,
      label: face.label,
      embedding: face.embedding || [],
      representativeBox: face.representativeBox || undefined
    })),
    status: "uploaded"
  });

  // Finalize Cloudinary upload in the background and write originalUrl when ready.
  pendingCloudinaryUploads.set(
    String(video._id),
    cloudinaryPromise
      .then(async (uploadResult) => {
        await Video.findByIdAndUpdate(video._id, {
          originalUrl: uploadResult.secure_url,
          originalPublicId: uploadResult.public_id
        });
        return uploadResult;
      })
      .catch(async (err) => {
        await Video.findByIdAndUpdate(video._id, {
          status: "failed",
          error: `Upload failed: ${err.message}`
        });
        throw err;
      })
      .finally(() => {
        pendingCloudinaryUploads.delete(String(video._id));
      })
  );

  return res.status(201).json({
    message: "Video uploaded and analyzed",
    video,
    analysis
  });
}

export async function uploadOverlay(req, res) {
  if (!req.file) {
    return res.status(400).json({ message: "Overlay image file is required" });
  }

  if (!pngHasAlphaChannel(req.file.buffer)) {
    return res.status(400).json({
      message: "Overlay PNG must include transparency (an alpha channel)"
    });
  }

  const uploadResult = await uploadBuffer(req.file.buffer, {
    folder: "cartoon-face-filter/overlays",
    public_id: `overlay-${Date.now()}`,
    resource_type: "image"
  });

  return res.status(201).json({
    message: "Overlay uploaded successfully",
    overlayUrl: uploadResult.secure_url,
    overlayPublicId: uploadResult.public_id
  });
}

export async function queueVideoProcessing(req, res) {
  const { videoId } = req.params;
  const { filterAssignments } = req.body;

  if (!Array.isArray(filterAssignments) || filterAssignments.length === 0) {
    return res.status(400).json({ message: "filterAssignments is required" });
  }

  const video = await Video.findById(videoId);

  if (!video) {
    return res.status(404).json({ message: "Video not found" });
  }

  if (!video.detectedFaces.length) {
    return res.status(400).json({
      message: "Analyze faces before queueing video processing"
    });
  }

  // Cloudinary upload may still be in flight (we returned the upload response
  // before it finished). Wait for it here so the worker has originalUrl.
  if (!video.originalUrl) {
    const pending = pendingCloudinaryUploads.get(String(video._id));
    if (pending) {
      try {
        await pending;
      } catch (err) {
        return res.status(500).json({ message: `Upload failed: ${err.message}` });
      }
      // Refresh from DB to pick up originalUrl set by the upload finalizer
      const refreshed = await Video.findById(videoId);
      if (refreshed) {
        video.originalUrl = refreshed.originalUrl;
        video.originalPublicId = refreshed.originalPublicId;
      }
    }
    if (!video.originalUrl) {
      return res.status(409).json({ message: "Video upload has not completed yet" });
    }
  }

  const knownFaceIds = new Set(video.detectedFaces.map((face) => face.faceId));
  const normalizedAssignments = filterAssignments.map((assignment) => ({
    faceId: assignment?.faceId,
    overlayImageUrl: assignment?.overlayImageUrl
  }));

  const hasInvalidAssignment = normalizedAssignments.some(
    (assignment) =>
      !assignment.faceId ||
      !assignment.overlayImageUrl ||
      !knownFaceIds.has(assignment.faceId)
  );

  if (hasInvalidAssignment) {
    return res.status(400).json({
      message: "Each filter assignment must reference a detected face and uploaded overlay"
    });
  }

  video.overlayImageUrl = normalizedAssignments[0]?.overlayImageUrl;
  video.filterAssignments = normalizedAssignments;
  video.error = undefined;
  await video.save();

  const job = await videoQueue.add(
    "process-video",
    {
      videoId: video.id
    },
    {
      attempts: 3,
      backoff: {
        type: "exponential",
        delay: 5000
      },
      removeOnComplete: 100,
      removeOnFail: 100
    }
  );

  video.status = "queued";
  video.jobId = String(job.id);
  await video.save();

  return res.json({
    message: "Video queued for processing",
    jobId: job.id,
    video
  });
}

export async function analyzeVideoFaces(req, res) {
  const { videoId } = req.params;

  const video = await Video.findById(videoId);
  if (!video) {
    return res.status(404).json({ message: "Video not found" });
  }

  const analysis = await analyzeVideoWithAI({ videoUrl: video.originalUrl });

  video.detectedFaces = (analysis.faces || []).map((face) => ({
    faceId: face.faceId,
    label: face.label,
    embedding: face.embedding || [],
    representativeBox: face.representativeBox || undefined
  }));
  await video.save();

  return res.json({
    message: "Video analyzed successfully",
    video,
    analysis
  });
}

export async function getVideoStatus(req, res) {
  const { videoId } = req.params;
  const video = await Video.findById(videoId);

  if (!video) {
    return res.status(404).json({ message: "Video not found" });
  }

  return res.json({ video });
}
