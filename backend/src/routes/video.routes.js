import { Router } from "express";
import {
  analyzeUploadedVideoFaces,
  analyzeVideoFaces,
  getVideoStatus,
  overlayUpload,
  queueVideoProcessing,
  uploadOverlay,
  uploadVideo,
  videoUpload
} from "../controllers/video.controller.js";
import { requireAuth } from "../middleware/requireAuth.js";

const router = Router();
const asyncHandler = (handler) => (req, res, next) =>
  Promise.resolve(handler(req, res, next)).catch(next);

router.use(requireAuth);

router.post("/upload", videoUpload.single("video"), asyncHandler(uploadVideo));
router.post(
  "/upload-overlay",
  overlayUpload.single("overlay"),
  asyncHandler(uploadOverlay)
);
router.post(
  "/:videoId/analyze-upload",
  videoUpload.single("video"),
  asyncHandler(analyzeUploadedVideoFaces)
);
router.post("/:videoId/analyze", asyncHandler(analyzeVideoFaces));
router.post("/:videoId/process", asyncHandler(queueVideoProcessing));
router.get("/:videoId/status", asyncHandler(getVideoStatus));

export default router;
