import { Router } from "express";
import {
  analyzeVideoFaces,
  getVideoStatus,
  overlayUpload,
  queueVideoProcessing,
  uploadOverlay,
  uploadVideo,
  videoUpload
} from "../controllers/video.controller.js";

const router = Router();
const asyncHandler = (handler) => (req, res, next) =>
  Promise.resolve(handler(req, res, next)).catch(next);

router.post("/upload", videoUpload.single("video"), asyncHandler(uploadVideo));
router.post(
  "/upload-overlay",
  overlayUpload.single("overlay"),
  asyncHandler(uploadOverlay)
);
router.post("/:videoId/analyze", asyncHandler(analyzeVideoFaces));
router.post("/:videoId/process", asyncHandler(queueVideoProcessing));
router.get("/:videoId/status", asyncHandler(getVideoStatus));

export default router;
