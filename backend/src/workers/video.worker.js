import { Worker } from "bullmq";
import { Video } from "../models/Video.js";
import { redisConnection, videoQueueName } from "../queues/videoQueue.js";
import { processVideoWithAI } from "../services/ai.service.js";
import { uploadBuffer } from "../services/cloudinary.service.js";

export function createVideoWorker() {
  return new Worker(
    videoQueueName,
    async (job) => {
      const { videoId } = job.data;
      const video = await Video.findById(videoId);

      if (!video) {
        throw new Error("Video not found");
      }

      video.status = "processing";
      video.error = undefined;
      await video.save();

      // If the Cloudinary upload is still in flight, poll until the URL lands.
      if (!video.originalUrl) {
        for (let i = 0; i < 90; i++) {
          await new Promise((r) => setTimeout(r, 2000));
          const fresh = await Video.findById(videoId).select("originalUrl status");
          if (fresh?.originalUrl) { video.originalUrl = fresh.originalUrl; break; }
          if (fresh?.status === "failed") throw new Error("Video upload to Cloudinary failed");
        }
        if (!video.originalUrl) throw new Error("Timed out waiting for video upload");
      }

      try {
        const processedBuffer = await processVideoWithAI({
          videoUrl: video.originalUrl,
          detectedFaces: video.detectedFaces,
          filterAssignments: video.filterAssignments
        });

        const uploadResult = await uploadBuffer(processedBuffer, {
          folder: "cartoon-face-filter/processed",
          resource_type: "video"
        });

        video.status = "completed";
        video.processedUrl = uploadResult.secure_url;
        video.processedPublicId = uploadResult.public_id;
        await video.save();

        return {
          videoId: video.id,
          processedUrl: video.processedUrl
        };
      } catch (error) {
        console.error("[worker] processing failed", error);
        video.status = "failed";
        video.error = "Processing failed. Please try again.";
        await video.save();
        throw error;
      }
    },
    {
      connection: redisConnection,
      concurrency: 2
    }
  );
}
