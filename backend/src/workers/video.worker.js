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
        video.status = "failed";
        video.error = error.message;
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
