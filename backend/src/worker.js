import "dotenv/config";
import "./config/cloudinary.js";
import { connectDatabase } from "./config/db.js";
import { createVideoWorker } from "./workers/video.worker.js";

await connectDatabase();

const worker = createVideoWorker();

worker.on("completed", (job, result) => {
  console.log(`Job ${job.id} completed`, result);
});

worker.on("failed", (job, error) => {
  console.error(`Job ${job?.id} failed`, error.message);
});

console.log("Video worker started");

