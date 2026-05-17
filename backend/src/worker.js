import "dotenv/config";
import * as Sentry from "@sentry/node";
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
  Sentry.captureException(error, { tags: { jobId: job?.id, jobName: job?.name } });
});

console.log("Video worker started");

let shuttingDown = false;
async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`[worker] received ${signal}, draining active jobs…`);
  try {
    await worker.close();
    console.log("[worker] shut down cleanly");
    process.exit(0);
  } catch (err) {
    console.error("[worker] error during shutdown", err);
    process.exit(1);
  }
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
