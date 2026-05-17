import "dotenv/config";
import "./config/cloudinary.js";
import app from "./app.js";
import { connectDatabase } from "./config/db.js";
import { createVideoWorker } from "./workers/video.worker.js";

const port = Number(process.env.PORT || 5000);

await connectDatabase();

const server = app.listen(port, () => {
  console.log(`API server listening on port ${port}`);
});

// On single-process hosts (e.g. Render free tier) run the worker in-process.
// Locally we still launch the worker as a separate `npm run worker` for clean
// dev logs, so leave RUN_WORKER unset there.
let worker = null;
if (process.env.RUN_WORKER === "true") {
  worker = createVideoWorker();
  worker.on("completed", (job, result) => {
    console.log(`Job ${job.id} completed`, result);
  });
  worker.on("failed", (job, error) => {
    console.error(`Job ${job?.id} failed`, error.message);
  });
  console.log("In-process video worker started");
}

let shuttingDown = false;
async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`[server] received ${signal}, shutting down…`);
  try {
    if (worker) await worker.close();
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(1), 10_000).unref();
  } catch (err) {
    console.error("[server] error during shutdown", err);
    process.exit(1);
  }
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
