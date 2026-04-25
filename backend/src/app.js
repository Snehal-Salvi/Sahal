import "dotenv/config";
import express from "express";
import cors from "cors";
import videoRoutes from "./routes/video.routes.js";

const app = express();
const stringifyError = (error) => {
  if (!error) {
    return "Internal server error";
  }

  if (typeof error.message === "string" && error.message.trim()) {
    return error.message;
  }

  if (typeof error === "string" && error.trim()) {
    return error;
  }

  try {
    return JSON.stringify(error);
  } catch {
    return "Internal server error";
  }
};

app.use(
  cors({
    origin: process.env.FRONTEND_URL?.split(",") ?? ["http://localhost:5173"]
  })
);
app.use(express.json());

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.use("/api/videos", videoRoutes);

app.use((error, req, res, next) => {
  console.error(error);
  const knownBadRequestMessages = new Set([
    "Only video files are allowed",
    "Only transparent PNG overlay images are allowed"
  ]);
  const statusCode =
    knownBadRequestMessages.has(error.message) || error.code === "LIMIT_FILE_SIZE"
      ? 400
      : 500;

  res.status(statusCode).json({
    message:
      error.code === "LIMIT_FILE_SIZE"
        ? "Uploaded file is too large"
        : stringifyError(error)
  });
});

export default app;
