import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import * as Sentry from "@sentry/node";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import videoRoutes from "./routes/video.routes.js";
import filterRoutes from "./routes/filters.routes.js";
import authRoutes from "./routes/auth.routes.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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

// Trust X-Forwarded-* headers from the hosting platform's proxy so we can
// detect the real protocol and enforce HTTPS in production.
app.set("trust proxy", 1);
app.use((req, res, next) => {
  if (
    process.env.NODE_ENV === "production" &&
    req.headers["x-forwarded-proto"] &&
    req.headers["x-forwarded-proto"] !== "https"
  ) {
    return res.redirect(308, `https://${req.headers.host}${req.url}`);
  }
  return next();
});

// CSP is disabled here because Google Identity Services injects its own
// scripts/iframes; tighten with a custom directive set later if desired.
// CORP is set to cross-origin so the frontend (different origin) can load
// the filter PNGs served from /filters/*.
app.use(
  helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false,
    crossOriginResourcePolicy: { policy: "cross-origin" }
  })
);
app.use(
  cors({
    origin: process.env.FRONTEND_URL?.split(",") ?? ["http://localhost:5173"]
  })
);
app.use(express.json());

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false
});
const videoLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.use("/filters", express.static(join(__dirname, "../public/filters")));
app.use("/api/auth", authLimiter, authRoutes);
app.use("/api/filters", filterRoutes);
app.use("/api/videos", videoLimiter, videoRoutes);

Sentry.setupExpressErrorHandler(app);

app.use((error, req, res, next) => {
  console.error(error);
  const knownBadRequestMessages = new Set([
    "Only video files are allowed",
    "Only transparent PNG overlay images are allowed"
  ]);
  const statusCode =
    error.statusCode ||
    (knownBadRequestMessages.has(error.message) || error.code === "LIMIT_FILE_SIZE"
      ? 400
      : 500);

  res.status(statusCode).json({
    message:
      error.code === "LIMIT_FILE_SIZE"
        ? "Uploaded file is too large"
        : stringifyError(error)
  });
});

export default app;
