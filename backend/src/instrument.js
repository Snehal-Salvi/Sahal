import "dotenv/config";
import * as Sentry from "@sentry/node";

const dsn = process.env.SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    environment: process.env.NODE_ENV || "development",
    tracesSampleRate: 0,
    sendDefaultPii: false
  });
  console.log("[sentry] initialized");
} else {
  console.log("[sentry] SENTRY_DSN not set — error reporting disabled");
}
