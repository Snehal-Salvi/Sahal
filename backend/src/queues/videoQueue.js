import { Queue, QueueEvents } from "bullmq";
import IORedis from "ioredis";

// Prefer REDIS_URL (e.g. Upstash `rediss://default:pw@host:6379`) when set so
// password + TLS are wired automatically. Fall back to host/port for local dev.
export const redisConnection = process.env.REDIS_URL
  ? new IORedis(process.env.REDIS_URL, { maxRetriesPerRequest: null })
  : new IORedis({
      host: process.env.REDIS_HOST,
      port: Number(process.env.REDIS_PORT),
      maxRetriesPerRequest: null
    });

export const videoQueueName = "video-processing";

export const videoQueue = new Queue(videoQueueName, {
  connection: redisConnection
});

export const videoQueueEvents = new QueueEvents(videoQueueName, {
  connection: redisConnection
});

