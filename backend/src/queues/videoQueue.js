import { Queue, QueueEvents } from "bullmq";
import IORedis from "ioredis";

export const redisConnection = new IORedis({
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

