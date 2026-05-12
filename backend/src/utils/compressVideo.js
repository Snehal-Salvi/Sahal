import { spawn } from "child_process";
import { mkdtemp, writeFile, readFile, rm } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

// Cloudinary's free plan caps each video upload at 100 MiB. We re-encode
// anything above this threshold with ffmpeg so the upload still succeeds.
// ffmpeg is already a hard dependency of the project (the AI service uses
// it) so no new package install is required.
export const CLOUDINARY_VIDEO_BYTES_LIMIT = 100 * 1024 * 1024;
export const COMPRESS_THRESHOLD_BYTES = 95 * 1024 * 1024;

export async function compressVideoBuffer(buffer) {
  const dir = await mkdtemp(join(tmpdir(), "sahal-compress-"));
  const inPath = join(dir, "in.mp4");
  const outPath = join(dir, "out.mp4");

  try {
    await writeFile(inPath, buffer);

    await new Promise((resolve, reject) => {
      // CRF 28 + 720p cap + AAC 128k typically lands a 4K iPhone clip well
      // under 100 MB while staying visually good for filter processing.
      const ff = spawn("ffmpeg", [
        "-y",
        "-i", inPath,
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        outPath,
      ]);

      let stderr = "";
      ff.stderr.on("data", (c) => { stderr += c.toString(); });
      ff.on("error", reject);
      ff.on("close", (code) => {
        if (code === 0) resolve();
        else reject(new Error(`ffmpeg compression failed (exit ${code}): ${stderr.slice(-500)}`));
      });
    });

    return await readFile(outPath);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
