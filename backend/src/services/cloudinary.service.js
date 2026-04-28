import { Readable } from "stream";
import { cloudinary } from "../config/cloudinary.js";

export function uploadBuffer(buffer, options = {}) {
  return new Promise((resolve, reject) => {
    const uploadOptions = {
      resource_type: "auto",
      folder: "cartoon-face-filter",
      ...options,
    };

    const isLarge = Buffer.isBuffer(buffer)
      ? buffer.length >= 8 * 1024 * 1024
      : false;

    const isVideo = uploadOptions.resource_type === "video";

    let stream;
    if (isVideo || isLarge) {
      stream = cloudinary.uploader.upload_chunked_stream(
        { chunk_size: 6 * 1024 * 1024, ...uploadOptions },
        (error, result) => {
          if (error) reject(error);
          else resolve(result);
        }
      );
    } else {
      stream = cloudinary.uploader.upload_stream(
        uploadOptions,
        (error, result) => {
          if (error) reject(error);
          else resolve(result);
        }
      );
    }

    Readable.from(buffer).pipe(stream);
  });
}
