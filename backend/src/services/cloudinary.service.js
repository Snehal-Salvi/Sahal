import { cloudinary } from "../config/cloudinary.js";

export function uploadBuffer(buffer, options = {}) {
  return new Promise((resolve, reject) => {
    const uploadOptions = {
      resource_type: "video",
      folder: "cartoon-face-filter",
      timeout: 120000,
      ...options
    };
    const useChunkedUpload =
      uploadOptions.resource_type === "video" || buffer.length >= 8 * 1024 * 1024;
    const uploadMethod = useChunkedUpload
      ? cloudinary.uploader.upload_chunked_stream
      : cloudinary.uploader.upload_stream;
    const streamOptions = useChunkedUpload
      ? {
          chunk_size: 6 * 1024 * 1024,
          ...uploadOptions
        }
      : uploadOptions;
    const stream = uploadMethod(
      streamOptions,
      (error, result) => {
        if (error) {
          reject(error);
          return;
        }

        resolve(result);
      }
    );

    stream.end(buffer);
  });
}
