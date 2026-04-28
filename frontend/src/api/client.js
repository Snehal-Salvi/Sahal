import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:5001/api"
});

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("video", file);

  const { data } = await api.post("/videos/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });

  return data;
}

export async function uploadOverlay(file) {
  const formData = new FormData();
  formData.append("overlay", file);

  const { data } = await api.post("/videos/upload-overlay", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });

  return data;
}

export async function analyzeVideoFaces(videoId) {
  const { data } = await api.post(`/videos/${videoId}/analyze`);
  return data;
}

export async function queueVideoProcessing(videoId, filterAssignments) {
  const { data } = await api.post(`/videos/${videoId}/process`, {
    filterAssignments
  });

  return data;
}

export async function getVideoStatus(videoId) {
  const { data } = await api.get(`/videos/${videoId}/status`);
  return data;
}

export async function getBuiltInFilters() {
  const { data } = await api.get("/filters");
  return data;
}
