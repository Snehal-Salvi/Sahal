import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeVideoFaces,
  getBuiltInFilters,
  getVideoStatus,
  queueVideoProcessing,
  uploadOverlay,
  uploadVideo,
} from "../api/client";
import { pngHasAlphaChannel } from "../utils/pngValidation";

function buildFaceSelectionMap(faces, faceSelections, sharedFilterId, applySameFilterToAll) {
  return faces.reduce((map, face) => {
    map[face.faceId] = applySameFilterToAll
      ? sharedFilterId || ""
      : faceSelections[face.faceId] || "";
    return map;
  }, {});
}

export function useVideoProcessor() {
  const [step, setStep] = useState(1);
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState("");
  const [videoRecord, setVideoRecord] = useState(null);
  const [filterLibrary, setFilterLibrary] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [selectedFaceId, setSelectedFaceId] = useState("");
  const [faceSelections, setFaceSelections] = useState({});
  const [applySameFilterToAll, setApplySameFilterToAll] = useState(false);
  const [sharedFilterId, setSharedFilterId] = useState("");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const pollingRef = useRef(null);
  const videoPreviewRef = useRef("");
  const filterLibraryRef = useRef([]);
  const fileInputRef = useRef(null);
  const filterInputRef = useRef(null);
  // Tracks the file we already auto-triggered analysis for, to avoid re-firing
  const autoAnalyzedKeyRef = useRef("");

  useEffect(() => {
    videoPreviewRef.current = videoPreviewUrl;
    filterLibraryRef.current = filterLibrary;
  }, [filterLibrary, videoPreviewUrl]);

  useEffect(() => {
    return () => {
      if (videoPreviewRef.current) URL.revokeObjectURL(videoPreviewRef.current);
      // Only revoke object URLs for custom uploads, not built-in filters
      for (const f of filterLibraryRef.current) {
        if (!f.isBuiltIn) URL.revokeObjectURL(f.previewUrl);
      }
      if (pollingRef.current) window.clearInterval(pollingRef.current);
    };
  }, []);

  // Auto-trigger analysis the moment the user arrives at step 2 with a fresh video
  useEffect(() => {
    if (step !== 2 || !videoFile || analysis) return;
    const key = `${videoFile.name}-${videoFile.size}-${videoFile.lastModified}`;
    if (autoAnalyzedKeyRef.current === key) return;
    autoAnalyzedKeyRef.current = key;
    handleAnalyzeFaces();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, videoFile, analysis]);

  // Load built-in filters from the backend on mount
  useEffect(() => {
    getBuiltInFilters()
      .then((filters) => {
        const builtins = filters.map((f) => ({
          id: f.id,
          name: f.name,
          previewUrl: f.url,
          url: f.url,
          isBuiltIn: true,
        }));
        setFilterLibrary((cur) => {
          // Prepend built-ins; don't duplicate if already present
          const existingIds = new Set(cur.map((f) => f.id));
          return [...builtins.filter((b) => !existingIds.has(b.id)), ...cur];
        });
      })
      .catch(() => {}); // Silently ignore if backend isn't up yet
  }, []);

  const assignedFilters = useMemo(
    () => buildFaceSelectionMap(analysis?.faces || [], faceSelections, sharedFilterId, applySameFilterToAll),
    [analysis?.faces, applySameFilterToAll, faceSelections, sharedFilterId]
  );

  const activeFace = useMemo(
    () => analysis?.faces?.find((f) => f.faceId === selectedFaceId) || analysis?.faces?.[0] || null,
    [analysis?.faces, selectedFaceId]
  );

  function resetSelections(nextFaces = []) {
    setFaceSelections({});
    setSharedFilterId("");
    setApplySameFilterToAll(false);
    setSelectedFaceId(nextFaces[0]?.faceId || "");
  }

  function goTo(n) {
    setStep(n);
    setError("");
  }

  function startPolling(videoId) {
    if (pollingRef.current) window.clearInterval(pollingRef.current);
    pollingRef.current = window.setInterval(async () => {
      try {
        const { video } = await getVideoStatus(videoId);
        setVideoRecord(video);
        setStatus(video.status);
        if (video.status === "completed") {
          window.clearInterval(pollingRef.current);
          pollingRef.current = null;
          goTo(5);
        }
        if (video.status === "failed") {
          window.clearInterval(pollingRef.current);
          pollingRef.current = null;
          setError(video.error || "Processing failed.");
          goTo(3);
        }
      } catch (err) {
        setError(err.response?.data?.message || err.message);
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
        goTo(3);
      }
    }, 4000);
  }

  async function ensureVideoUploaded() {
    if (videoRecord?._id) return videoRecord;
    if (!videoFile) throw new Error("Choose a video file first.");
    setStatus("uploading");
    const res = await uploadVideo(videoFile);
    setVideoRecord(res.video);
    return res.video;
  }

  function applyVideoFile(file) {
    if (!file) return;
    setVideoFile(file);
    setVideoRecord(null);
    setAnalysis(null);
    resetSelections();
    setStatus("idle");
    setError("");
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(URL.createObjectURL(file));
    goTo(2);
  }

  function handleVideoFileChange(event) {
    applyVideoFile(event.target.files?.[0]);
  }

  function handleVideoFileDrop(file) {
    applyVideoFile(file);
  }

  async function handleAnalyzeFaces() {
    if (!videoFile) { setError("Choose a video file first."); return; }
    try {
      setError("");
      const uploadedVideo = await ensureVideoUploaded();
      setStatus("analyzing");
      const response = await analyzeVideoFaces(uploadedVideo._id);
      const detectedFaces = response.analysis?.faces || [];
      setAnalysis(response.analysis);
      setVideoRecord(response.video);
      resetSelections(detectedFaces);
      setStatus("ready");
    } catch (err) {
      setStatus("error");
      setError(err.response?.data?.message || err.message);
    }
  }

  async function handleFilterLibraryChange(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    const next = [];
    for (const file of files) {
      const hasAlpha = await pngHasAlphaChannel(file);
      if (!hasAlpha) { setError(`Skipped "${file.name}" — must be a transparent PNG.`); continue; }
      next.push({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        file,
        previewUrl: URL.createObjectURL(file),
        name: file.name,
      });
    }
    setFilterLibrary((cur) => [...cur, ...next]);
    event.target.value = "";
  }

  function removeFilter(filterId) {
    setFilterLibrary((cur) => {
      const target = cur.find((f) => f.id === filterId);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return cur.filter((f) => f.id !== filterId);
    });
    setFaceSelections((cur) =>
      Object.fromEntries(Object.entries(cur).filter(([, v]) => v !== filterId))
    );
    if (sharedFilterId === filterId) setSharedFilterId("");
  }

  function assignFilter(filterId) {
    if (applySameFilterToAll) { setSharedFilterId(filterId); return; }
    if (!selectedFaceId) return;
    setFaceSelections((cur) => ({ ...cur, [selectedFaceId]: filterId }));
  }

  async function handleProcessVideo() {
    if (!analysis?.faces?.length) { setError("Analyze the video first."); return; }
    if (!filterLibrary.length) { setError("Upload at least one transparent PNG filter first."); return; }
    const unassigned = analysis.faces.find((f) => !assignedFilters[f.faceId]);
    if (unassigned) { setError(`Assign a filter to ${unassigned.label} before processing.`); return; }
    setError("");
    goTo(4);
    setStatus("uploading");
    try {
      const uploadedVideo = await ensureVideoUploaded();
      const uploadCache = new Map();
      for (const filter of filterLibrary) {
        if (Object.values(assignedFilters).includes(filter.id)) {
          if (filter.isBuiltIn) {
            // Built-in filters are already hosted on the backend — use their URL directly
            uploadCache.set(filter.id, filter.url);
          } else {
            const res = await uploadOverlay(filter.file);
            uploadCache.set(filter.id, res.overlayUrl);
          }
        }
      }
      const filterAssignments = analysis.faces.map((face) => ({
        faceId: face.faceId,
        overlayImageUrl: uploadCache.get(assignedFilters[face.faceId]),
      }));
      setStatus("queued");
      const processRes = await queueVideoProcessing(uploadedVideo._id, filterAssignments);
      setVideoRecord(processRes.video);
      setStatus(processRes.video.status);
      startPolling(uploadedVideo._id);
    } catch (err) {
      setStatus("error");
      setError(err.response?.data?.message || err.message);
      goTo(3);
    }
  }

  function handleReset() {
    for (const f of filterLibraryRef.current) {
      if (!f.isBuiltIn) URL.revokeObjectURL(f.previewUrl);
    }
    if (videoPreviewRef.current) URL.revokeObjectURL(videoPreviewRef.current);
    setVideoFile(null);
    setVideoRecord(null);
    setAnalysis(null);
    // Keep built-in filters; clear only custom uploads
    setFilterLibrary((cur) => cur.filter((f) => f.isBuiltIn));
    resetSelections();
    setStatus("idle");
    setError("");
    setVideoPreviewUrl("");
    goTo(1);
  }

  return {
    step,
    videoFile,
    videoPreviewUrl,
    videoRecord,
    filterLibrary,
    analysis,
    selectedFaceId,
    setSelectedFaceId,
    applySameFilterToAll,
    setApplySameFilterToAll,
    sharedFilterId,
    status,
    error,
    assignedFilters,
    activeFace,
    fileInputRef,
    filterInputRef,
    goTo,
    handleVideoFileChange,
    handleVideoFileDrop,
    handleAnalyzeFaces,
    handleFilterLibraryChange,
    removeFilter,
    assignFilter,
    handleProcessVideo,
    handleReset,
  };
}
