import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeVideoFaces,
  getVideoStatus,
  queueVideoProcessing,
  uploadOverlay,
  uploadVideo
} from "../api/client";

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

async function pngHasAlphaChannel(file) {
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);

  if (bytes.length < 33) {
    return false;
  }

  const hasSignature = PNG_SIGNATURE.every((value, index) => bytes[index] === value);
  if (!hasSignature) {
    return false;
  }

  const view = new DataView(arrayBuffer);
  const ihdrLength = view.getUint32(8);
  const ihdrType = String.fromCharCode(...bytes.slice(12, 16));

  if (ihdrLength !== 13 || ihdrType !== "IHDR") {
    return false;
  }

  const colorType = bytes[25];
  if (colorType === 4 || colorType === 6) {
    return true;
  }

  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const chunkLength = view.getUint32(offset);
    const chunkType = String.fromCharCode(...bytes.slice(offset + 4, offset + 8));

    if (chunkType === "tRNS") {
      return true;
    }

    offset += chunkLength + 12;
  }

  return false;
}

function buildFaceSelectionMap(faces, faceSelections, sharedFilterId, applySameFilterToAll) {
  return faces.reduce((selectionMap, face) => {
    selectionMap[face.faceId] = applySameFilterToAll
      ? sharedFilterId || ""
      : faceSelections[face.faceId] || "";
    return selectionMap;
  }, {});
}

export default function VideoUploader() {
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
  const [message, setMessage] = useState("Upload a video, analyze faces, and assign filters.");
  const pollingRef = useRef(null);
  const videoPreviewRef = useRef("");
  const filterLibraryRef = useRef([]);

  useEffect(() => {
    videoPreviewRef.current = videoPreviewUrl;
    filterLibraryRef.current = filterLibrary;
  }, [filterLibrary, videoPreviewUrl]);

  useEffect(() => {
    return () => {
      if (videoPreviewRef.current) {
        URL.revokeObjectURL(videoPreviewRef.current);
      }

      for (const filter of filterLibraryRef.current) {
        URL.revokeObjectURL(filter.previewUrl);
      }

      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, []);

  const assignedFilters = useMemo(() => {
    const faces = analysis?.faces || [];
    return buildFaceSelectionMap(faces, faceSelections, sharedFilterId, applySameFilterToAll);
  }, [analysis?.faces, applySameFilterToAll, faceSelections, sharedFilterId]);

  const activeFace = useMemo(
    () => analysis?.faces?.find((face) => face.faceId === selectedFaceId) || analysis?.faces?.[0] || null,
    [analysis?.faces, selectedFaceId]
  );

  function resetSelections(nextFaces = []) {
    setFaceSelections({});
    setSharedFilterId("");
    setApplySameFilterToAll(false);
    setSelectedFaceId(nextFaces[0]?.faceId || "");
  }

  function startPolling(videoId) {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current);
    }

    pollingRef.current = window.setInterval(async () => {
      try {
        const { video } = await getVideoStatus(videoId);
        setVideoRecord(video);
        setStatus(video.status);

        if (video.status === "completed") {
          setMessage("Processing completed.");
          window.clearInterval(pollingRef.current);
          pollingRef.current = null;
        }

        if (video.status === "failed") {
          setMessage(video.error || "Processing failed.");
          window.clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
      } catch (error) {
        setMessage(error.response?.data?.message || error.message);
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }, 4000);
  }

  async function ensureVideoUploaded() {
    if (videoRecord?._id) {
      return videoRecord;
    }

    if (!videoFile) {
      throw new Error("Choose a video file first.");
    }

    setStatus("uploading");
    setMessage("Uploading your source video...");
    const uploadResponse = await uploadVideo(videoFile);
    setVideoRecord(uploadResponse.video);
    return uploadResponse.video;
  }

  async function handleAnalyzeFaces() {
    try {
      const uploadedVideo = await ensureVideoUploaded();
      setStatus("analyzing");
      setMessage("Detecting and tracking faces across the video...");

      const response = await analyzeVideoFaces(uploadedVideo._id);
      const detectedFaces = response.analysis?.faces || [];
      setAnalysis(response.analysis);
      setVideoRecord(response.video);
      resetSelections(detectedFaces);
      setStatus("ready");
      setMessage(`Detected ${detectedFaces.length} face${detectedFaces.length === 1 ? "" : "s"}. Assign filters to continue.`);
    } catch (error) {
      setStatus("error");
      setMessage(error.response?.data?.message || error.message);
    }
  }

  async function handleProcessVideo() {
    if (!analysis?.faces?.length) {
      setMessage("Analyze the video first so we can label and track faces.");
      return;
    }

    if (!filterLibrary.length) {
      setMessage("Upload at least one transparent PNG filter first.");
      return;
    }

    const unassignedFace = analysis.faces.find((face) => !assignedFilters[face.faceId]);
    if (unassignedFace) {
      setMessage(`Select a filter for ${unassignedFace.label} before processing.`);
      return;
    }

    try {
      const uploadedVideo = await ensureVideoUploaded();
      setStatus("uploading");
      setMessage("Uploading selected filters...");

      const uploadCache = new Map();
      for (const filter of filterLibrary) {
        if (Object.values(assignedFilters).includes(filter.id)) {
          const response = await uploadOverlay(filter.file);
          uploadCache.set(filter.id, response.overlayUrl);
        }
      }

      const filterAssignments = analysis.faces.map((face) => ({
        faceId: face.faceId,
        overlayImageUrl: uploadCache.get(assignedFilters[face.faceId])
      }));

      setStatus("queued");
      setMessage("Queueing video processing...");
      const processResponse = await queueVideoProcessing(uploadedVideo._id, filterAssignments);
      setVideoRecord(processResponse.video);
      setStatus(processResponse.video.status);
      setMessage("Processing started. Polling for completion...");
      startPolling(uploadedVideo._id);
    } catch (error) {
      setStatus("error");
      setMessage(error.response?.data?.message || error.message);
    }
  }

  function handleVideoFileChange(event) {
    const nextFile = event.target.files?.[0];
    setVideoFile(nextFile || null);
    setVideoRecord(null);
    setAnalysis(null);
    resetSelections();
    setStatus("idle");

    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }

    if (nextFile) {
      setVideoPreviewUrl(URL.createObjectURL(nextFile));
      setMessage("Video ready. Analyze faces when you’re ready.");
    } else {
      setVideoPreviewUrl("");
      setMessage("Upload a video, analyze faces, and assign filters.");
    }
  }

  async function handleFilterLibraryChange(event) {
    const incomingFiles = Array.from(event.target.files || []);
    if (!incomingFiles.length) {
      return;
    }

    const nextFilters = [];
    for (const file of incomingFiles) {
      const hasAlpha = await pngHasAlphaChannel(file);
      if (!hasAlpha) {
        setMessage(`Skipped ${file.name}. Filters must be transparent PNG files.`);
        continue;
      }

      nextFilters.push({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        file,
        previewUrl: URL.createObjectURL(file),
        name: file.name
      });
    }

    setFilterLibrary((current) => [...current, ...nextFilters]);
    if (nextFilters.length) {
      setMessage("Filter library updated. Pick a face and assign a filter.");
    }

    event.target.value = "";
  }

  function removeFilter(filterId) {
    setFilterLibrary((current) => {
      const target = current.find((filter) => filter.id === filterId);
      if (target) {
        URL.revokeObjectURL(target.previewUrl);
      }

      return current.filter((filter) => filter.id !== filterId);
    });

    setFaceSelections((current) =>
      Object.fromEntries(Object.entries(current).filter(([, value]) => value !== filterId))
    );

    if (sharedFilterId === filterId) {
      setSharedFilterId("");
    }
  }

  function assignFilter(filterId) {
    if (applySameFilterToAll) {
      setSharedFilterId(filterId);
      return;
    }

    if (!selectedFaceId) {
      setMessage("Select a detected face first.");
      return;
    }

    setFaceSelections((current) => ({
      ...current,
      [selectedFaceId]: filterId
    }));
  }

  return (
    <section className="panel">
      <div className="panel-copy">
        <p className="eyebrow">React + Express + BullMQ + FastAPI</p>
        <h1>Multi-Face Cartoon Filter</h1>
        <p className="lede">
          Upload a video, analyze every tracked face, assign one filter per face or
          one filter for everyone, and preview the selection before rendering.
        </p>
      </div>

      <div className="workflow-grid">
        <article className="status-card">
          <strong>Status:</strong> {status}
          <p>{message}</p>
          {videoRecord?.jobId ? <p>Job ID: {videoRecord.jobId}</p> : null}
        </article>

        <article className="status-card">
          <strong>Workflow</strong>
          <p>1. Upload video</p>
          <p>2. Analyze faces</p>
          <p>3. Upload filters and assign them</p>
          <p>4. Process the video</p>
        </article>
      </div>

      <div className="field-group">
        <label className="field">
          <span>Video upload</span>
          <input type="file" accept="video/*" onChange={handleVideoFileChange} />
        </label>

        <div className="button-row">
          <button className="primary-button" type="button" onClick={handleAnalyzeFaces} disabled={!videoFile || status === "analyzing" || status === "processing" || status === "queued"}>
            {status === "analyzing" ? "Analyzing..." : "Upload and Analyze Faces"}
          </button>
          <button
            className="primary-button secondary-button"
            type="button"
            onClick={handleProcessVideo}
            disabled={!analysis?.faces?.length || status === "processing" || status === "queued"}
          >
            {status === "uploading" || status === "queued" || status === "processing" ? "Working..." : "Process Video"}
          </button>
        </div>
      </div>

      <div className="video-grid">
        <article className="video-card">
          <h2>Source Video</h2>
          {videoPreviewUrl ? <video src={videoPreviewUrl} controls playsInline /> : <p>No video selected yet.</p>}
        </article>

        <article className="video-card">
          <h2>Tracked Preview</h2>
          {analysis?.representativeFrameDataUrl ? (
            <div className="analysis-preview">
              <img src={analysis.representativeFrameDataUrl} alt="Representative frame preview" />
              {analysis.faces.map((face) => {
                const box = face.representativeBox;
                const selectedFilter = filterLibrary.find((filter) => filter.id === assignedFilters[face.faceId]);

                if (!box) {
                  return null;
                }

                return (
                  <div
                    key={face.faceId}
                    className={`preview-face-box ${selectedFaceId === face.faceId ? "is-active" : ""}`}
                    style={{
                      left: `${box.x * 100}%`,
                      top: `${box.y * 100}%`,
                      width: `${box.width * 100}%`,
                      height: `${box.height * 100}%`
                    }}
                  >
                    {selectedFilter ? (
                      <img
                        className="preview-face-filter"
                        src={selectedFilter.previewUrl}
                        alt={`${face.label} filter preview`}
                      />
                    ) : null}
                    <span>{face.label}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p>Representative preview appears after face analysis.</p>
          )}
        </article>

        <article className="video-card">
          <h2>Processed Output</h2>
          {videoRecord?.processedUrl ? (
            <video src={videoRecord.processedUrl} controls playsInline />
          ) : (
            <p>Processed video will appear here when the job completes.</p>
          )}
        </article>
      </div>

      <div className="assignment-grid">
        <article className="video-card">
          <div className="card-header">
            <h2>Detected Faces</h2>
            <span>{analysis?.faces?.length || 0} tracked</span>
          </div>
          {analysis?.faces?.length ? (
            <div className="face-grid">
              {analysis.faces.map((face) => (
                <button
                  key={face.faceId}
                  className={`face-chip ${selectedFaceId === face.faceId ? "is-selected" : ""}`}
                  type="button"
                  onClick={() => setSelectedFaceId(face.faceId)}
                >
                  <img src={face.thumbnailDataUrl} alt={face.label} />
                  <strong>{face.label}</strong>
                  <small>
                    {assignedFilters[face.faceId]
                      ? filterLibrary.find((filter) => filter.id === assignedFilters[face.faceId])?.name || "Assigned"
                      : "No filter yet"}
                  </small>
                </button>
              ))}
            </div>
          ) : (
            <p>Analyze the video to generate labeled face thumbnails.</p>
          )}
        </article>

        <article className="video-card">
          <div className="card-header">
            <h2>Filter Library</h2>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={applySameFilterToAll}
                onChange={(event) => setApplySameFilterToAll(event.target.checked)}
              />
              <span>Apply the same filter to all faces</span>
            </label>
          </div>

          <label className="field">
            <span>Upload transparent PNG filters</span>
            <input type="file" accept="image/png" multiple onChange={handleFilterLibraryChange} />
          </label>

          {activeFace ? (
            <p className="selection-copy">
              {applySameFilterToAll
                ? "Selecting a filter now applies it to every tracked face."
                : `Choose a filter for ${activeFace.label}.`}
            </p>
          ) : (
            <p className="selection-copy">Analyze faces first, then select a face and assign a filter.</p>
          )}

          {filterLibrary.length ? (
            <div className="filter-grid">
              {filterLibrary.map((filter) => {
                const isSelected = applySameFilterToAll
                  ? sharedFilterId === filter.id
                  : assignedFilters[selectedFaceId] === filter.id;

                return (
                  <div key={filter.id} className={`filter-card ${isSelected ? "is-selected" : ""}`}>
                    <button type="button" className="filter-button" onClick={() => assignFilter(filter.id)}>
                      <img src={filter.previewUrl} alt={filter.name} />
                      <strong>{filter.name}</strong>
                    </button>
                    <button type="button" className="remove-filter-button" onClick={() => removeFilter(filter.id)}>
                      Remove
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <p>No filters uploaded yet.</p>
          )}
        </article>
      </div>
    </section>
  );
}
