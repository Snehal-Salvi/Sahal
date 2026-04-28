export default function Step2Detect({
  videoFile,
  analysis,
  applySameFilterToAll,
  setApplySameFilterToAll,
  status,
  error,
  handleAnalyzeFaces,
  goTo,
}) {
  const isLoading = status === "uploading" || status === "analyzing";
  const isDone = !!analysis && !isLoading;

  return (
    <div className="s2">
      {/* ── Error banner ── */}
      {error && !isLoading && (
        <div className="d2-error">
          <span>{error}</span>
          <button onClick={handleAnalyzeFaces}>Retry</button>
        </div>
      )}

      {/* ── Loading state ── */}
      {isLoading && (
        <div className="d2-loading">
          <div className="d2l-orbit">
            <div className="d2l-core">
              <svg viewBox="0 0 40 40" fill="none">
                <path d="M14 16c0-3.314 2.686-6 6-6s6 2.686 6 6v1H14v-1z" fill="rgba(244,114,182,0.7)" />
                <ellipse cx="20" cy="30" rx="8" ry="4.5" fill="rgba(244,114,182,0.7)" />
              </svg>
            </div>
            <div className="d2l-ring1" />
            <div className="d2l-ring2" />
          </div>

          <div className="d2l-status">
            <div className="d2l-status-label">
              {status === "uploading" ? "Preparing your video…" : "Looking for faces…"}
            </div>
            <div className="d2l-status-sub">
              {status === "uploading"
                ? "Just a moment, getting things ready"
                : "Sit tight, this usually takes a few seconds"}
            </div>
          </div>

          {videoFile && (
            <div className="d2l-file-chip">
              <span className="d2l-file-dot" />
              {videoFile.name}
            </div>
          )}

          <div className="d2l-track">
            <div className="d2l-track-fill" />
          </div>
        </div>
      )}

      {/* ── Results state ── */}
      {isDone && (
        <div className="d2-result">
          {/* Frame with bounding boxes */}
          <div className="d2r-frame">
            <img
              src={analysis.representativeFrameDataUrl}
              alt="Preview frame"
              className="d2r-img"
            />
            {analysis.faces.map((face) => {
              const b = face.representativeBox;
              if (!b) return null;
              return (
                <div
                  key={face.faceId}
                  className="d2r-fbox"
                  style={{
                    left: `${b.x * 100}%`,
                    top: `${b.y * 100}%`,
                    width: `${b.width * 100}%`,
                    height: `${b.height * 100}%`,
                  }}
                >
                  <span className="d2r-ftag">{face.label}</span>
                </div>
              );
            })}
            <div className="d2r-badge">
              <div className="d2r-dot" />
              {analysis.faces.length} {analysis.faces.length === 1 ? "face" : "faces"} detected
            </div>
          </div>

          {/* Face cards */}
          <div className="d2r-faces">
            {analysis.faces.map((face) => (
              <div key={face.faceId} className="d2r-face">
                <div className="d2r-avatar">
                  <img src={face.thumbnailDataUrl} alt={face.label} />
                </div>
                <div className="d2r-face-info">
                  <div className="d2r-face-name">{face.label}</div>
                  <div className="d2r-face-status">
                    <span className="d2r-face-ok">✓</span> Ready
                  </div>
                </div>
              </div>
            ))}

            {analysis.faces.length > 1 && (
              <div className="d2r-same-row">
                <div className="d2r-same-text">
                  <span className="d2r-same-title">Apply same filter</span>
                  <span className="d2r-same-sub">One style for every face</span>
                </div>
                <div
                  className={`tog${applySameFilterToAll ? " on" : ""}`}
                  onClick={() => setApplySameFilterToAll((v) => !v)}
                >
                  <div className="knob" />
                </div>
              </div>
            )}
          </div>

          {/* Actions */}
          <button className="d2r-cta" onClick={() => goTo(3)}>
            Choose Filters
            <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
          </button>

          <button className="d2r-rescan" onClick={handleAnalyzeFaces}>
            Re-scan faces
          </button>
        </div>
      )}
    </div>
  );
}
