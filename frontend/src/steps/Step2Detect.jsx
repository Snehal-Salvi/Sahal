export default function Step2Detect({
  videoPreviewUrl,
  videoFile,
  analysis,
  selectedFaceId,
  applySameFilterToAll,
  setSelectedFaceId,
  setApplySameFilterToAll,
  status,
  error,
  handleAnalyzeFaces,
  goTo,
}) {
  return (
    <div className="s2">
      <div className="sh">
        <div className="stag">Step 2</div>
        <h2>Detect faces in your video</h2>
        <p>Preview your video, then click "Detect Faces" to scan automatically.</p>
      </div>

      {error && <div className="error-msg">{error}</div>}

      <div className="det-grid">
        {/* ── Left: video preview + detect button ── */}
        <div>
          <div className="vcard" style={{ marginBottom: 10 }}>
            <div className="varea">
              {!videoPreviewUrl && (
                <div className="det-placeholder">
                  <div style={{ fontSize: 28, marginBottom: 6 }}>🎥</div>
                  No video selected
                </div>
              )}

              {videoPreviewUrl && !analysis && (
                <video src={videoPreviewUrl} controls playsInline className="det-video" />
              )}

              {videoPreviewUrl && analysis && (
                <div className="det-overlay-wrap">
                  <img src={analysis.representativeFrameDataUrl} alt="Frame" className="rep-frame" />
                  {analysis.faces.map((face) => {
                    const box = face.representativeBox;
                    if (!box) return null;
                    return (
                      <div
                        key={face.faceId}
                        className="fbox"
                        style={{
                          left: `${box.x * 100}%`,
                          top: `${box.y * 100}%`,
                          width: `${box.width * 100}%`,
                          height: `${box.height * 100}%`,
                        }}
                      >
                        <div className="ftag">{face.label} · 98%</div>
                      </div>
                    );
                  })}
                  <div className="vbadge">
                    <div className="vdot" />
                    468 landmarks · {analysis.faces.length}{" "}
                    {analysis.faces.length === 1 ? "face" : "faces"}
                  </div>
                </div>
              )}
            </div>

            <div className="vctrls">
              <div className="vplay">
                <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              </div>
              <div className="vtl"><div className="vtlf" /></div>
              <span className="vt">{videoFile?.name || "—"}</span>
            </div>
          </div>

          <button
            className="dbtn"
            onClick={handleAnalyzeFaces}
            disabled={!videoFile || status === "uploading" || status === "analyzing"}
          >
            <svg viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" />
            </svg>
            <span>
              {status === "uploading"
                ? "Uploading…"
                : status === "analyzing"
                ? "Scanning…"
                : analysis
                ? "Re-detect Faces"
                : "Detect Faces"}
            </span>
          </button>
        </div>

        {/* ── Right: sidebar ── */}
        <div className="sidebar">
          {analysis ? (
            <>
              <div className="scard">
                <div className="scard-t">Detected faces</div>
                {analysis.faces.map((face) => (
                  <div
                    key={face.faceId}
                    className={`frow${selectedFaceId === face.faceId ? " sel" : ""}`}
                    onClick={() => setSelectedFaceId(face.faceId)}
                  >
                    <div className="fcirc">
                      <img
                        src={face.thumbnailDataUrl}
                        alt={face.label}
                        style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      />
                    </div>
                    <div>
                      <div className="finfo-n">{face.label}</div>
                      <div className="finfo-s">Front view</div>
                    </div>
                    <div className={`fchk${selectedFaceId === face.faceId ? "" : " off"}`} />
                  </div>
                ))}
              </div>

              <div className="scard">
                <div className="tog-row">
                  <div className="tog-title">Same filter for all</div>
                  <div
                    className={`tog${applySameFilterToAll ? " on" : ""}`}
                    onClick={() => setApplySameFilterToAll((v) => !v)}
                  >
                    <div className="knob" />
                  </div>
                </div>
                <div className="tog-desc">
                  Toggle to apply one filter to every detected face at once.
                </div>
              </div>

              <div className="scard">
                <div className="check-row">
                  <div className="ckdot" />
                  {analysis.faces.length} {analysis.faces.length === 1 ? "face" : "faces"} detected
                </div>
                <div className="check-row"><div className="ckdot" />Landmarks mapped (468 pts)</div>
                <div className="check-row"><div className="ckdot" />Expression tracking ready</div>
                <button className="ready-btn" onClick={() => goTo(3)}>Choose Filters →</button>
              </div>
            </>
          ) : (
            <div className="scard">
              <div className="scard-t">Tips</div>
              <div style={{ fontSize: 11, color: "#666", lineHeight: 1.5, marginBottom: 6 }}>
                <span style={{ color: "#f472b6", fontWeight: 600 }}>Best results:</span>{" "}
                Use well-lit videos with faces visible from the front.
              </div>
              <div style={{ fontSize: 11, color: "#666", lineHeight: 1.5 }}>
                <span style={{ color: "#f472b6", fontWeight: 600 }}>Multiple faces:</span>{" "}
                Each person gets their own filter in the next step.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
