import "./Step5Output.css";

export default function Step5Output({
  videoRecord,
  analysis,
  filterLibrary,
  assignedFilters,
  goTo,
  handleReset,
}) {
  const faceCount = analysis?.faces?.length || 0;

  return (
    <div className="s5-wrap">
      <div className="s5-left">
        <div className="out-hero">
          <div className="ok-pulse">
            <div className="ok-pulse-in">
              <svg viewBox="0 0 24 24" width="26" height="26">
                <path
                  d="M9 16.2 4.8 12l-1.4 1.4L9 19l12-12-1.4-1.4z"
                  fill="currentColor"
                />
              </svg>
            </div>
          </div>
          <div className="out-status">
            <span className="out-status-dot" />
            Done
          </div>
          <h2>Your Sahal video is ready</h2>
          <p>
            Cartoon filters applied to {faceCount}{" "}
            {faceCount === 1 ? "face" : "faces"} with full expression sync.
            Download it, share it, or jump back and try a different look.
          </p>
        </div>

        <div className="stats-card">
          <div className="stat">
            <div className="stat-v">{faceCount}</div>
            <div className="stat-l">Faces</div>
          </div>
          <div className="stat">
            <div className="stat-v">HD</div>
            <div className="stat-l">Quality</div>
          </div>
          <div className="stat">
            <div className="stat-v">MP4</div>
            <div className="stat-l">Format</div>
          </div>
          <div className="stat">
            <div className="stat-v">✓</div>
            <div className="stat-l">Audio</div>
          </div>
        </div>

        {videoRecord?.processedUrl && (
          <div className="url-card">
            <div className="url-card-t">Video URL</div>
            <div className="slink">
              <span className="slink-t">{videoRecord.processedUrl}</span>
              <button
                className="copy-b"
                onClick={(e) => {
                  navigator.clipboard.writeText(videoRecord.processedUrl);
                  e.target.textContent = "Copied!";
                  setTimeout(() => {
                    if (e.target) e.target.textContent = "Copy";
                  }, 2000);
                }}
              >
                Copy
              </button>
            </div>
          </div>
        )}

        <div className="out-acts">
          {videoRecord?.processedUrl && (
            <a
              className="act act-pk"
              href={videoRecord.processedUrl}
              download="sahal-output.mp4"
              target="_blank"
              rel="noreferrer"
            >
              <svg viewBox="0 0 24 24">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" />
              </svg>
              Download Video
            </a>
          )}
          <button className="act act-gh" onClick={() => goTo(2)}>
            <svg viewBox="0 0 24 24">
              <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
            </svg>
            Edit Again
          </button>
          <button className="act act-gh" onClick={handleReset}>
            <svg viewBox="0 0 24 24">
              <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6z" />
            </svg>
            New Video
          </button>
        </div>
      </div>

      <div className="s5-right">
        <div className="s5-preview">
          {videoRecord?.processedUrl ? (
            <div className="s5-prev-stage">
              <video
                src={videoRecord.processedUrl}
                controls
                playsInline
                className="s5-prev-vid"
              />
              <div className="s5-prev-label">Ready</div>
            </div>
          ) : (
            <div className="s5-prev-empty">Loading video…</div>
          )}
        </div>

        {analysis?.faces?.length > 0 && (
          <div className="fp2">
            {analysis.faces.map((face) => {
              const filter = filterLibrary.find(
                (f) => f.id === assignedFilters[face.faceId],
              );
              return (
                <div key={face.faceId} className="fpi">
                  <div className="fpi-top">
                    <div className="fpi-f">
                      <img src={face.thumbnailDataUrl} alt={face.label} />
                    </div>
                    <div className="fpi-copy">
                      <div className="fpi-n">{face.label}</div>
                      <div className="fpi-filter">
                        {filter
                          ? filter.name.replace(/\.[^.]+$/, "")
                          : "Mask applied"}
                      </div>
                    </div>
                    <div className="fpi-ck">✓</div>
                  </div>
                  <div className="fpi-bg">
                    <div className="fpi-fill" style={{ width: "100%" }} />
                  </div>
                  <div className="fpi-pct">Applied</div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
