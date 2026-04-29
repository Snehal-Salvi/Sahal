import "./Step5Output.css";

export default function Step5Output({
  videoRecord,
  analysis,
  filterLibrary,
  assignedFilters,
  goTo,
  handleReset,
}) {
  return (
    <div className="s5">
      <div className="out-hdr">
        <div className="conf">
          {["#f472b6", "#fbbf24", "#4ade80", "#f472b6", "#60a5fa"].map((c, i) => (
            <div key={i} className="cd" style={{ background: c }} />
          ))}
        </div>
        <div className="ok-badge"><div className="gdot" />Done</div>
        <h2>Your Sahal video is ready!</h2>
        <p>
          Cartoon filters applied to {analysis?.faces?.length || 0}{" "}
          {analysis?.faces?.length === 1 ? "face" : "faces"} with full expression sync.
        </p>
      </div>

      <div className="out-grid">
        {/* ── Left: video player + actions ── */}
        <div>
          <div className="out-vcard">
            <div className="ba">
              <div className="ok-rib"><div className="gdot" />Processed successfully</div>
              {videoRecord?.processedUrl ? (
                <video src={videoRecord.processedUrl} controls playsInline />
              ) : (
                <div style={{ color: "#666", fontSize: 12 }}>Loading video…</div>
              )}
            </div>

            <div className="stats-row">
              <div className="stat">
                <div className="stat-v">{analysis?.faces?.length || 0}</div>
                <div className="stat-l">Faces</div>
              </div>
              <div className="stat"><div className="stat-v">—</div><div className="stat-l">Frames</div></div>
              <div className="stat"><div className="stat-v">HD</div><div className="stat-l">Quality</div></div>
              <div className="stat"><div className="stat-v">MP4</div><div className="stat-l">Format</div></div>
            </div>

            <div className="out-acts">
              {videoRecord?.processedUrl && (
                <a
                  className="act act-pk"
                  href={videoRecord.processedUrl}
                  download="sahal-output.mp4"
                  target="_blank"
                  rel="noreferrer"
                >
                  <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" /></svg>
                  Download Video
                </a>
              )}
              <button className="act act-gh" onClick={() => goTo(2)}>
                <svg viewBox="0 0 24 24">
                  <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
                </svg>
                Edit Again
              </button>
            </div>
          </div>
        </div>

        {/* ── Right: filter summary + share link ── */}
        <div className="out-right">
          <div className="orc">
            <div className="orc-t">Filters applied</div>
            {analysis?.faces?.map((face) => {
              const filter = filterLibrary.find((f) => f.id === assignedFilters[face.faceId]);
              return (
                <div key={face.faceId} className="fa-row">
                  <div className="fa-f">
                    <img
                      src={face.thumbnailDataUrl}
                      alt={face.label}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                  </div>
                  <div>
                    <div className="fa-n">{face.label}</div>
                    <div className="fa-fi">{filter?.name?.replace(/\.[^.]+$/, "") || "Filter"}</div>
                  </div>
                  <div className="fa-ck">✓</div>
                </div>
              );
            })}
          </div>

          {videoRecord?.processedUrl && (
            <div className="orc">
              <div className="orc-t">Video URL</div>
              <div className="slink">
                <span className="slink-t">{videoRecord.processedUrl}</span>
                <button
                  className="copy-b"
                  onClick={(e) => {
                    navigator.clipboard.writeText(videoRecord.processedUrl);
                    e.target.textContent = "Copied!";
                    setTimeout(() => { if (e.target) e.target.textContent = "Copy"; }, 2000);
                  }}
                >
                  Copy
                </button>
              </div>
            </div>
          )}

          <button className="proc-ano" onClick={handleReset}>
            + Process another video
          </button>
        </div>
      </div>
    </div>
  );
}
