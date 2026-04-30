import { useEffect, useState } from "react";
import "./Step2Detect.css";

const FAKE_BOXES = [
  { left: "16%", top: "18%", width: "28%", height: "44%", label: "Face 1", scanAt: 28 },
  { left: "58%", top: "22%", width: "24%", height: "38%", label: "Face 2", scanAt: 62 },
];

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

  const [scanPct, setScanPct] = useState(0);
  const [foundCount, setFoundCount] = useState(0);

  useEffect(() => {
    if (!isLoading) { setScanPct(0); setFoundCount(0); return; }
    let pct = 0;
    let found = 0;
    const t = setInterval(() => {
      pct = (pct + 0.6) % 100;
      setScanPct(pct);
      const nowFound = FAKE_BOXES.filter((b) => pct > b.scanAt).length;
      if (nowFound !== found) { found = nowFound; setFoundCount(nowFound); }
    }, 25);
    return () => clearInterval(t);
  }, [isLoading]);

  return (
    <div className="d2-wrap">

      {/* ── Left panel ── */}
      <div className="d2-left">
        {error && !isLoading && (
          <div className="d2-error">
            <span>{error}</span>
            <button onClick={handleAnalyzeFaces}>Retry</button>
          </div>
        )}

        {isLoading && (
          <div className="d2-loading">
            <div className="d2l-orbit">
              <div className="d2l-core">
                <svg viewBox="0 0 40 40" fill="none">
                  <path d="M14 16c0-3.314 2.686-6 6-6s6 2.686 6 6v1H14v-1z" fill="rgba(244,114,182,0.8)" />
                  <ellipse cx="20" cy="30" rx="8" ry="4.5" fill="rgba(244,114,182,0.8)" />
                </svg>
              </div>
              <div className="d2l-ring1" />
              <div className="d2l-ring2" />
            </div>
            <div className="d2l-status">
              <div className="d2l-status-label">Detecting faces…</div>
              <div className="d2l-status-sub">
                Sit tight, this usually takes a few seconds
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

        {isDone && (
          <div className="d2-done-panel">
            <div className="d2-done-badge">
              <span className="d2-done-dot" />
              {analysis.faces.length} {analysis.faces.length === 1 ? "face" : "faces"} detected
            </div>
            <h2 className="d2-done-title">Ready to animate</h2>
            <p className="d2-done-sub">
              {analysis.faces.length === 1
                ? "We found 1 person in your video."
                : `We found ${analysis.faces.length} people in your video.`}{" "}
              Assign a filter to each one to get started.
            </p>

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
                  <div>
                    <span className="d2r-same-title">Apply same filter</span>
                    <span className="d2r-same-sub"> — one style for everyone</span>
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

      {/* ── Right panel ── */}
      <div className="d2-right">
        {isLoading && (
          <div className="d2-scan-wrap">
            <div className="d2-scan-header">
              <span className="d2-scan-status-dot" />
              {`Scanning · ${foundCount} found`}
            </div>

            <div className="d2-scan-frame">
              {/* Corner brackets */}
              <div className="d2-corner d2-tl" />
              <div className="d2-corner d2-tr" />
              <div className="d2-corner d2-bl" />
              <div className="d2-corner d2-br" />

              {/* Sweep line */}
              <div className="d2-scan-line" style={{ top: `${scanPct}%` }} />

              {/* Fake face boxes */}
              {FAKE_BOXES.map((box, i) => (
                scanPct > box.scanAt && (
                  <div
                    key={i}
                    className="d2-fake-box"
                    style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
                  >
                    <span className="d2-fake-tag">{box.label}</span>
                  </div>
                )
              ))}

              {/* Grid overlay */}
              <div className="d2-scan-grid" />
            </div>

            <div className="d2-scan-footer">
              Analysing frame by frame for accurate detection
            </div>
          </div>
        )}

        {isDone && (
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
        )}
      </div>
    </div>
  );
}
