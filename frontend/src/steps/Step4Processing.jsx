import ProcessingStep from "../components/ProcessingStep";

export default function Step4Processing({ analysis, filterLibrary, assignedFilters, videoRecord, status }) {
  const procPct =
    status === "completed" ? "100%" : status === "processing" ? "62%" : "15%";

  return (
    <div className="s4">
      <div className="proc-center">
        <div className="proc-hero">
          <div className="spin"><div className="spin-in">🎥</div></div>
          <h2>Processing your video</h2>
          <p>
            Applying expression-aware cartoon filters frame by frame using
            MediaPipe Face Mesh and OpenCV.
          </p>
        </div>

        <div className="steps-card">
          <ProcessingStep
            icon="✅" iconClass="id"
            name="Face detection & landmark mapping"
            desc={`${analysis?.faces?.length || 0} faces · 468 landmarks`}
            state="done"
          />
          <ProcessingStep
            icon="✅" iconClass="id"
            name="Expression coefficient extraction"
            desc="Blink, smile, mouth & eyebrow tracking complete"
            state="done"
          />
          <ProcessingStep
            icon="🎭" iconClass="ia"
            name="Applying cartoon filters with mesh warp"
            desc="Syncing filter deformation to facial expressions..."
            state={status === "completed" ? "done" : "active"}
          />
          <ProcessingStep
            icon="🎥" iconClass="ip"
            name="FFmpeg video composition"
            desc="Rendering final output with original audio"
            state={status === "completed" ? "done" : "wait"}
          />
          <ProcessingStep
            icon="☁️" iconClass="ip"
            name="Uploading to Cloudinary"
            desc="Generating secure download & share link"
            state={status === "completed" ? "done" : "wait"}
          />
        </div>

        <div className="prog-card">
          <div className="prog-top">
            <span className="prog-lbl">Overall progress</span>
            <span className="prog-pct">{procPct}</span>
          </div>
          <div className="pbg"><div className="pfill" style={{ width: procPct }} /></div>
          <div className="pmeta">
            <span>{status === "completed" ? "Done!" : "Processing…"}</span>
            <span>Job: {videoRecord?.jobId || "—"}</span>
          </div>

          {analysis?.faces?.length > 0 && (
            <div className="fp2">
              {analysis.faces.map((face) => {
                const filter = filterLibrary.find((f) => f.id === assignedFilters[face.faceId]);
                return (
                  <div key={face.faceId} className="fpi">
                    <div className="fpi-top">
                      <div className="fpi-f">
                        <img
                          src={face.thumbnailDataUrl}
                          alt={face.label}
                          style={{ width: "100%", height: "100%", objectFit: "cover" }}
                        />
                      </div>
                      <div className="fpi-n">
                        {face.label}
                        {filter ? ` · ${filter.name.replace(/\.[^.]+$/, "")}` : ""}
                      </div>
                    </div>
                    <div className="fpi-bg">
                      <div
                        className="fpi-fill"
                        style={{ width: status === "completed" ? "100%" : "60%" }}
                      />
                    </div>
                    <div className="fpi-pct">{status === "completed" ? "100%" : "60%"}</div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
