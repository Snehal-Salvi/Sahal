import ProcessingStep from "../components/ProcessingStep";
import "./Step4Processing.css";

export default function Step4Processing({ analysis, filterLibrary, assignedFilters, videoRecord, status }) {
  const procPct =
    status === "completed" ? "100%" : status === "processing" ? "62%" : "15%";
  const isComplete = status === "completed";
  const statusText = isComplete ? "Your video is ready" : "Creating your video...";
  const friendlyStatus =
    status === "queued" ? "Getting ready" :
    status === "processing" ? "Creating" :
    isComplete ? "Ready" :
    status === "failed" ? "Needs attention" :
    "Starting";

  return (
    <div className="s4-wrap">
      <div className="s4-left">
        <div className="proc-hero">
          <div className="spin"><div className="spin-in">S</div></div>
          <h2>{statusText}</h2>
          {!isComplete && (
            <div className="proc-status">
              <span className="proc-status-dot" />
              Making your edits shine
            </div>
          )}
          <p>
            We are adding your chosen masks, keeping the faces matched, and
            getting the final video ready for you.
          </p>
        </div>

        <div className="steps-card">
          <ProcessingStep
            icon="✅" iconClass="id"
            name="Faces matched"
            desc={`${analysis?.faces?.length || 0} ${analysis?.faces?.length === 1 ? "person" : "people"} ready`}
            state="done"
          />
          <ProcessingStep
            icon="🎭" iconClass="id"
            name="Choices saved"
            desc="Each face has its selected mask"
            state="done"
          />
          <ProcessingStep
            icon="🎥" iconClass="ia"
            name="Adding the look"
            desc="Your masks are being placed on the right faces"
            state={isComplete ? "done" : "active"}
          />
          <ProcessingStep
            icon="☁️" iconClass="ip"
            name="Polishing the video"
            desc="Keeping the motion smooth and the sound in place"
            state={isComplete ? "done" : "wait"}
          />
          <ProcessingStep
            icon="⏬️" iconClass="ip"
            name="Preparing download"
            desc="Your finished video will appear on the next screen"
            state={isComplete ? "done" : "wait"}
          />
        </div>

        <div className="prog-card">
          <div className="prog-top">
            <span className="prog-lbl">Overall progress</span>
            <span className="prog-pct">{procPct}</span>
          </div>
          <div className="pbg"><div className="pfill" style={{ width: procPct }} /></div>
          <div className="pmeta">
            <span>{isComplete ? "Done" : "Almost there"}</span>
            <span>{friendlyStatus}</span>
          </div>
        </div>
      </div>

      <div className="s4-right">
        <div className="s4-preview">
          {analysis?.representativeFrameDataUrl ? (
            <div className="s4-prev-stage">
              <img src={analysis.representativeFrameDataUrl} alt="Preview" className="s4-prev-img" />
              <div className="s4-prev-label">{isComplete ? "Ready" : "Preview"}</div>
            </div>
          ) : (
            <div className="s4-prev-empty">Preview will appear here</div>
          )}
        </div>

        {analysis?.faces?.length > 0 && (
          <div className="fp2">
            {analysis.faces.map((face) => {
              const filter = filterLibrary.find((f) => f.id === assignedFilters[face.faceId]);
              return (
                <div key={face.faceId} className="fpi">
                  <div className="fpi-top">
                    <div className="fpi-f">
                      <img src={face.thumbnailDataUrl} alt={face.label} />
                    </div>
                    <div className="fpi-copy">
                      <div className="fpi-n">{face.label}</div>
                      <div className="fpi-filter">
                        {filter ? filter.name.replace(/\.[^.]+$/, "") : "Mask selected"}
                      </div>
                    </div>
                  </div>
                  <div className="fpi-bg">
                    <div className="fpi-fill" style={{ width: isComplete ? "100%" : "60%" }} />
                  </div>
                  <div className="fpi-pct">{isComplete ? "Ready" : "In progress"}</div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
