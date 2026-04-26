export default function Step1Upload({ onChooseFile }) {
  return (
    <div className="s1">
      <div className="hero">
        <div className="tag">AI face filter studio</div>
        <h1>Turn faces into<br /><em>cartoon magic</em></h1>
        <p>
          Upload your video, detect faces automatically, and apply stunning
          expression-aware cartoon filters to each person individually.
        </p>
      </div>

      <div className="upload-box" onClick={onChooseFile}>
        <div className="u-icon">
          <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" /></svg>
        </div>
        <h3>Drop your video here</h3>
        <p>Drag and drop or click to browse your files<br />MP4, MOV, AVI, WEBM · Max 250MB</p>
        <div className="btn-pk" onClick={(e) => { e.stopPropagation(); onChooseFile(); }}>
          <svg style={{ width: 14, height: 14, fill: "#0a0a0a" }} viewBox="0 0 24 24">
            <path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z" />
          </svg>
          Choose Video
        </div>
        <div className="chips">
          <span className="chip">MP4</span>
          <span className="chip">MOV</span>
          <span className="chip">AVI</span>
          <span className="chip">WEBM</span>
          <span className="chip">Up to 250MB</span>
        </div>
      </div>

      <div className="feat3">
        <div className="fc">
          <div className="fc-i">
            <svg viewBox="0 0 24 24">
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
            </svg>
          </div>
          <h4>Multi-face detection</h4>
          <p>Every face found and tracked automatically with unique IDs.</p>
        </div>
        <div className="fc">
          <div className="fc-i">
            <svg viewBox="0 0 24 24">
              <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z" />
            </svg>
          </div>
          <h4>Expression sync</h4>
          <p>Filters follow blinks, smiles and mouth using 468 landmarks.</p>
        </div>
        <div className="fc">
          <div className="fc-i">
            <svg viewBox="0 0 24 24">
              <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z" />
            </svg>
          </div>
          <h4>Per-face control</h4>
          <p>Different filters for each person, or same for all in one tap.</p>
        </div>
      </div>
    </div>
  );
}
