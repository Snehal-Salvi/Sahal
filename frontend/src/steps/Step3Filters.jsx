export default function Step3Filters({
  analysis,
  filterLibrary,
  assignedFilters,
  selectedFaceId,
  applySameFilterToAll,
  sharedFilterId,
  activeFace,
  error,
  setSelectedFaceId,
  setApplySameFilterToAll,
  assignFilter,
  removeFilter,
  handleProcessVideo,
  onAddFilter,
}) {
  return (
    <div className="s3">
      <div className="sh">
        <div className="stag">Step 3</div>
        <h2>Choose filters for each face</h2>
        <p>Select a face tab and assign a cartoon filter. Each person can have a unique style.</p>
      </div>

      {error && <div className="error-msg">{error}</div>}

      <div className="fl-grid">
        {/* ── Left: preview + face tabs + process button ── */}
        <div>
          <div className="prev-card">
            <div className="prev-vid">
              <div className="prev-lbl">Live preview</div>
              {analysis?.representativeFrameDataUrl ? (
                <div className="analysis-preview-s3">
                  <img src={analysis.representativeFrameDataUrl} alt="Preview" />
                  {analysis.faces.map((face) => {
                    const box = face.representativeBox;
                    if (!box) return null;
                    const assignedFilter = filterLibrary.find(
                      (f) => f.id === assignedFilters[face.faceId]
                    );
                    return (
                      <div
                        key={face.faceId}
                        className={`prev-face-box${selectedFaceId === face.faceId ? " active" : ""}`}
                        style={{
                          left: `${box.x * 100}%`,
                          top: `${box.y * 100}%`,
                          width: `${box.width * 100}%`,
                          height: `${box.height * 100}%`,
                        }}
                      >
                        {assignedFilter && (
                          <img
                            src={assignedFilter.previewUrl}
                            alt="filter"
                            className="prev-filter-img"
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div style={{ color: "rgba(255,255,255,0.2)", fontSize: 12 }}>
                  No preview available
                </div>
              )}
            </div>

            {/* Face tabs */}
            <div className="tabs-row">
              {analysis?.faces?.map((face) => (
                <button
                  key={face.faceId}
                  className={`ftab${selectedFaceId === face.faceId ? " on" : ""}`}
                  onClick={() => setSelectedFaceId(face.faceId)}
                >
                  <div className="tdot" />{face.label}
                </button>
              ))}
              <div className="same-tog">
                Same:
                <div
                  className={`tog tog-sm${applySameFilterToAll ? " on" : ""}`}
                  onClick={() => setApplySameFilterToAll((v) => !v)}
                >
                  <div className="knob" />
                </div>
              </div>
            </div>

            {/* Active face bar */}
            {activeFace && (
              <div className="active-bar">
                <div className="abar-inner">
                  <div className="aface">
                    <img
                      src={activeFace.thumbnailDataUrl}
                      alt={activeFace.label}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                  </div>
                  <div>
                    <div className="aname">{activeFace.label}</div>
                    <div className="asub">
                      {assignedFilters[activeFace.faceId]
                        ? filterLibrary
                            .find((f) => f.id === assignedFilters[activeFace.faceId])
                            ?.name?.replace(/\.[^.]+$/, "") || "Filter assigned"
                        : "No filter assigned yet"}
                    </div>
                  </div>
                  <div className="abadge">Active</div>
                </div>
              </div>
            )}

            <div style={{ padding: "10px 12px" }}>
              <button className="proc-btn" onClick={handleProcessVideo}>
                <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
                Process Video Now
              </button>
            </div>
          </div>
        </div>

        {/* ── Right: filter library ── */}
        <div>
          <div className="filt-panel">
            <div className="fp-hdr">
              <div className="fp-title">Filter library</div>
              <div className="fp-cnt">{filterLibrary.length} filters</div>
            </div>

            {filterLibrary.length > 0 ? (
              <div className="fg">
                {filterLibrary.map((filter) => {
                  const isSelected = applySameFilterToAll
                    ? sharedFilterId === filter.id
                    : assignedFilters[selectedFaceId] === filter.id;
                  return (
                    <div
                      key={filter.id}
                      className={`ft${isSelected ? " on" : ""}`}
                      onClick={() => assignFilter(filter.id)}
                    >
                      <div className="ft-face">
                        <img
                          src={filter.previewUrl}
                          alt={filter.name}
                          style={{ width: "100%", height: "100%", objectFit: "contain" }}
                        />
                      </div>
                      <div className="ft-n">{filter.name.replace(/\.[^.]+$/, "")}</div>
                      <button
                        className="ft-remove"
                        onClick={(e) => { e.stopPropagation(); removeFilter(filter.id); }}
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="no-filters-hint">No filters yet — upload a PNG below</div>
            )}

            <div className="uc" onClick={onAddFilter}>
              <div className="uc-i">+</div>
              <div>
                <div className="uc-t">Upload custom filter</div>
                <div className="uc-s">PNG with transparent background</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
