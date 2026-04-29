import "./Step3Filters.css";

function FilterCard({ filter, isSelected, onSelect, onRemove }) {
  return (
    <div
      className={`f3-card${isSelected ? " on" : ""}`}
      onClick={() => onSelect(filter.id)}
    >
      {isSelected && <div className="f3-check">✓</div>}
      <div className="f3-thumb">
        <img src={filter.previewUrl} alt={filter.name} />
      </div>
      <div className="f3-name">{filter.name.replace(/\.[^.]+$/, "")}</div>
      {filter.isBuiltIn ? (
        <div className="f3-builtin">✦</div>
      ) : (
        <button
          className="f3-rm"
          onClick={(e) => { e.stopPropagation(); onRemove(filter.id); }}
        >×</button>
      )}
    </div>
  );
}

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
  const builtinFilters = filterLibrary.filter((f) => f.isBuiltIn);
  const customFilters = filterLibrary.filter((f) => !f.isBuiltIn);

  function isSelected(filterId) {
    return applySameFilterToAll
      ? sharedFilterId === filterId
      : assignedFilters[selectedFaceId] === filterId;
  }

  const currentFilterId = applySameFilterToAll ? sharedFilterId : assignedFilters[selectedFaceId];
  const currentFilter = filterLibrary.find((f) => f.id === currentFilterId);

  const allAssigned = analysis?.faces?.every((f) => assignedFilters[f.faceId]);

  return (
    <div className="s3">
      {error && <div className="error-msg">{error}</div>}

      {/* ── Preview ── */}
      <div className="f3-preview">
        {analysis?.representativeFrameDataUrl ? (
          <div className="f3-prev-inner">
            <img src={analysis.representativeFrameDataUrl} alt="Preview" className="f3-prev-img" />
            {analysis.faces.map((face) => {
              const box = face.representativeBox;
              if (!box) return null;
              const fFilter = filterLibrary.find((f) => f.id === assignedFilters[face.faceId]);
              return (
                <div
                  key={face.faceId}
                  className={`f3-prev-box${selectedFaceId === face.faceId ? " active" : ""}`}
                  style={{
                    left: `${box.x * 100}%`,
                    top: `${box.y * 100}%`,
                    width: `${box.width * 100}%`,
                    height: `${box.height * 100}%`,
                  }}
                  onClick={() => setSelectedFaceId(face.faceId)}
                >
                  {fFilter && (
                    <img src={fFilter.previewUrl} alt="filter" className="f3-prev-filter" />
                  )}
                </div>
              );
            })}
            <div className="f3-prev-label">Live Preview</div>
          </div>
        ) : (
          <div className="f3-prev-empty">No preview available</div>
        )}
      </div>

      {/* ── Face selector ── */}
      <div className="f3-face-bar">
        <div className="f3-face-tabs">
          {analysis?.faces?.map((face) => {
            const assigned = assignedFilters[face.faceId];
            return (
              <button
                key={face.faceId}
                className={`f3-ftab${selectedFaceId === face.faceId ? " on" : ""}${assigned ? " done" : ""}`}
                onClick={() => setSelectedFaceId(face.faceId)}
              >
                <div className="f3-ftab-av">
                  <img src={face.thumbnailDataUrl} alt={face.label} />
                  {assigned && <div className="f3-ftab-ck">✓</div>}
                </div>
                <span>{face.label}</span>
              </button>
            );
          })}
        </div>

        {analysis?.faces?.length > 1 && (
          <div className="f3-same-tog">
            <span>Same for all</span>
            <div
              className={`tog tog-sm${applySameFilterToAll ? " on" : ""}`}
              onClick={() => setApplySameFilterToAll((v) => !v)}
            >
              <div className="knob" />
            </div>
          </div>
        )}
      </div>

      {/* ── Active face info ── */}
      {activeFace && (
        <div className="f3-active-face">
          <div className="f3-af-av">
            <img src={activeFace.thumbnailDataUrl} alt={activeFace.label} />
          </div>
          <div className="f3-af-info">
            <div className="f3-af-name">{activeFace.label}</div>
            <div className="f3-af-filter">
              {currentFilter
                ? currentFilter.name.replace(/\.[^.]+$/, "")
                : "No filter chosen yet"}
            </div>
          </div>
          <div className="f3-af-badge">{applySameFilterToAll ? "All faces" : "Active"}</div>
        </div>
      )}

      {/* ── Filter library ── */}
      <div className="f3-lib">
        {builtinFilters.length > 0 && (
          <div className="f3-section">
            <div className="f3-sec-hdr">
              <span className="f3-sec-title">Built-in filters</span>
              <span className="f3-sec-badge builtin">✦ curated</span>
            </div>
            <div className="f3-grid">
              {builtinFilters.map((f) => (
                <FilterCard key={f.id} filter={f} isSelected={isSelected(f.id)} onSelect={assignFilter} onRemove={removeFilter} />
              ))}
            </div>
          </div>
        )}

        {customFilters.length > 0 && (
          <div className="f3-section">
            <div className="f3-sec-hdr">
              <span className="f3-sec-title">Your uploads</span>
              <span className="f3-sec-badge">{customFilters.length}</span>
            </div>
            <div className="f3-grid">
              {customFilters.map((f) => (
                <FilterCard key={f.id} filter={f} isSelected={isSelected(f.id)} onSelect={assignFilter} onRemove={removeFilter} />
              ))}
            </div>
          </div>
        )}

        {filterLibrary.length === 0 && (
          <div className="f3-loading">Loading filters…</div>
        )}

        <button className="f3-upload" onClick={onAddFilter}>
          <span className="f3-upload-plus">+</span>
          <div>
            <div className="f3-upload-title">Upload custom filter</div>
            <div className="f3-upload-sub">Transparent PNG only</div>
          </div>
        </button>
      </div>

      {/* ── Process button ── */}
      <button
        className={`f3-process${allAssigned ? "" : " dim"}`}
        onClick={handleProcessVideo}
      >
        <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
        Process Video
      </button>
    </div>
  );
}
