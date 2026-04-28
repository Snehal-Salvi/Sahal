import { useVideoProcessor } from "../hooks/useVideoProcessor";
import Navbar from "./Navbar";
import Step1Upload from "../steps/Step1Upload";
import Step2Detect from "../steps/Step2Detect";
import Step3Filters from "../steps/Step3Filters";
import Step4Processing from "../steps/Step4Processing";
import Step5Output from "../steps/Step5Output";

export default function VideoUploader() {
  const p = useVideoProcessor();

  return (
    <div className="wrap">
      <Navbar step={p.step} goTo={p.goTo} />

      {/* Hidden file inputs shared across steps */}
      <input
        ref={p.fileInputRef}
        type="file"
        accept="video/*"
        style={{ display: "none" }}
        onChange={p.handleVideoFileChange}
      />
      <input
        ref={p.filterInputRef}
        type="file"
        accept="image/png"
        multiple
        style={{ display: "none" }}
        onChange={p.handleFilterLibraryChange}
      />

      {p.step === 1 && (
        <Step1Upload
          onChooseFile={() => p.fileInputRef.current?.click()}
          onDropFile={p.handleVideoFileDrop}
        />
      )}

      {p.step === 2 && (
        <Step2Detect
          videoPreviewUrl={p.videoPreviewUrl}
          videoFile={p.videoFile}
          analysis={p.analysis}
          selectedFaceId={p.selectedFaceId}
          applySameFilterToAll={p.applySameFilterToAll}
          status={p.status}
          error={p.error}
          handleAnalyzeFaces={p.handleAnalyzeFaces}
          setSelectedFaceId={p.setSelectedFaceId}
          setApplySameFilterToAll={p.setApplySameFilterToAll}
          goTo={p.goTo}
        />
      )}

      {p.step === 3 && (
        <Step3Filters
          analysis={p.analysis}
          filterLibrary={p.filterLibrary}
          assignedFilters={p.assignedFilters}
          selectedFaceId={p.selectedFaceId}
          applySameFilterToAll={p.applySameFilterToAll}
          sharedFilterId={p.sharedFilterId}
          activeFace={p.activeFace}
          error={p.error}
          setSelectedFaceId={p.setSelectedFaceId}
          setApplySameFilterToAll={p.setApplySameFilterToAll}
          assignFilter={p.assignFilter}
          removeFilter={p.removeFilter}
          handleProcessVideo={p.handleProcessVideo}
          onAddFilter={() => p.filterInputRef.current?.click()}
        />
      )}

      {p.step === 4 && (
        <Step4Processing
          analysis={p.analysis}
          filterLibrary={p.filterLibrary}
          assignedFilters={p.assignedFilters}
          videoRecord={p.videoRecord}
          status={p.status}
        />
      )}

      {p.step === 5 && (
        <Step5Output
          videoRecord={p.videoRecord}
          analysis={p.analysis}
          filterLibrary={p.filterLibrary}
          assignedFilters={p.assignedFilters}
          goTo={p.goTo}
          handleReset={p.handleReset}
        />
      )}
    </div>
  );
}
