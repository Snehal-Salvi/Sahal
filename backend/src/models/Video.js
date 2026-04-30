import mongoose from "mongoose";

const detectedFaceSchema = new mongoose.Schema(
  {
    faceId: {
      type: String,
      required: true
    },
    label: {
      type: String,
      required: true
    },
    embedding: {
      type: [Number],
      default: []
    },
    representativeBox: {
      x: Number,
      y: Number,
      width: Number,
      height: Number
    }
  },
  {
    _id: false
  }
);

const filterAssignmentSchema = new mongoose.Schema(
  {
    faceId: {
      type: String,
      required: true
    },
    overlayImageUrl: {
      type: String,
      required: true
    }
  },
  {
    _id: false
  }
);

const videoSchema = new mongoose.Schema(
  {
    originalUrl: {
      type: String,
      default: ""
    },
    originalPublicId: {
      type: String,
      default: ""
    },
    processedUrl: String,
    processedPublicId: String,
    overlayImageUrl: String,
    detectedFaces: {
      type: [detectedFaceSchema],
      default: []
    },
    filterAssignments: {
      type: [filterAssignmentSchema],
      default: []
    },
    status: {
      type: String,
      enum: ["uploaded", "queued", "processing", "completed", "failed"],
      default: "uploaded"
    },
    jobId: String,
    error: String
  },
  {
    timestamps: true
  }
);

export const Video = mongoose.model("Video", videoSchema);
