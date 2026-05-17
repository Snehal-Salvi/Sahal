import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    googleId: {
      type: String,
      required: true,
      unique: true,
      index: true
    },
    email: {
      type: String,
      required: true,
      lowercase: true,
      trim: true
    },
    name: String,
    picture: String
  },
  {
    timestamps: true
  }
);

export const User = mongoose.model("User", userSchema);
