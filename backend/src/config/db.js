import mongoose from "mongoose";

export async function connectDatabase() {
  const uri = process.env.MONGODB_URI;
  if (!uri) {
    throw new Error("MONGODB_URI is not set");
  }
  if (process.env.NODE_ENV === "production") {
    const isTlsScheme = uri.startsWith("mongodb+srv://");
    const hasTlsFlag = /[?&]tls=true/.test(uri) || /[?&]ssl=true/.test(uri);
    if (!isTlsScheme && !hasTlsFlag) {
      throw new Error(
        "MONGODB_URI must use mongodb+srv:// or include tls=true in production"
      );
    }
    if (/\/\/[^:@/]+:?@/.test(uri) || uri.includes("://localhost")) {
      console.warn("[db] MONGODB_URI looks unauthenticated or local — verify before deploying");
    }
  }
  await mongoose.connect(uri);
  console.log("MongoDB connected");
}

