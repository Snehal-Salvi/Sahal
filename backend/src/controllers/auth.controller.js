import { OAuth2Client } from "google-auth-library";
import { User } from "../models/User.js";
import { Video } from "../models/Video.js";
import { destroyAsset } from "../services/cloudinary.service.js";
import { signAppToken } from "../utils/jwt.js";

const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;

if (!GOOGLE_CLIENT_ID) {
  console.warn("[auth] GOOGLE_CLIENT_ID is not set — Google sign-in will fail");
}

const googleClient = new OAuth2Client(GOOGLE_CLIENT_ID);

export async function googleSignIn(req, res) {
  const { credential } = req.body;

  if (!credential || typeof credential !== "string") {
    return res.status(400).json({ message: "credential is required" });
  }

  let ticket;
  try {
    ticket = await googleClient.verifyIdToken({
      idToken: credential,
      audience: GOOGLE_CLIENT_ID
    });
  } catch {
    return res.status(401).json({ message: "Invalid Google credential" });
  }

  const payload = ticket.getPayload();
  if (!payload?.sub || !payload?.email_verified) {
    return res.status(401).json({ message: "Google account not verified" });
  }

  const user = await User.findOneAndUpdate(
    { googleId: payload.sub },
    {
      googleId: payload.sub,
      email: payload.email,
      name: payload.name,
      picture: payload.picture
    },
    { upsert: true, new: true, setDefaultsOnInsert: true }
  );

  const token = signAppToken({ sub: user.id, email: user.email });

  return res.json({
    token,
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      picture: user.picture
    }
  });
}

export async function deleteAccount(req, res) {
  const userId = req.user.id;

  const videos = await Video.find({ ownerId: userId });
  for (const video of videos) {
    if (video.originalPublicId) await destroyAsset(video.originalPublicId, "video");
    if (video.processedPublicId) await destroyAsset(video.processedPublicId, "video");
  }
  await Video.deleteMany({ ownerId: userId });
  await User.deleteOne({ _id: userId });

  return res.json({ message: "Account and associated data deleted" });
}
