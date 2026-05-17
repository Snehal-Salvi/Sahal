import jwt from "jsonwebtoken";

const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || "7d";

if (!JWT_SECRET) {
  throw new Error(
    "JWT_SECRET is not set. Generate one with `openssl rand -base64 48` and set it in your environment."
  );
}
if (JWT_SECRET.length < 32) {
  throw new Error(
    "JWT_SECRET is too short. Use at least 32 characters of random data (e.g. `openssl rand -base64 48`)."
  );
}

export function signAppToken(payload) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

export function verifyAppToken(token) {
  return jwt.verify(token, JWT_SECRET);
}
