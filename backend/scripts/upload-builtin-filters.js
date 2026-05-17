// One-shot script: uploads every PNG + JSON manifest under public/filters to
// Cloudinary as raw assets, so the AI service can fetch them with its
// HTTPS-only SSRF guard intact. Re-running this script is idempotent — files
// already in cloudinary-map.json are skipped.
//
// Usage: cd backend && node scripts/upload-builtin-filters.js
//
// Requires CLOUDINARY_* env vars in backend/.env.

import "dotenv/config";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { cloudinary } from "../src/config/cloudinary.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FILTERS_DIR = path.join(__dirname, "../public/filters");
const MAP_PATH = path.join(FILTERS_DIR, "cloudinary-map.json");
const FOLDER = "cartoon-face-filter/builtin-filters";

const map = fs.existsSync(MAP_PATH)
  ? JSON.parse(fs.readFileSync(MAP_PATH, "utf8"))
  : {};

const entries = fs
  .readdirSync(FILTERS_DIR)
  .filter((f) => f !== "cloudinary-map.json")
  .filter((f) => f.endsWith(".png") || f.endsWith(".json"));

for (const filename of entries) {
  if (map[filename]) {
    console.log(`skip (already uploaded): ${filename}`);
    continue;
  }
  const filepath = path.join(FILTERS_DIR, filename);
  // Upload as resource_type "raw" with the full filename (including extension)
  // as public_id so the resulting URL ends in .png / .json. This is what makes
  // <overlay>.png ↔ <overlay>.json work for the AI service's manifest fetch.
  const result = await cloudinary.uploader.upload(filepath, {
    folder: FOLDER,
    public_id: filename,
    resource_type: "raw",
    overwrite: true,
    use_filename: false,
    unique_filename: false
  });
  map[filename] = result.secure_url;
  console.log(`uploaded ${filename} -> ${result.secure_url}`);
}

fs.writeFileSync(MAP_PATH, JSON.stringify(map, null, 2));
console.log(`\nmap written: ${MAP_PATH}`);
