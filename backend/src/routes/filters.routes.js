import { Router } from "express";
import { readdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join, extname, basename } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FILTERS_DIR = join(__dirname, "../../public/filters");

const router = Router();

// Pretty-print filenames: "nobita-face.png" → "Nobita Face"
function toDisplayName(filename) {
  return basename(filename, extname(filename))
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

router.get("/", (req, res) => {
  let files;
  try {
    files = readdirSync(FILTERS_DIR).filter((f) => f.toLowerCase().endsWith(".png"));
  } catch {
    return res.json([]);
  }

  const baseUrl = `${req.protocol}://${req.get("host")}`;
  const filters = files.map((filename) => ({
    id: `builtin-${filename}`,
    name: toDisplayName(filename),
    filename,
    url: `${baseUrl}/filters/${filename}`,
  }));

  res.json(filters);
});

export default router;
