import { Router } from "express";
import { existsSync, readFileSync, readdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join, extname, basename } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FILTERS_DIR = join(__dirname, "../../public/filters");
const CLOUDINARY_MAP_PATH = join(FILTERS_DIR, "cloudinary-map.json");

const cloudinaryMap = existsSync(CLOUDINARY_MAP_PATH)
  ? JSON.parse(readFileSync(CLOUDINARY_MAP_PATH, "utf8"))
  : {};

const router = Router();

// Pretty-print filenames: "nobita-face.png" → "Nobita Face"
function toDisplayName(filename) {
  return basename(filename, extname(filename))
    .replace(/[-_]/g, " ")
    .replace(/^Ar /, "")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function readFilterManifest(filename) {
  const manifestPath = join(FILTERS_DIR, `${basename(filename, extname(filename))}.json`);
  if (!existsSync(manifestPath)) return {};

  try {
    return JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    return {};
  }
}

router.get("/", (req, res) => {
  let files;
  try {
    files = readdirSync(FILTERS_DIR).filter((f) => f.toLowerCase().endsWith(".png"));
  } catch {
    return res.json([]);
  }

  const includeClassic = req.query.includeClassic === "1";
  const filters = files
    .map((filename) => {
      const manifest = readFilterManifest(filename);
      const cloudUrl = cloudinaryMap[filename];
      if (!cloudUrl) return null;
      return {
        id: `builtin-${filename}`,
        name: manifest.title || toDisplayName(filename),
        filename,
        url: cloudUrl,
        category: manifest.category || "Built-in filter",
        description: manifest.description || "",
        isAR: Boolean(manifest.ar_ready),
        filterType: manifest.filter_type || manifest.coverage || "accessory",
        character: manifest.character || "",
        blendMode: manifest.blend_mode || "over",
        revealEyes: manifest.reveal_eyes !== false,
        revealMouth: manifest.reveal_mouth !== false
      };
    })
    .filter(Boolean)
    .filter((filter) => includeClassic || filter.isAR)
    .sort((a, b) => Number(b.isAR) - Number(a.isAR) || a.name.localeCompare(b.name));

  res.json(filters);
});

export default router;
