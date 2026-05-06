import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Reuse the live AR texture function so the static preview PNG matches
# what the renderer actually draws each frame.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-service"))
from app.processor import _porcelain_cup_texture  # noqa: E402


OUT_DIR = Path(__file__).resolve().parents[1] / "backend" / "public" / "filters"
POOKIE_SOURCE = Path("/Users/snehalashoksalvi/Downloads/pookie.png")
SIZE = 768


def rgba(hex_color, alpha=255):
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
        alpha,
    )


def cv_color(hex_color, alpha=255):
    r, g, b, a = rgba(hex_color, alpha)
    return (b, g, r, a)


def canvas():
    return np.zeros((SIZE, SIZE, 4), dtype=np.uint8)


def blur_alpha(img, k=25):
    alpha = cv2.GaussianBlur(img[:, :, 3], (0, 0), sigmaX=k)
    img[:, :, 3] = np.maximum(img[:, :, 3], alpha)
    return img


def ellipse(img, center, axes, angle, color, thickness=-1):
    cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness, lineType=cv2.LINE_AA)


def line(img, p1, p2, color, thickness=6):
    cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def poly(img, points, color, thickness=-1):
    cv2.fillPoly(img, [np.array(points, dtype=np.int32)], color, lineType=cv2.LINE_AA)


def save_asset(slug, img, manifest):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug}.png"
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)).save(path)
    with (OUT_DIR / f"{slug}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "title": manifest["title"],
                "category": "AR filter",
                "ar_ready": True,
                "blend_mode": manifest.get("blend_mode", "over"),
                "reveal_eyes": manifest.get("reveal_eyes", True),
                "reveal_mouth": manifest.get("reveal_mouth", True),
                "strip_background": False,
                "coverage": manifest.get("coverage", "accessory"),
                "filter_type": manifest.get("filter_type", manifest.get("coverage", "accessory")),
                "character": manifest.get("character", ""),
                "description": manifest["description"],
            },
            f,
            indent=2,
        )
        f.write("\n")


def aurora_glow():
    img = canvas()
    glow = canvas()
    ellipse(glow, (384, 214), (152, 34), 0, cv_color("#22d3ee", 80), 8)
    ellipse(glow, (260, 432), (48, 18), -18, cv_color("#f472b6", 80), -1)
    ellipse(glow, (508, 432), (48, 18), 18, cv_color("#f472b6", 80), -1)
    blur_alpha(glow, 10)
    img[:] = np.maximum(img, glow)

    ellipse(img, (384, 214), (150, 32), 0, cv_color("#22d3ee", 205), 5)
    ellipse(img, (384, 224), (104, 18), 0, cv_color("#f472b6", 155), 4)
    line(img, (270, 462), (318, 438), cv_color("#facc15", 190), 7)
    line(img, (498, 438), (546, 462), cv_color("#facc15", 190), 7)
    line(img, (294, 486), (334, 464), cv_color("#34d399", 150), 5)
    line(img, (474, 464), (514, 486), cv_color("#34d399", 150), 5)
    save_asset(
        "ar-aurora-glow",
        img,
        {
            "title": "Aurora Glow",
            "blend_mode": "over",
            "description": "Neon forehead halo and cheek accents that track the face without covering skin.",
        },
    )


def cyber_visor():
    img = canvas()
    line(img, (238, 262), (530, 262), cv_color("#67e8f9", 220), 5)
    line(img, (248, 278), (330, 278), cv_color("#f472b6", 180), 4)
    line(img, (438, 278), (520, 278), cv_color("#f472b6", 180), 4)
    for x in [268, 308, 460, 500]:
        line(img, (x, 244), (x + 16, 292), cv_color("#ffffff", 88), 3)
    line(img, (220, 456), (294, 426), cv_color("#67e8f9", 190), 8)
    line(img, (548, 456), (474, 426), cv_color("#67e8f9", 190), 8)
    ellipse(img, (384, 526), (74, 18), 0, cv_color("#22d3ee", 100), 4)
    save_asset(
        "ar-cyber-visor",
        img,
        {
            "title": "Cyber Visor",
            "blend_mode": "over",
            "description": "Lightweight HUD lines and cheek strips that leave eyes, glasses and lips untouched.",
        },
    )


def comic_hero():
    img = canvas()
    poly(img, [(240, 238), (316, 194), (384, 226), (452, 194), (528, 238), (466, 270), (384, 250), (302, 270)], cv_color("#facc15", 230))
    line(img, (246, 240), (308, 262), cv_color("#ef4444", 210), 6)
    line(img, (522, 240), (460, 262), cv_color("#ef4444", 210), 6)
    line(img, (232, 496), (292, 454), cv_color("#facc15", 220), 9)
    line(img, (536, 496), (476, 454), cv_color("#facc15", 220), 9)
    line(img, (246, 526), (294, 492), cv_color("#ef4444", 170), 5)
    line(img, (522, 526), (474, 492), cv_color("#ef4444", 170), 5)
    save_asset(
        "ar-comic-hero",
        img,
        {
            "title": "Comic Hero",
            "blend_mode": "over",
            "description": "Comic crown and cheek bolts that add character without replacing the face.",
        },
    )


def silver_star():
    img = canvas()
    for cx, cy, r in [(224, 238, 30), (544, 238, 30), (188, 420, 22), (580, 420, 22), (384, 170, 26), (304, 536, 18), (464, 536, 18)]:
        pts = []
        for step in range(10):
            angle = -np.pi / 2 + step * np.pi / 5
            radius = r if step % 2 == 0 else r * 0.42
            pts.append((int(cx + np.cos(angle) * radius), int(cy + np.sin(angle) * radius)))
        poly(img, pts, cv_color("#f8fafc", 215))
    line(img, (232, 560), (536, 560), cv_color("#a78bfa", 140), 5)
    save_asset(
        "ar-silver-star",
        img,
        {
            "title": "Silver Star",
            "blend_mode": "over",
            "description": "Small glam stars and a chin sparkle line that track with the face.",
        },
    )


def festival_tilak():
    img = canvas()
    ellipse(img, (384, 236), (34, 92), 0, cv_color("#ef4444", 220), -1)
    ellipse(img, (384, 236), (20, 58), 0, cv_color("#facc15", 210), -1)
    for offset, color in [(-128, "#22c55e"), (-82, "#f97316"), (82, "#f97316"), (128, "#22c55e")]:
        ellipse(img, (384 + offset, 296), (28, 52), 0, cv_color(color, 185), -1)
    line(img, (236, 514), (304, 548), cv_color("#ef4444", 150), 8)
    line(img, (532, 514), (464, 548), cv_color("#ef4444", 150), 8)
    save_asset(
        "ar-festival-tilak",
        img,
        {
            "title": "Festival Tilak",
            "blend_mode": "over",
            "description": "Festive forehead and cheek accents designed to leave facial expressions fully visible.",
        },
    )


def pookie_bot():
    if not POOKIE_SOURCE.exists():
        return

    rgb = np.array(Image.open(POOKIE_SOURCE).convert("RGB"))
    f = rgb.astype(np.float32)
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = mx - mn

    # Keep the robot shell/accessories, drop the beige face center and drawn features.
    teal_shell = (rgb[:, :, 1] > 120) & (rgb[:, :, 2] > 120) & (rgb[:, :, 0] < 120)
    orange = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 85) & (rgb[:, :, 1] < 190) & (rgb[:, :, 2] < 80)
    yellow = (rgb[:, :, 0] > 190) & (rgb[:, :, 1] > 145) & (rgb[:, :, 2] < 90)
    metal = (sat < 42) & (mx > 65) & (mx < 235)
    dark_panel = (mx < 72)
    keep = teal_shell | orange | yellow | metal | dark_panel

    # Remove neutral checkerboard background and tiny noise.
    background = (sat < 12) & (mx > 205)
    keep &= ~background
    mask = keep.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > 400:
            cleaned[labels == label] = 255
    alpha = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=1.0)
    alpha[cleaned == 255] = 255
    alpha[alpha < 18] = 0
    rgba_img = np.dstack([rgb, alpha]).astype(np.uint8)

    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return
    pad = 36
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    rgba_img = rgba_img[y1:y2, x1:x2]

    target = SIZE
    scale = min((target - 24) / rgba_img.shape[1], (target - 24) / rgba_img.shape[0])
    nw = max(1, int(round(rgba_img.shape[1] * scale)))
    nh = max(1, int(round(rgba_img.shape[0] * scale)))
    resized = cv2.resize(rgba_img, (nw, nh), interpolation=cv2.INTER_AREA)
    img = np.zeros((target, target, 4), dtype=np.uint8)
    ox = (target - nw) // 2
    oy = (target - nh) // 2
    img[oy:oy + nh, ox:ox + nw] = resized

    # The source is a full robot face. For AR, keep it as a helmet frame and
    # remove the center face panel so the real person supplies eyes and lips.
    pil = Image.fromarray(img)
    alpha = pil.getchannel("A")
    cutout = Image.new("L", pil.size, 0)
    draw = ImageDraw.Draw(cutout)
    draw.ellipse((128, 202, 640, 594), fill=255)
    cutout = cutout.filter(ImageFilter.GaussianBlur(5))
    alpha = Image.composite(Image.new("L", pil.size, 0), alpha, cutout)
    pil.putalpha(alpha)
    img = np.array(pil)

    save_asset(
        "ar-pookie",
        cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA),
        {
            "title": "Pookie Bot",
            "blend_mode": "over",
            "description": "Pookie robot helmet, antenna and side pods only; the human face stays visible for blinks and speech.",
        },
    )


def porcelain_cup():
    # The shipped porcelain cup is a hand-curated photoreal asset, NOT a
    # procedural draw. Skip if a real asset is already on disk so re-running
    # this script doesn't clobber it. To regenerate the procedural fallback,
    # delete ar-porcelain-cup.png first.
    target = OUT_DIR / "ar-porcelain-cup.png"
    if target.exists():
        print(f"[skip] porcelain_cup: keeping existing artist asset at {target}")
        return

    img = _porcelain_cup_texture(
        {
            "blink_left": 0.0,
            "blink_right": 0.0,
            "smile": 0.0,
            "mouth_open": 0.0,
            "yaw": 0.5,
        },
        size=SIZE,
    )
    save_asset(
        "ar-porcelain-cup",
        img,
        {
            "title": "Porcelain Cup",
            "blend_mode": "over",
            "coverage": "character_mask",
            "filter_type": "character_mask",
            "character": "porcelain_cup",
            "reveal_eyes": False,
            "reveal_mouth": False,
            "description": "Full character mask with synthetic blinking eyes and lips driven by face expressions.",
        },
    )


def main():
    porcelain_cup()
    aurora_glow()
    cyber_visor()
    comic_hero()
    silver_star()
    festival_tilak()
    pookie_bot()


if __name__ == "__main__":
    main()
