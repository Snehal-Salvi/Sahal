import base64
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests


mp_face_mesh = mp.solutions.face_mesh

EPSILON = 1e-6
ANALYSIS_SAMPLE_FRAMES = 2      # 2 frames is enough for stable face ID
MAX_ANALYSIS_FRAME_WIDTH = 960  # preview resize
MAX_ANALYSIS_DETECT_WIDTH = 480 # MediaPipe input resize (smaller = much faster)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]
LEFT_EYE_RING = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_RING = [263, 387, 385, 362, 380, 373]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
MOUTH_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
MOUTH_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]

# Full eye contour regions (used for always-on mask transparency)
LEFT_EYE_FULL = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # eyebrow
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
]
RIGHT_EYE_FULL = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    # eyebrow
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
]
DESCRIPTOR_KEYPOINTS = [10, 152, 234, 454, 33, 133, 159, 145, 263, 362, 386, 374, 4, 61, 291, 13, 14]



@dataclass(frozen=True)
class ExpressionTuning:
    blink_ratio_min: float = 0.02
    blink_ratio_max: float = 0.12
    mouth_open_min: float = 0.01
    mouth_open_max: float = 0.18
    smile_width_min: float = 0.26
    smile_width_max: float = 0.42
    smile_lift_min: float = 0.02
    smile_lift_max: float = 0.11
    brow_raise_min: float = 0.035
    brow_raise_max: float = 0.095
    smoothing_alpha: float = 0.35
    max_num_faces: int = 5


@dataclass
class CartoonStyle:
    texture: np.ndarray
    filter_landmarks: Optional[np.ndarray] = None   # (468, 2) face pts in filter PNG
    filter_triangles: Optional[List[Tuple[int, int, int]]] = None  # Delaunay on above


@dataclass
class ExpressionState:
    values: Dict[str, float] = field(default_factory=dict)

    def smooth(self, current: Dict[str, float], alpha: float) -> Dict[str, float]:
        if not self.values:
            self.values = current.copy()
            return self.values.copy()

        for key, value in current.items():
            previous = self.values.get(key, value)
            self.values[key] = previous + alpha * (value - previous)

        return self.values.copy()


@dataclass
class LandmarkSmoother:
    """Per-face EMA smoother for the raw 468×2 landmark coordinate array."""
    smoothed: Optional[np.ndarray] = None

    def update(self, points: np.ndarray, alpha: float) -> np.ndarray:
        if self.smoothed is None:
            self.smoothed = points.copy()
        else:
            self.smoothed = self.smoothed + alpha * (points - self.smoothed)
        return self.smoothed.copy()

    def reset(self) -> None:
        self.smoothed = None


@dataclass
class FaceDetection:
    points: np.ndarray
    bbox: Tuple[int, int, int, int]
    normalized_box: Dict[str, float]
    center: np.ndarray
    size: float
    descriptor: np.ndarray
    thumbnail_data_url: Optional[str] = None


@dataclass
class FaceProfile:
    face_id: str
    label: str
    descriptor: np.ndarray
    center: Optional[np.ndarray] = None
    size: float = 0.0
    last_seen_frame: int = -1
    hits: int = 0
    best_area: float = 0.0
    thumbnail_data_url: Optional[str] = None
    representative_box: Optional[Dict[str, float]] = None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def remap(value: float, source_min: float, source_max: float) -> float:
    return clamp((value - source_min) / max(source_max - source_min, EPSILON), 0.0, 1.0)


def load_expression_tuning() -> ExpressionTuning:
    return ExpressionTuning(
        blink_ratio_min=env_float("CARTOON_BLINK_RATIO_MIN", 0.02),
        blink_ratio_max=env_float("CARTOON_BLINK_RATIO_MAX", 0.12),
        mouth_open_min=env_float("CARTOON_MOUTH_OPEN_MIN", 0.01),
        mouth_open_max=env_float("CARTOON_MOUTH_OPEN_MAX", 0.18),
        smile_width_min=env_float("CARTOON_SMILE_WIDTH_MIN", 0.26),
        smile_width_max=env_float("CARTOON_SMILE_WIDTH_MAX", 0.42),
        smile_lift_min=env_float("CARTOON_SMILE_LIFT_MIN", 0.02),
        smile_lift_max=env_float("CARTOON_SMILE_LIFT_MAX", 0.11),
        brow_raise_min=env_float("CARTOON_BROW_RAISE_MIN", 0.035),
        brow_raise_max=env_float("CARTOON_BROW_RAISE_MAX", 0.095),
        smoothing_alpha=clamp(env_float("CARTOON_EXPRESSION_SMOOTHING", 0.35), 0.0, 1.0),
        max_num_faces=max(1, env_int("CARTOON_MAX_FACES", 5)),
    )


def download_file(url: str, destination: str) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as file_handle:
            shutil.copyfileobj(response.raw, file_handle)


def landmark_to_point(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def landmarks_to_points(face_landmarks, width: int, height: int) -> np.ndarray:
    return np.array(
        [landmark_to_point(landmark, width, height) for landmark in face_landmarks.landmark],
        dtype=np.float32,
    )


def encode_image_data_url(image: np.ndarray, extension: str = ".jpg") -> str:
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Image encoding failed")
    mime_type = "image/jpeg" if extension == ".jpg" else "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def resize_for_preview(frame: np.ndarray, max_width: int = MAX_ANALYSIS_FRAME_WIDTH) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= max_width:
        return frame
    scale = max_width / float(frame_width)
    return cv2.resize(frame, (max_width, max(1, int(round(frame_height * scale)))), interpolation=cv2.INTER_AREA)


def bbox_from_points(points: np.ndarray, frame_shape: Tuple[int, int, int], padding_ratio: float = 0.14) -> Tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    size = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1.0)
    padding = size * padding_ratio

    x1 = int(clamp(min_xy[0] - padding, 0, frame_width - 1))
    y1 = int(clamp(min_xy[1] - padding, 0, frame_height - 1))
    x2 = int(clamp(max_xy[0] + padding, x1 + 1, frame_width))
    y2 = int(clamp(max_xy[1] + padding, y1 + 1, frame_height))
    return x1, y1, x2, y2


def normalize_box(box: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box
    return {
        "x": round(x1 / max(frame_width, 1), 6),
        "y": round(y1 / max(frame_height, 1), 6),
        "width": round((x2 - x1) / max(frame_width, 1), 6),
        "height": round((y2 - y1) / max(frame_height, 1), 6),
    }


def crop_box(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def compute_face_descriptor(frame: np.ndarray, points: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    face_width = max(float(np.linalg.norm(points[454] - points[234])), 1.0)
    face_height = max(float(np.linalg.norm(points[152] - points[10])), 1.0)
    face_center = (points[10] + points[152] + points[234] + points[454]) * 0.25

    geometry = points[DESCRIPTOR_KEYPOINTS].copy()
    geometry[:, 0] = (geometry[:, 0] - face_center[0]) / face_width
    geometry[:, 1] = (geometry[:, 1] - face_center[1]) / face_height
    geometry_descriptor = geometry.flatten()

    crop = crop_box(frame, box)
    if crop.size == 0:
        color_descriptor = np.zeros(24, dtype=np.float32)
    else:
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv_crop], [0, 1], None, [6, 4], [0, 180, 0, 256]).flatten()
        histogram_sum = float(histogram.sum())
        if histogram_sum > 0:
            histogram /= histogram_sum
        color_descriptor = histogram.astype(np.float32)

    descriptor = np.concatenate([geometry_descriptor.astype(np.float32), color_descriptor], axis=0)
    norm = float(np.linalg.norm(descriptor))
    if norm > EPSILON:
        descriptor /= norm
    return descriptor


def create_face_thumbnail(frame: np.ndarray, box: Tuple[int, int, int, int]) -> str:
    crop = crop_box(frame, box)
    if crop.size == 0:
        crop = np.zeros((96, 96, 3), dtype=np.uint8)
    thumbnail = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    return encode_image_data_url(thumbnail, ".jpg")


def detect_faces(face_mesh, frame: np.ndarray, tuning: ExpressionTuning, with_thumbnails: bool = False) -> List[FaceDetection]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    detections: List[FaceDetection] = []

    if not results.multi_face_landmarks:
        return detections

    for face_landmarks in results.multi_face_landmarks[: tuning.max_num_faces]:
        points = landmarks_to_points(face_landmarks, frame.shape[1], frame.shape[0])
        box = bbox_from_points(points, frame.shape)
        center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)
        size = float(max(box[2] - box[0], box[3] - box[1]))
        descriptor = compute_face_descriptor(frame, points, box)
        thumbnail = create_face_thumbnail(frame, box) if with_thumbnails else None
        detections.append(
            FaceDetection(
                points=points,
                bbox=box,
                normalized_box=normalize_box(box, frame.shape),
                center=center,
                size=size,
                descriptor=descriptor,
                thumbnail_data_url=thumbnail,
            )
        )

    detections.sort(key=lambda detection: (detection.center[0], detection.center[1]))
    return detections


def sample_frame_indices(capture: cv2.VideoCapture, sample_count: int) -> List[int]:
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        return [0]

    if total_frames <= sample_count:
        return list(range(total_frames))

    indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=np.int32)
    return sorted({int(index) for index in indices})


def face_match_cost(detection: FaceDetection, profile: FaceProfile, frame_shape: Tuple[int, int, int], frame_index: int) -> float:
    descriptor_cost = float(np.linalg.norm(detection.descriptor - profile.descriptor))

    diagonal = max(float(np.linalg.norm(np.array([frame_shape[1], frame_shape[0]], dtype=np.float32))), 1.0)
    if profile.center is None:
        center_cost = 0.0
    else:
        recency_penalty = 1.0 if profile.last_seen_frame < 0 else min(frame_index - profile.last_seen_frame, 20) / 20.0
        center_cost = float(np.linalg.norm(detection.center - profile.center) / diagonal) * (0.8 + recency_penalty)

    if profile.size <= 0 or detection.size <= 0:
        size_cost = 0.0
    else:
        size_cost = abs(math.log((detection.size + EPSILON) / (profile.size + EPSILON)))

    return descriptor_cost * 0.78 + center_cost * 2.2 + size_cost * 0.25


def assign_detections_to_profiles(
    detections: List[FaceDetection],
    profiles: List[FaceProfile],
    frame_shape: Tuple[int, int, int],
    frame_index: int,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not detections or not profiles:
        return [], list(range(len(detections))), list(range(len(profiles)))

    candidates: List[Tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for profile_index, profile in enumerate(profiles):
            cost = face_match_cost(detection, profile, frame_shape, frame_index)
            if cost <= threshold:
                candidates.append((cost, detection_index, profile_index))

    candidates.sort(key=lambda item: item[0])

    used_detections = set()
    used_profiles = set()
    matches: List[Tuple[int, int]] = []

    for _, detection_index, profile_index in candidates:
        if detection_index in used_detections or profile_index in used_profiles:
            continue
        used_detections.add(detection_index)
        used_profiles.add(profile_index)
        matches.append((detection_index, profile_index))

    unmatched_detections = [index for index in range(len(detections)) if index not in used_detections]
    unmatched_profiles = [index for index in range(len(profiles)) if index not in used_profiles]
    return matches, unmatched_detections, unmatched_profiles


def update_profile(profile: FaceProfile, detection: FaceDetection, frame_index: int) -> None:
    if profile.hits <= 0:
        profile.descriptor = detection.descriptor.copy()
    else:
        profile.descriptor = profile.descriptor * 0.72 + detection.descriptor * 0.28
        descriptor_norm = float(np.linalg.norm(profile.descriptor))
        if descriptor_norm > EPSILON:
            profile.descriptor /= descriptor_norm

    profile.center = detection.center.copy()
    profile.size = detection.size
    profile.last_seen_frame = frame_index
    profile.hits += 1

    area = float((detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]))
    if area >= profile.best_area:
        profile.best_area = area
        if detection.thumbnail_data_url:
            profile.thumbnail_data_url = detection.thumbnail_data_url
        profile.representative_box = detection.normalized_box.copy()


def remove_white_background(rgba: np.ndarray, lo_diff: int = 18) -> np.ndarray:
    """Flood-fill from every edge pixel that is near-white.
    Only the edge-connected background becomes transparent; internal white areas
    (like eye whites or clothing) are preserved."""
    out = rgba.copy()
    h, w = out.shape[:2]
    bgr = out[:, :, :3].copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    seeds = (
        [(x, 0) for x in range(0, w, 4)] +
        [(x, h - 1) for x in range(0, w, 4)] +
        [(0, y) for y in range(0, h, 4)] +
        [(w - 1, y) for y in range(0, h, 4)]
    )
    for sx, sy in seeds:
        b, g, r = int(bgr[sy, sx, 0]), int(bgr[sy, sx, 1]), int(bgr[sy, sx, 2])
        if b > 200 and g > 200 and r > 200:
            cv2.floodFill(bgr, mask, (sx, sy), (255, 255, 255),
                          loDiff=(lo_diff, lo_diff, lo_diff),
                          upDiff=(lo_diff, lo_diff, lo_diff),
                          flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))

    out[mask[1:-1, 1:-1].astype(bool), 3] = 0
    return out


def detect_face_in_filter(bgra: np.ndarray) -> Optional[np.ndarray]:
    """Run MediaPipe Face Mesh on the filter PNG and return (468, 2) pixel coords if a
    face is detected, else None. Tries multiple preprocessing variants to handle
    stylized/cartoon artwork."""
    h, w = bgra.shape[:2]
    rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)

    candidates = [
        rgb,
        cv2.convertScaleAbs(rgb, alpha=1.4, beta=15),   # boost contrast
        cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),  # greyscale → RGB
                     cv2.COLOR_GRAY2RGB),
    ]

    for img in candidates:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.15,
        ) as face_mesh:
            results = face_mesh.process(img)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            return np.array([[l.x * w, l.y * h] for l in lm.landmark], dtype=np.float32)
    return None


def compute_delaunay_triangles(
    points: np.ndarray,
    img_shape: Tuple[int, int],
) -> List[Tuple[int, int, int]]:
    """Delaunay triangulation of (N, 2) face landmark array within image bounds."""
    h, w = img_shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for pt in points:
        subdiv.insert((float(clamp(pt[0], 0, w - 1)), float(clamp(pt[1], 0, h - 1))))

    triangles: List[Tuple[int, int, int]] = []
    seen: set = set()
    for tri in subdiv.getTriangleList():
        verts = tri.reshape(3, 2)
        indices = []
        for v in verts:
            dists = np.linalg.norm(points - v, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 2.0:
                indices = []
                break
            indices.append(idx)
        if len(indices) == 3 and len(set(indices)) == 3:
            key = tuple(sorted(indices))
            if key not in seen:
                seen.add(key)
                triangles.append(key)
    return triangles


def build_style_from_overlay(overlay_rgba: np.ndarray) -> CartoonStyle:
    texture = overlay_rgba.copy()
    # If every pixel is fully opaque the PNG has no real transparency → strip background
    if int(texture[:, :, 3].min()) == 255:
        texture = remove_white_background(texture)

    # Try to detect a face in the filter PNG for expression-synced mesh warp
    filter_lm = detect_face_in_filter(texture)
    filter_tri = None
    if filter_lm is not None:
        filter_tri = compute_delaunay_triangles(filter_lm, texture.shape)

    return CartoonStyle(texture=texture, filter_landmarks=filter_lm, filter_triangles=filter_tri)


def extract_expression_coefficients(points: np.ndarray, tuning: ExpressionTuning) -> Dict[str, float]:
    face_width = max(float(np.linalg.norm(points[454] - points[234])), 1.0)
    face_height = max(float(np.linalg.norm(points[152] - points[10])), 1.0)
    left_eye_ratio = np.linalg.norm(points[159] - points[145]) / max(np.linalg.norm(points[33] - points[133]), EPSILON)
    right_eye_ratio = np.linalg.norm(points[386] - points[374]) / max(np.linalg.norm(points[263] - points[362]), EPSILON)
    mouth_open_ratio = np.linalg.norm(points[13] - points[14]) / face_height
    mouth_width_ratio = np.linalg.norm(points[61] - points[291]) / face_width
    smile_lift = ((points[13][1] + points[14][1]) * 0.5 - (points[61][1] + points[291][1]) * 0.5) / face_height
    left_brow_distance = ((points[159][1] + points[145][1]) * 0.5 - np.mean(points[LEFT_BROW], axis=0)[1]) / face_height
    right_brow_distance = ((points[386][1] + points[374][1]) * 0.5 - np.mean(points[RIGHT_BROW], axis=0)[1]) / face_height
    face_center_x = (points[234][0] + points[454][0]) * 0.5

    blink_left = 1.0 - remap(float(left_eye_ratio), tuning.blink_ratio_min, tuning.blink_ratio_max)
    blink_right = 1.0 - remap(float(right_eye_ratio), tuning.blink_ratio_min, tuning.blink_ratio_max)
    smile = clamp(
        remap(float(mouth_width_ratio), tuning.smile_width_min, tuning.smile_width_max) * 0.65
        + remap(float(smile_lift), tuning.smile_lift_min, tuning.smile_lift_max) * 0.35,
        0.0,
        1.0,
    )
    mouth_open = remap(float(mouth_open_ratio), tuning.mouth_open_min, tuning.mouth_open_max)
    brow_raise_left = remap(float(left_brow_distance), tuning.brow_raise_min, tuning.brow_raise_max)
    brow_raise_right = remap(float(right_brow_distance), tuning.brow_raise_min, tuning.brow_raise_max)
    yaw = clamp(float((points[4][0] - face_center_x) / max(face_width * 0.18, EPSILON)), -1.0, 1.0)
    roll = clamp(float(math.degrees(math.atan2(points[263][1] - points[33][1], points[263][0] - points[33][0])) / 30.0), -1.0, 1.0)

    return {
        "blink_left": clamp(blink_left, 0.0, 1.0),
        "blink_right": clamp(blink_right, 0.0, 1.0),
        "smile": smile,
        "mouth_open": mouth_open,
        "brow_raise_left": brow_raise_left,
        "brow_raise_right": brow_raise_right,
        "yaw": (yaw + 1.0) * 0.5,
        "roll": (roll + 1.0) * 0.5,
    }


def _composite_over(canvas: np.ndarray, patch: np.ndarray,
                    cy1: int, cy2: int, cx1: int, cx2: int,
                    py1: int, py2: int, px1: int, px2: int) -> None:
    """Alpha-composite patch region over canvas region in place."""
    dst = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
    src = patch[py1:py2, px1:px2].astype(np.float32)
    sa = src[:, :, 3:4] / 255.0
    da = dst[:, :, 3:4] / 255.0
    oa = sa + da * (1.0 - sa)
    safe = np.maximum(oa, 1e-6)
    out_rgb = (src[:, :, :3] * sa + dst[:, :, :3] * da * (1.0 - sa)) / safe
    canvas[cy1:cy2, cx1:cx2, :3] = out_rgb.clip(0, 255).astype(np.uint8)
    canvas[cy1:cy2, cx1:cx2, 3] = (oa * 255).clip(0, 255).astype(np.uint8).reshape(cy2 - cy1, cx2 - cx1)


def _composite_filter(
    frame: np.ndarray,
    warped_canvas: np.ndarray,
    face_points: np.ndarray,
) -> np.ndarray:
    """Composite warped filter onto frame with a soft feathered boundary and
    luminance matching so the filter looks lit by the same light as the face.

    Inside the face oval the blend fades softly to zero at the skin boundary
    (no hard sticker edge). Outside the oval the filter's own alpha is used
    as-is so cartoon hair / ears / accessories stay crisp.
    """
    h, w = frame.shape[:2]

    oval_pts = np.int32(face_points[FACE_OVAL])
    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(hard_mask, cv2.convexHull(oval_pts), 255)

    # Feather radius: 3 % of face width (tight edge — wide fade caused chin bleed-through)
    face_w = float(np.linalg.norm(face_points[454] - face_points[234]))
    feather_px = max(4, int(face_w * 0.03))

    # Soft edge: small erosion + narrow Gaussian so the filter is opaque across
    # most of the face and only fades in the last few pixels at the boundary.
    erode_k = max(1, feather_px // 2)
    inner = cv2.erode(hard_mask, np.ones((erode_k * 2 + 1, erode_k * 2 + 1), np.uint8))
    soft = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), sigmaX=float(feather_px))
    if soft.max() > 0:
        soft /= soft.max()
    soft = np.clip(soft, 0.0, 1.0)

    filter_alpha = warped_canvas[:, :, 3].astype(np.float32) / 255.0
    src_bgr = warped_canvas[:, :, :3].astype(np.float32)
    dst = frame.astype(np.float32)

    # Luminance matching: scale filter brightness to the face's ambient light.
    # This makes a cartoon filter look like it belongs in the same scene.
    face_px = hard_mask > 127
    w_sum = float(filter_alpha[face_px].sum())
    if face_px.any() and w_sum > 200:
        dst_l = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
        src_l = cv2.cvtColor(warped_canvas[:, :, :3], cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32)
        ref_lum = float(dst_l[face_px].mean())
        src_lum = float((src_l[face_px] * filter_alpha[face_px]).sum() / w_sum)
        if src_lum > 5.0:
            lum_scale = float(np.clip(ref_lum / src_lum, 0.5, 1.8))
            src_bgr = np.clip(src_bgr * lum_scale, 0.0, 255.0)

    # Build final per-pixel blend alpha:
    #   inside oval  → filter alpha × soft feather (fades at skin boundary)
    #   outside oval → filter alpha unchanged (accessories stay sharp)
    inside = (hard_mask[:, :, np.newaxis] > 0).astype(np.float32)
    f3 = filter_alpha[:, :, np.newaxis]
    s3 = soft[:, :, np.newaxis]
    blend_alpha = f3 * (s3 * inside + (1.0 - inside))

    return ((1.0 - blend_alpha) * dst + blend_alpha * src_bgr).clip(0, 255).astype(np.uint8)


def _warp_mesh(frame: np.ndarray,
               texture: np.ndarray,
               src_lm: np.ndarray,
               dst_lm: np.ndarray,
               triangles: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Warp the filter face (src_lm) onto the live face (dst_lm) triangle-by-triangle.
    Because dst_lm changes with every expression, the filter's eye and mouth
    triangles compress/expand with the real face — giving automatic blink/talk sync.
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    for i, j, k in triangles:
        src_tri = np.float32([src_lm[i], src_lm[j], src_lm[k]])
        dst_tri = np.float32([dst_lm[i], dst_lm[j], dst_lm[k]])

        sr = cv2.boundingRect(src_tri)
        dr = cv2.boundingRect(dst_tri)
        if sr[2] <= 0 or sr[3] <= 0 or dr[2] <= 0 or dr[3] <= 0:
            continue

        sx, sy, sw, sh = sr
        dx, dy, dw, dh = dr
        if dx + dw <= 0 or dy + dh <= 0 or dx >= w or dy >= h:
            continue

        crop = texture[sy:sy + sh, sx:sx + sw]
        if crop.size == 0:
            continue

        M = cv2.getAffineTransform(
            src_tri - np.float32([sx, sy]),
            dst_tri - np.float32([dx, dy]),
        )
        warped_patch = cv2.warpAffine(crop, M, (dw, dh),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0, 0))

        tri_mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask,
                           np.int32(np.round(dst_tri - np.float32([dx, dy]))),
                           255, lineType=cv2.LINE_AA)
        warped_patch[:, :, 3] = cv2.bitwise_and(warped_patch[:, :, 3], tri_mask)

        cx1, cy1 = max(0, dx), max(0, dy)
        cx2, cy2 = min(w, dx + dw), min(h, dy + dh)
        px1, py1 = cx1 - dx, cy1 - dy
        px2, py2 = px1 + (cx2 - cx1), py1 + (cy2 - cy1)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        _composite_over(canvas, warped_patch, cy1, cy2, cx1, cx2, py1, py2, px1, px2)

    return canvas  # RGBA — caller composites with frame after applying expression holes


def _warp_perspective_to_canvas(
    frame_shape: Tuple[int, int, int],
    texture: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Perspective-warp the filter texture to align with the live face.

    Returns an RGBA canvas the same size as the frame.  The caller decides how
    to composite it (allowing expression-transparency holes to be punched first).

    Padding ratios are tuned so cartoon heads with large hair (Nobita, Shizuka)
    fit without being clipped at the top or sides.
    """
    h, w = frame_shape[:2]
    th, tw = texture.shape[:2]

    face_w = float(np.linalg.norm(points[454] - points[234]))
    face_h = float(np.linalg.norm(points[152] - points[10]))
    if face_w < 4 or face_h < 4:
        return np.zeros((h, w, 4), dtype=np.uint8)

    right = (points[454] - points[234]) / face_w
    down  = (points[152] - points[10])  / face_h

    # 42 % above forehead (room for hair/hat), 12 % below chin, 72 % each side
    top_mid    = points[10]  - down  * face_h * 0.42
    bottom_mid = points[152] + down  * face_h * 0.12

    tl = top_mid    - right * face_w * 0.72
    tr = top_mid    + right * face_w * 0.72
    br = bottom_mid + right * face_w * 0.65
    bl = bottom_mid - right * face_w * 0.65

    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]),
        np.float32([tl, tr, br, bl]),
    )
    return cv2.warpPerspective(
        texture, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _apply_expression_transparency(
    canvas: np.ndarray,
    dst: np.ndarray,
    coefficients: Dict[str, float],
) -> np.ndarray:
    """Banuba-style mask: eyes and mouth are always transparent so the real face
    shows through and naturally syncs with every blink and lip movement.

    The mask covers cheeks / forehead / nose; the eye + brow and mouth regions
    are permanently punched out with a soft feathered edge.
    """
    h, w = canvas.shape[:2]
    result = canvas.copy()
    alpha = result[:, :, 3].astype(np.float32)

    def fade_region(indices: List[int], strength: float, sigma: float = 7.0) -> None:
        pts = np.int32(dst[indices])
        region = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(region, cv2.convexHull(pts), 255)
        soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), sigmaX=sigma) / 255.0
        np.multiply(alpha, 1.0 - soft * strength, out=alpha)

    # Always reveal real eyes + brows so blinks are live
    fade_region(LEFT_EYE_FULL, 1.0, sigma=9.0)
    fade_region(RIGHT_EYE_FULL, 1.0, sigma=9.0)

    # Always reveal real mouth so lip movement is live
    fade_region(MOUTH_OUTER + MOUTH_INNER, 1.0, sigma=10.0)

    result[:, :, 3] = alpha.clip(0, 255).astype(np.uint8)
    return result


def render_cartoon_face(
    frame: np.ndarray,
    points: np.ndarray,
    style: CartoonStyle,
    coefficients: Dict[str, float],
) -> np.ndarray:
    """Render the filter as a Banuba-style face mask.

    1. Perspective placement — aligns the filter to the live face (position,
       scale, rotation, tilt) via a 4-point warp.

    2. Always-on eye + mouth cutouts — eye/brow and mouth regions are permanently
       transparent so the real eyes (including blinks) and real lips (including
       talking/smiling) always show through the mask in sync with the video.
    """
    canvas = _warp_perspective_to_canvas(frame.shape, style.texture, points)
    canvas = _apply_expression_transparency(canvas, points, coefficients)
    return _composite_filter(frame, canvas, points)


def analyze_video(video_url: str) -> Dict[str, object]:
    tuning = load_expression_tuning()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "analysis-input.mp4")
        download_file(video_url, input_path)

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("Input video could not be opened")

        sample_indices = sample_frame_indices(capture, ANALYSIS_SAMPLE_FRAMES)
        profiles: List[FaceProfile] = []
        sample_snapshots: List[Dict[str, object]] = []

        # static_image_mode=True: correct for random-seek sampling (no stale tracker state)
        # refine_landmarks=False: iris refinement model not needed for face ID
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=tuning.max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for frame_index in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue

                # Downscale before MediaPipe — the model runs much faster on smaller input
                h0, w0 = frame.shape[:2]
                if w0 > MAX_ANALYSIS_DETECT_WIDTH:
                    scale = MAX_ANALYSIS_DETECT_WIDTH / w0
                    small = cv2.resize(frame, (MAX_ANALYSIS_DETECT_WIDTH, max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA)
                else:
                    small = frame

                detections_small = detect_faces(face_mesh, small, tuning, with_thumbnails=False)

                # Scale landmark coords back to full-resolution frame space
                if w0 > MAX_ANALYSIS_DETECT_WIDTH:
                    inv = w0 / MAX_ANALYSIS_DETECT_WIDTH
                    for det in detections_small:
                        det.points[:] *= inv
                        det.bbox = (int(det.bbox[0]*inv), int(det.bbox[1]*inv),
                                    int(det.bbox[2]*inv), int(det.bbox[3]*inv))
                        det.normalized_box = normalize_box(det.bbox, frame.shape)
                        det.center[:] *= inv
                        det.size *= inv

                # Re-extract thumbnails from full-res frame now that bbox is in full coords
                for det in detections_small:
                    det.thumbnail_data_url = create_face_thumbnail(frame, det.bbox)

                detections = detections_small
                matches, unmatched_detections, _ = assign_detections_to_profiles(detections, profiles, frame.shape, frame_index, threshold=1.08)
                assigned_ids: Dict[int, str] = {}

                for detection_index, profile_index in matches:
                    profile = profiles[profile_index]
                    update_profile(profile, detections[detection_index], frame_index)
                    assigned_ids[detection_index] = profile.face_id

                for detection_index in unmatched_detections:
                    next_id = len(profiles) + 1
                    detection = detections[detection_index]
                    profile = FaceProfile(
                        face_id=f"face-{next_id}",
                        label=f"Face {next_id}",
                        descriptor=detection.descriptor.copy(),
                    )
                    update_profile(profile, detection, frame_index)
                    profiles.append(profile)
                    assigned_ids[detection_index] = profile.face_id

                sample_snapshots.append(
                    {
                        "frame": resize_for_preview(frame),
                        "faces": [
                            {
                                "faceId": assigned_ids.get(index),
                                "representativeBox": detection.normalized_box,
                            }
                            for index, detection in enumerate(detections)
                            if assigned_ids.get(index)
                        ],
                    }
                )

        capture.release()

        if not profiles:
            raise ValueError("No faces were detected in the uploaded video")

        representative_snapshot = max(sample_snapshots, key=lambda snapshot: len(snapshot["faces"]))
        preview_boxes = {face["faceId"]: face["representativeBox"] for face in representative_snapshot["faces"]}

        faces = []
        for profile in sorted(profiles, key=lambda item: item.face_id):
            faces.append(
                {
                    "faceId": profile.face_id,
                    "label": profile.label,
                    "thumbnailDataUrl": profile.thumbnail_data_url,
                    "representativeBox": preview_boxes.get(profile.face_id, profile.representative_box),
                    "embedding": [round(float(value), 6) for value in profile.descriptor.tolist()],
                }
            )

        return {
            "representativeFrameDataUrl": encode_image_data_url(representative_snapshot["frame"], ".jpg"),
            "faces": faces,
        }


def mux_original_audio(input_video_path: str, processed_video_path: str, output_video_path: str) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i", processed_video_path,
        "-i", input_video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_video_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_styles_for_assignments(assignments: List[Dict[str, str]], tmpdir: str) -> Dict[str, CartoonStyle]:
    styles_by_face: Dict[str, CartoonStyle] = {}
    overlay_cache: Dict[str, CartoonStyle] = {}

    for index, assignment in enumerate(assignments):
        overlay_url = assignment["overlayImageUrl"]
        if overlay_url not in overlay_cache:
            overlay_path = os.path.join(tmpdir, f"overlay-{index}.png")
            download_file(overlay_url, overlay_path)
            overlay_rgba = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
            if overlay_rgba is None or overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
                raise ValueError("Overlay image must have an alpha channel")
            overlay_cache[overlay_url] = build_style_from_overlay(overlay_rgba)
        styles_by_face[assignment["faceId"]] = overlay_cache[overlay_url]

    return styles_by_face


def profiles_from_detected_faces(detected_faces: List[Dict[str, object]]) -> List[FaceProfile]:
    profiles: List[FaceProfile] = []
    for index, face in enumerate(detected_faces):
        embedding = np.array(face.get("embedding") or [], dtype=np.float32)
        if embedding.size == 0:
            continue
        norm = float(np.linalg.norm(embedding))
        if norm > EPSILON:
            embedding /= norm
        profiles.append(
            FaceProfile(
                face_id=str(face.get("faceId") or f"face-{index + 1}"),
                label=str(face.get("label") or f"Face {index + 1}"),
                descriptor=embedding,
            )
        )
    return profiles


def process_video(video_url: str, detected_faces: List[Dict[str, object]], filter_assignments: List[Dict[str, str]]) -> str:
    if not filter_assignments:
        raise ValueError("At least one filter assignment is required")

    tuning = load_expression_tuning()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        silent_output_path = os.path.join(tmpdir, "processed-silent.mp4")
        final_output_path = os.path.join(tmpdir, "processed-final.mp4")
        download_file(video_url, input_path)

        styles_by_face = load_styles_for_assignments(filter_assignments, tmpdir)
        profiles = profiles_from_detected_faces(detected_faces)

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("Input video could not be opened")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(silent_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        expression_states: Dict[str, ExpressionState] = {face_id: ExpressionState() for face_id in styles_by_face}
        landmark_smoothers: Dict[str, LandmarkSmoother] = {}
        fallback_smoothers: List[LandmarkSmoother] = []
        fallback_expr_states: List[ExpressionState] = []
        fallback_style = styles_by_face[filter_assignments[0]["faceId"]] if len(filter_assignments) == 1 else None
        frame_counter = 0

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=tuning.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                detections = detect_faces(face_mesh, frame, tuning, with_thumbnails=False)

                if profiles:
                    matches, _, _ = assign_detections_to_profiles(detections, profiles, frame.shape, frame_counter, threshold=1.05)
                    for detection_index, profile_index in matches:
                        detection = detections[detection_index]
                        profile = profiles[profile_index]
                        style = styles_by_face.get(profile.face_id)
                        if not style:
                            continue

                        smoother = landmark_smoothers.setdefault(profile.face_id, LandmarkSmoother())
                        if profile.last_seen_frame >= 0 and frame_counter - profile.last_seen_frame > 10:
                            smoother.reset()
                        update_profile(profile, detection, frame_counter)
                        smoothed_points = smoother.update(detection.points, tuning.smoothing_alpha)

                        raw_coeff = extract_expression_coefficients(smoothed_points, tuning)
                        coefficients = expression_states[profile.face_id].smooth(raw_coeff, tuning.smoothing_alpha)
                        frame = render_cartoon_face(frame, smoothed_points, style, coefficients)
                elif fallback_style:
                    for idx, detection in enumerate(detections):
                        while len(fallback_smoothers) <= idx:
                            fallback_smoothers.append(LandmarkSmoother())
                            fallback_expr_states.append(ExpressionState())
                        smoothed_points = fallback_smoothers[idx].update(detection.points, tuning.smoothing_alpha)
                        raw_coeff = extract_expression_coefficients(smoothed_points, tuning)
                        coefficients = fallback_expr_states[idx].smooth(raw_coeff, tuning.smoothing_alpha)
                        frame = render_cartoon_face(frame, smoothed_points, fallback_style, coefficients)

                writer.write(frame)
                frame_counter += 1

        capture.release()
        writer.release()
        mux_original_audio(input_path, silent_output_path, final_output_path)

        preserved_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        preserved_output.close()
        shutil.copyfile(final_output_path, preserved_output.name)
        return preserved_output.name


def cleanup_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        os.remove(path)
