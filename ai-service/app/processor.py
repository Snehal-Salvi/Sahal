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

CANONICAL_CANVAS_SIZE = 512
EPSILON = 1e-6
ANALYSIS_SAMPLE_FRAMES = 8
MAX_ANALYSIS_FRAME_WIDTH = 960

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
DESCRIPTOR_KEYPOINTS = [10, 152, 234, 454, 33, 133, 159, 145, 263, 362, 386, 374, 4, 61, 291, 13, 14]

CANONICAL_CONTROL_POINTS: Dict[int, Tuple[float, float]] = {
    10: (0.50, 0.10),
    338: (0.63, 0.13),
    297: (0.72, 0.18),
    332: (0.79, 0.25),
    284: (0.85, 0.33),
    251: (0.89, 0.43),
    389: (0.90, 0.52),
    356: (0.89, 0.60),
    454: (0.87, 0.68),
    323: (0.83, 0.75),
    361: (0.77, 0.83),
    288: (0.69, 0.89),
    397: (0.60, 0.95),
    152: (0.50, 0.98),
    172: (0.40, 0.95),
    58: (0.31, 0.89),
    132: (0.23, 0.82),
    93: (0.17, 0.74),
    234: (0.13, 0.67),
    127: (0.11, 0.58),
    162: (0.10, 0.47),
    54: (0.15, 0.33),
    67: (0.24, 0.18),
    109: (0.37, 0.13),
    33: (0.31, 0.43),
    160: (0.35, 0.39),
    158: (0.39, 0.39),
    133: (0.43, 0.43),
    153: (0.39, 0.47),
    144: (0.35, 0.47),
    263: (0.69, 0.43),
    387: (0.65, 0.39),
    385: (0.61, 0.39),
    362: (0.57, 0.43),
    380: (0.65, 0.47),
    373: (0.61, 0.47),
    70: (0.28, 0.33),
    63: (0.34, 0.30),
    105: (0.40, 0.29),
    66: (0.45, 0.31),
    107: (0.24, 0.35),
    336: (0.72, 0.33),
    296: (0.66, 0.30),
    334: (0.60, 0.29),
    293: (0.55, 0.31),
    300: (0.76, 0.35),
    168: (0.50, 0.34),
    6: (0.50, 0.46),
    4: (0.50, 0.57),
    195: (0.50, 0.64),
    61: (0.34, 0.71),
    0: (0.50, 0.66),
    291: (0.66, 0.71),
    17: (0.50, 0.80),
    78: (0.36, 0.71),
    13: (0.50, 0.70),
    308: (0.64, 0.71),
    14: (0.50, 0.76),
}
CONTROL_POINT_IDS = list(CANONICAL_CONTROL_POINTS.keys())
CANONICAL_POINTS = np.array(
    [[x_value * CANONICAL_CANVAS_SIZE, y_value * CANONICAL_CANVAS_SIZE] for x_value, y_value in CANONICAL_CONTROL_POINTS.values()],
    dtype=np.float32,
)


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
    base_color: Tuple[int, int, int]
    outline_color: Tuple[int, int, int]
    shadow_color: Tuple[int, int, int]
    blush_color: Tuple[int, int, int]
    lip_color: Tuple[int, int, int]
    mouth_inner_color: Tuple[int, int, int]
    brow_color: Tuple[int, int, int]
    eye_white_color: Tuple[int, int, int]
    pupil_color: Tuple[int, int, int]
    texture: np.ndarray


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


def to_int_tuple(color: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(clamp(float(channel), 0.0, 255.0)) for channel in color)


def adjust_color(color: Sequence[float], factor: float) -> Tuple[int, int, int]:
    return to_int_tuple([channel * factor for channel in color])


def blend_colors(color_a: Sequence[float], color_b: Sequence[float], alpha: float) -> Tuple[int, int, int]:
    return to_int_tuple(
        [(1.0 - alpha) * float(channel_a) + alpha * float(channel_b) for channel_a, channel_b in zip(color_a, color_b)]
    )


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


def prepare_overlay_texture(overlay_rgba: np.ndarray, canvas_size: int = CANONICAL_CANVAS_SIZE) -> np.ndarray:
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    overlay_height, overlay_width = overlay_rgba.shape[:2]
    scale = min((canvas_size * 0.84) / max(overlay_width, 1), (canvas_size * 0.88) / max(overlay_height, 1))
    resized = cv2.resize(
        overlay_rgba,
        (
            max(1, int(round(overlay_width * scale))),
            max(1, int(round(overlay_height * scale))),
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    x_offset = (canvas_size - resized.shape[1]) // 2
    y_offset = int(canvas_size * 0.06) + (canvas_size - int(canvas_size * 0.12) - resized.shape[0]) // 2
    y_offset = int(clamp(y_offset, 0, canvas_size - resized.shape[0]))
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return canvas


def build_style_from_overlay(overlay_rgba: np.ndarray) -> CartoonStyle:
    visible_pixels = overlay_rgba[overlay_rgba[:, :, 3] > 0]
    if visible_pixels.size == 0:
        base_color = np.array([176, 208, 255], dtype=np.float32)
    else:
        base_color = visible_pixels[:, :3].astype(np.float32).mean(axis=0)

    texture = prepare_overlay_texture(overlay_rgba)
    light_tint = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    warm_tint = np.array([214.0, 142.0, 255.0], dtype=np.float32)
    deep_tint = np.array([32.0, 38.0, 48.0], dtype=np.float32)

    return CartoonStyle(
        base_color=blend_colors(base_color, light_tint, 0.18),
        outline_color=blend_colors(base_color, deep_tint, 0.68),
        shadow_color=adjust_color(base_color, 0.75),
        blush_color=blend_colors(base_color, warm_tint, 0.45),
        lip_color=blend_colors(base_color, np.array([88.0, 72.0, 240.0], dtype=np.float32), 0.55),
        mouth_inner_color=to_int_tuple((42.0, 28.0, 86.0)),
        brow_color=blend_colors(base_color, deep_tint, 0.78),
        eye_white_color=to_int_tuple((255, 255, 255)),
        pupil_color=to_int_tuple((24, 24, 24)),
        texture=texture,
    )


def compute_triangle_indices(points: np.ndarray, size: int) -> List[Tuple[int, int, int]]:
    subdiv = cv2.Subdiv2D((0, 0, size, size))
    for point in points:
        subdiv.insert((float(clamp(point[0], 0, size - 1)), float(clamp(point[1], 0, size - 1))))

    triangles: List[Tuple[int, int, int]] = []
    for triangle in subdiv.getTriangleList():
        vertices = triangle.reshape(3, 2)
        indices: List[int] = []
        for vertex in vertices:
            distances = [float(np.linalg.norm(point - vertex)) for point in points]
            nearest_index = int(np.argmin(distances))
            if distances[nearest_index] > 2.0:
                indices = []
                break
            indices.append(nearest_index)
        if len(set(indices)) == 3:
            normalized = tuple(sorted(indices))
            if normalized not in triangles:
                triangles.append(normalized)
    return triangles


CANONICAL_TRIANGLES = compute_triangle_indices(CANONICAL_POINTS, CANONICAL_CANVAS_SIZE)


def composite_rgba(destination: np.ndarray, source: np.ndarray) -> np.ndarray:
    src_alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    dst_alpha = destination[:, :, 3:4].astype(np.float32) / 255.0
    src_rgb = source[:, :, :3].astype(np.float32)
    dst_rgb = destination[:, :, :3].astype(np.float32)

    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    out_rgb = np.where(
        out_alpha > EPSILON,
        (src_rgb * src_alpha + dst_rgb * dst_alpha * (1.0 - src_alpha)) / np.maximum(out_alpha, EPSILON),
        0.0,
    )

    combined = np.zeros_like(destination)
    combined[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    combined[:, :, 3] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8).reshape(destination.shape[:2])
    return combined


def alpha_blend(frame: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    frame_float = frame.astype(np.float32)
    blended = alpha * overlay_rgb + (1.0 - alpha) * frame_float
    return blended.astype(np.uint8)


def warp_overlay_mesh(texture: np.ndarray, target_points: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    frame_height, frame_width = frame_shape[:2]
    warped = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

    for first_index, second_index, third_index in CANONICAL_TRIANGLES:
        src_triangle = np.float32([CANONICAL_POINTS[first_index], CANONICAL_POINTS[second_index], CANONICAL_POINTS[third_index]])
        dst_triangle = np.float32([target_points[first_index], target_points[second_index], target_points[third_index]])

        src_rect = cv2.boundingRect(src_triangle)
        dst_rect = cv2.boundingRect(dst_triangle)
        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            continue

        x_value, y_value, width, height = dst_rect
        if x_value >= frame_width or y_value >= frame_height or x_value + width <= 0 or y_value + height <= 0:
            continue

        src_x, src_y, src_width, src_height = src_rect
        src_crop = texture[src_y:src_y + src_height, src_x:src_x + src_width]
        if src_crop.size == 0:
            continue

        src_offset_triangle = src_triangle - np.array([src_x, src_y], dtype=np.float32)
        dst_offset_triangle = dst_triangle - np.array([x_value, y_value], dtype=np.float32)
        affine_matrix = cv2.getAffineTransform(src_offset_triangle, dst_offset_triangle)
        warped_patch = cv2.warpAffine(
            src_crop,
            affine_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        triangle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, np.int32(np.round(dst_offset_triangle)), 255, lineType=cv2.LINE_AA)
        warped_patch[:, :, 3] = cv2.bitwise_and(warped_patch[:, :, 3], triangle_mask)

        clipped_x1 = max(0, x_value)
        clipped_y1 = max(0, y_value)
        clipped_x2 = min(frame_width, x_value + width)
        clipped_y2 = min(frame_height, y_value + height)
        if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
            continue

        patch_x1 = clipped_x1 - x_value
        patch_y1 = clipped_y1 - y_value
        patch_x2 = patch_x1 + (clipped_x2 - clipped_x1)
        patch_y2 = patch_y1 + (clipped_y2 - clipped_y1)
        destination = warped[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
        source = warped_patch[patch_y1:patch_y2, patch_x1:patch_x2]
        warped[clipped_y1:clipped_y2, clipped_x1:clipped_x2] = composite_rgba(destination, source)

    return warped


def draw_translucent_polygon(layer: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], alpha: int) -> None:
    polygon = np.int32(np.round(points))
    if polygon.shape[0] >= 3:
        cv2.fillPoly(layer, [polygon], (color[0], color[1], color[2], alpha), lineType=cv2.LINE_AA)


def draw_face_base(layer: np.ndarray, points: np.ndarray, style: CartoonStyle) -> None:
    face_oval = points[FACE_OVAL]
    draw_translucent_polygon(layer, face_oval, style.base_color, 72)
    cv2.polylines(layer, [np.int32(np.round(face_oval))], True, (style.outline_color[0], style.outline_color[1], style.outline_color[2], 120), 3, lineType=cv2.LINE_AA)

    blush_radius = max(8, int(np.linalg.norm(points[234] - points[454]) * 0.05))
    for cheek in (points[123], points[352]):
        cv2.circle(layer, tuple(np.int32(np.round(cheek))), blush_radius, (style.blush_color[0], style.blush_color[1], style.blush_color[2], 78), -1, lineType=cv2.LINE_AA)

    highlight_center = tuple(np.int32(np.round((points[10] + points[4]) * 0.5)))
    highlight_radius = max(10, int(np.linalg.norm(points[10] - points[4]) * 0.18))
    cv2.circle(layer, highlight_center, highlight_radius, (255, 255, 255, 28), -1, lineType=cv2.LINE_AA)


def draw_brow(layer: np.ndarray, brow_points: np.ndarray, eye_center: np.ndarray, brow_raise: float, style: CartoonStyle) -> None:
    brow_center = brow_points.mean(axis=0)
    lift = brow_points + (brow_center - eye_center) * brow_raise * 0.30
    thickness = max(3, int(np.linalg.norm(lift[0] - lift[-1]) * 0.10))
    cv2.polylines(layer, [np.int32(np.round(lift))], False, (style.brow_color[0], style.brow_color[1], style.brow_color[2], 235), thickness, lineType=cv2.LINE_AA)


def draw_eye(layer: np.ndarray, eye_points: np.ndarray, blink: float, style: CartoonStyle) -> None:
    outer_corner = eye_points[0]
    upper_points = eye_points[1:3]
    inner_corner = eye_points[3]
    lower_points = eye_points[4:6]
    eye_center = eye_points.mean(axis=0)
    eye_width = max(float(np.linalg.norm(inner_corner - outer_corner)), 1.0)
    eye_height = max(float(np.linalg.norm(upper_points.mean(axis=0) - lower_points.mean(axis=0))), 1.0)
    angle = math.degrees(math.atan2(inner_corner[1] - outer_corner[1], inner_corner[0] - outer_corner[0]))
    openness = clamp(1.0 - blink, 0.0, 1.0)

    if openness < 0.18:
        direction = inner_corner - outer_corner
        direction /= max(float(np.linalg.norm(direction)), EPSILON)
        offset = direction * max(6, int(eye_width * 0.42))
        start = tuple(np.int32(np.round(eye_center - offset)))
        end = tuple(np.int32(np.round(eye_center + offset)))
        cv2.line(layer, start, end, (style.brow_color[0], style.brow_color[1], style.brow_color[2], 245), max(2, int(eye_width * 0.08)), lineType=cv2.LINE_AA)
        return

    axes = (max(4, int(eye_width * 0.32)), max(2, int(eye_height * (0.35 + openness * 1.8))))
    center = tuple(np.int32(np.round(eye_center)))
    cv2.ellipse(layer, center, axes, angle, 0, 360, (style.eye_white_color[0], style.eye_white_color[1], style.eye_white_color[2], 242), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(layer, center, axes, angle, 0, 360, (style.outline_color[0], style.outline_color[1], style.outline_color[2], 235), max(2, int(eye_width * 0.06)), lineType=cv2.LINE_AA)

    pupil_radius = max(3, int(eye_width * 0.13))
    pupil_center = eye_center + np.array([0.0, eye_height * 0.10], dtype=np.float32)
    cv2.circle(layer, tuple(np.int32(np.round(pupil_center))), pupil_radius, (style.pupil_color[0], style.pupil_color[1], style.pupil_color[2], 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(layer, tuple(np.int32(np.round(pupil_center + np.array([-pupil_radius * 0.35, -pupil_radius * 0.35], dtype=np.float32)))), max(1, pupil_radius // 3), (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)


def scale_points(points: np.ndarray, center: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    shifted = points - center
    shifted[:, 0] *= scale_x
    shifted[:, 1] *= scale_y
    return shifted + center


def draw_mouth(layer: np.ndarray, points: np.ndarray, coefficients: Dict[str, float], style: CartoonStyle) -> None:
    outer = points[MOUTH_OUTER].copy()
    inner = points[MOUTH_INNER].copy()
    mouth_center = (points[13] + points[14] + points[61] + points[291]) * 0.25

    outer = scale_points(outer, mouth_center, 1.0 + coefficients["smile"] * 0.26, 0.82 + coefficients["mouth_open"] * 0.95)
    cv2.fillPoly(layer, [np.int32(np.round(outer))], (style.lip_color[0], style.lip_color[1], style.lip_color[2], 220), lineType=cv2.LINE_AA)
    cv2.polylines(layer, [np.int32(np.round(outer))], True, (style.outline_color[0], style.outline_color[1], style.outline_color[2], 245), 2, lineType=cv2.LINE_AA)

    if coefficients["mouth_open"] > 0.08:
        inner = scale_points(inner, mouth_center, 0.95 + coefficients["smile"] * 0.14, 0.80 + coefficients["mouth_open"] * 1.30)
        cv2.fillPoly(layer, [np.int32(np.round(inner))], (style.mouth_inner_color[0], style.mouth_inner_color[1], style.mouth_inner_color[2], 240), lineType=cv2.LINE_AA)


def draw_nose(layer: np.ndarray, points: np.ndarray, style: CartoonStyle) -> None:
    radius = max(3, int(np.linalg.norm(points[4] - points[6]) * 0.28))
    cv2.circle(layer, tuple(np.int32(np.round(points[4]))), radius, (style.shadow_color[0], style.shadow_color[1], style.shadow_color[2], 165), -1, lineType=cv2.LINE_AA)


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


def render_cartoon_face(frame: np.ndarray, points: np.ndarray, style: CartoonStyle, coefficients: Dict[str, float]) -> np.ndarray:
    target_points = np.array([points[index] for index in CONTROL_POINT_IDS], dtype=np.float32)
    textured_mesh = warp_overlay_mesh(style.texture, target_points, frame.shape)
    feature_layer = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    draw_face_base(feature_layer, points, style)
    draw_brow(feature_layer, points[LEFT_BROW], points[LEFT_EYE_RING].mean(axis=0), coefficients["brow_raise_left"], style)
    draw_brow(feature_layer, points[RIGHT_BROW], points[RIGHT_EYE_RING].mean(axis=0), coefficients["brow_raise_right"], style)
    draw_eye(feature_layer, points[LEFT_EYE_RING], coefficients["blink_left"], style)
    draw_eye(feature_layer, points[RIGHT_EYE_RING], coefficients["blink_right"], style)
    draw_nose(feature_layer, points, style)
    draw_mouth(feature_layer, points, coefficients, style)
    return alpha_blend(frame, composite_rgba(textured_mesh, feature_layer))


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

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=tuning.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            for frame_index in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue

                detections = detect_faces(face_mesh, frame, tuning, with_thumbnails=True)
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
        "-i",
        processed_video_path,
        "-i",
        input_video_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
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

                        update_profile(profile, detection, frame_counter)
                        coefficients = extract_expression_coefficients(detection.points, tuning)
                        coefficients = expression_states.setdefault(profile.face_id, ExpressionState()).smooth(
                            coefficients,
                            tuning.smoothing_alpha,
                        )
                        frame = render_cartoon_face(frame, detection.points, style, coefficients)
                elif fallback_style:
                    for detection in detections:
                        coefficients = extract_expression_coefficients(detection.points, tuning)
                        frame = render_cartoon_face(frame, detection.points, fallback_style, coefficients)

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
