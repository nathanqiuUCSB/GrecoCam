"""Face crop enhancement — brighten backlit/dark faces for preview and recognition."""

import os

import cv2
import numpy as np

FACE_UPSCALE_MIN = int(os.environ.get("FACE_UPSCALE_MIN", "200"))
# Target mean brightness (0–255) after enhancement.
FACE_TARGET_MEAN = float(os.environ.get("FACE_TARGET_MEAN", "110"))


def face_brightness(crop):
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _apply_clahe(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _lift_to_target(crop, target_mean=FACE_TARGET_MEAN):
    """Scale luminance so the face region reaches target_mean (caps gain)."""
    mean = face_brightness(crop)
    if mean < 1.0 or mean >= target_mean:
        return crop
    gain = min(target_mean / mean, 2.5)
    lifted = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return lifted


def enhance_face_crop(crop, upscale_min=0, target_mean=FACE_TARGET_MEAN):
    """CLAHE + brightness lift. Optionally upscale small crops."""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    side = min(h, w)
    if upscale_min and side < upscale_min:
        scale = upscale_min / side
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    crop = _apply_clahe(crop)
    if face_brightness(crop) < target_mean:
        crop = _lift_to_target(crop, target_mean)
    return crop


def prepare_face_crop(face_crop):
    """Full prep for DeepFace recognition (upscale + enhance)."""
    return enhance_face_crop(face_crop, upscale_min=FACE_UPSCALE_MIN)


def face_too_blown_out(face_crop):
    """True when the crop is mostly white (door glare) — skip recognition."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    return mean > 210 and float(gray.std()) < 35


def brighten_face_in_frame(frame, box, target_mean=FACE_TARGET_MEAN):
    """Enhance just the face ROI and paste back — for live preview."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2].copy()
    if face_too_blown_out(roi):
        return frame
    enhanced = enhance_face_crop(roi, upscale_min=0, target_mean=target_mean)
    out = frame.copy()
    out[y1:y2, x1:x2] = enhanced
    return out
