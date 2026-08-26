"""Enroll known faces using the Arducam / Jetson CSI camera.

Interactive: open the browser preview, press Enter in SSH for each shot.
Timed: use --auto for countdown mode.

Usage (on Orin — stop detector.py first):
  cd ~/GrecoCam && source venv/bin/activate
  export JETSON_WIDTH=1280 JETSON_HEIGHT=720 JETSON_FPS=60

  python3 enroll.py nathan --clear
  # Brighter enroll (detector can stay at -1 for door glare):
  ENROLL_EXPOSURE_COMP=0.5 python3 enroll.py nathan --clear
  # Browser: http://192.168.0.26:8080/

  rm -f face_cache.npz
  STREAM_PREVIEW=1 python3 detector.py
"""

import argparse
import os
import sys
import threading
import time

import cv2
from ultralytics import YOLO

import stream
from camera import open_camera, use_jetson_camera, opencv_has_gstreamer
from face_prep import face_brightness, prepare_face_crop, brighten_face_in_frame

KNOWN_FACES_DIR = "known_faces"
WARMUP_FRAMES = 30
DEFAULT_COUNT = 10
PAD = 0.25
# Brighter than detector default (-1) — indoor enrollment is usually dim at -1.
DEFAULT_ENROLL_EXPOSURE = 0.0


def enroll_exposure_comp(cli_value=None):
    if cli_value is not None:
        return cli_value
    if os.environ.get("ENROLL_EXPOSURE_COMP") is not None:
        return float(os.environ["ENROLL_EXPOSURE_COMP"])
    if os.environ.get("JETSON_EXPOSURE_COMP") is not None:
        return float(os.environ["JETSON_EXPOSURE_COMP"])
    return DEFAULT_ENROLL_EXPOSURE


def next_index(name, directory):
    prefix = f"{name.lower()}_"
    existing = []
    for f in os.listdir(directory):
        if not f.lower().startswith(prefix):
            continue
        stem = os.path.splitext(f)[0]
        suffix = stem[len(prefix):]
        if suffix.isdigit():
            existing.append(int(suffix))
    return (max(existing) + 1) if existing else 1


def clear_person(name, directory):
    prefix = f"{name.lower()}_"
    removed = []
    for f in os.listdir(directory):
        if f.lower().startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(directory, f)
            os.remove(path)
            removed.append(f)
    return removed


def largest_face_crop(frame, model):
    results = model(frame, verbose=False, imgsz=640)
    best = None
    best_area = 0
    h, w = frame.shape[:2]
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            if bw < 40 or bh < 40:
                continue
            area = bw * bh
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
    if best is None:
        return None, None
    x1, y1, x2, y2 = best
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * PAD))
    y1 = max(0, int(y1 - bh * PAD))
    x2 = min(w, int(x2 + bw * PAD))
    y2 = min(h, int(y2 + bh * PAD))
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


class LivePreview:
    """Background thread: live MJPEG with face box while you enroll."""

    def __init__(self, cap, model, name, total, stream_port):
        self.cap = cap
        self.model = model
        self.name = name
        self.shot_num = 1
        self.total = total
        self.saved = 0
        self.status = "Press Enter in SSH to capture"
        self.flash_until = 0.0
        self.lock = threading.Lock()
        self._latest_frame = None
        self._latest_crop = None
        self._latest_box = None
        self.running = True
        self._detect_every = 2
        self._frame_i = 0
        if stream_port:
            stream.start_server(stream_port)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_shot(self, num):
        with self.lock:
            self.shot_num = num
            self.status = "Press Enter in SSH to capture"

    def mark_saved(self):
        with self.lock:
            self.saved += 1
            self.status = "SAVED!"
            self.flash_until = time.time() + 1.2

    def stop(self):
        self.running = False
        self._thread.join(timeout=2.0)

    def capture(self):
        """Return (crop, box) from latest detected face, or try a few fresh reads."""
        with self.lock:
            if self._latest_crop is not None:
                return self._latest_crop.copy(), self._latest_box
        for _ in range(8):
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            crop, box = largest_face_crop(frame, model=self.model)
            if crop is not None:
                return crop, box
            time.sleep(0.05)
        return None, None

    def _draw_overlay(self, frame, box, face_ok, crop=None):
        out = frame.copy()
        if box and face_ok:
            out = brighten_face_in_frame(out, box)
        h, w = out.shape[:2]
        if box:
            x1, y1, x2, y2 = box
            color = (0, 220, 0) if face_ok else (0, 140, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        with self.lock:
            status = self.status
            shot = self.shot_num
            total = self.total
            saved = self.saved
            flash = time.time() < self.flash_until
        if flash:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 220, 0), 8)
        lines = [
            f"Enroll: {self.name}",
            f"Shot {shot}/{total}  saved: {saved}",
            status,
            "SSH: Enter = capture, q = quit",
        ]
        y = 28
        for line in lines:
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        if not face_ok:
            cv2.putText(out, "No face — move closer / look at camera", (10, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
        elif crop is not None:
            bright = face_brightness(crop)
            enhanced = prepare_face_crop(crop)
            bright_after = face_brightness(enhanced)
            if bright < 70:
                hint = f"Face dark ({bright:.0f}) — enhancing to ~{bright_after:.0f}"
                color = (0, 200, 255)
            elif bright > 200:
                hint = f"Face bright ({bright:.0f}) — may be glare"
                color = (0, 140, 255)
            else:
                hint = f"Face OK ({bright:.0f} → saved ~{bright_after:.0f})"
                color = (0, 220, 0)
            cv2.putText(out, hint, (10, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return out

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            self._frame_i += 1
            box = None
            crop = None
            if self._frame_i % self._detect_every == 0:
                crop, box = largest_face_crop(frame, self.model)
                with self.lock:
                    self._latest_frame = frame.copy()
                    self._latest_crop = crop.copy() if crop is not None else None
                    self._latest_box = box
            else:
                with self.lock:
                    box = self._latest_box
                    crop = self._latest_crop
            display = self._draw_overlay(frame, box, crop is not None, crop)
            stream.update_frame(display)
            time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser(description="Enroll faces from the Orin camera")
    parser.add_argument("name", help="Person name, e.g. nathan or ethan")
    parser.add_argument("-n", "--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of photos (default {DEFAULT_COUNT})")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing photos for this person first")
    parser.add_argument("--auto", action="store_true",
                        help="Timed countdown instead of pressing Enter")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="Seconds between auto shots (default 1.5)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable browser preview")
    parser.add_argument("--port", type=int, default=8080,
                        help="MJPEG preview port (default 8080)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="Camera exposure compensation (-2..2). "
                             "Default 0 for enroll (detector uses -1 for glare).")
    args = parser.parse_args()

    name = args.name.strip().lower()
    if not name.isalnum():
        print("Name should be letters/numbers only (e.g. nathan)")

        sys.exit(1)

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    if args.clear:
        removed = clear_person(name, KNOWN_FACES_DIR)
        print(f"Removed {len(removed)} old photo(s) for '{name}'")
        for f in removed:
            print(f"  - {f}")
        print()

    if use_jetson_camera():
        print(f"OpenCV GStreamer support: {'yes' if opencv_has_gstreamer() else 'NO'}")

    print("Loading face detector...")
    model = YOLO("yolo26n-face.pt")

    exposure = enroll_exposure_comp(args.exposure)
    cap = open_camera(exposure_comp=exposure)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    print(f"Enrollment exposure compensation: {exposure}")
    print("  (detector uses JETSON_EXPOSURE_COMP, default -1, to tame door glare)")

    for _ in range(WARMUP_FRAMES):
        cap.read()

    stream_port = None if args.no_stream else args.port
    preview = LivePreview(cap, model, name, args.count, stream_port)

    print(f"Enrolling '{name}' — {args.count} face crops from IMX219")
    if stream_port:
        print("Open the preview in your browser (URL printed above).")
    print("Tips: face the camera, vary angle/distance, try door-open lighting.")
    if args.auto:
        print("Mode: auto countdown\n")
    else:
        print("Mode: press Enter in SSH to capture (Ctrl+C to stop early)\n")

    start = next_index(name, KNOWN_FACES_DIR)
    saved = []

    try:
        while len(saved) < args.count:
            i = len(saved) + 1
            preview.set_shot(i)

            if args.auto:
                for t in range(2, 0, -1):
                    print(f"  shot {i}/{args.count} in {t}...", end="\r")
                    time.sleep(1.0)
            else:
                try:
                    reply = input(
                        f"  [{i}/{args.count}] Pose, then press Enter "
                        f"(or q + Enter to quit): "
                    ).strip().lower()
                except EOFError:
                    break
                if reply in ("q", "quit"):
                    break

            crop, _box = preview.capture()
            if crop is None:
                print("  No face found — move closer / look at the camera, try again")
                continue

            bright = face_brightness(crop)
            if bright < 70:
                print(f"  Dark face ({bright:.0f}) — auto-enhancing on save")
            elif bright > 200:
                print(f"  Warning: face looks very bright ({bright:.0f}) — may be glare")

            idx = start + len(saved)
            path = os.path.join(KNOWN_FACES_DIR, f"{name}_{idx}.jpg")
            enhanced = prepare_face_crop(crop)
            cv2.imwrite(path, enhanced)
            saved.append(path)
            preview.mark_saved()
            h, w = enhanced.shape[:2]
            print(f"  Saved {path} ({w}x{h} crop, brightness {face_brightness(enhanced):.0f})")

            if args.auto and len(saved) < args.count:
                time.sleep(max(0.0, args.interval - 2.0))
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        preview.stop()
        cap.release()

    print(f"\nDone — {len(saved)} face crop(s) for {name}.")
    print("Rebuild cache, then run detector:")
    print("  rm -f face_cache.npz")
    print("  STREAM_PREVIEW=1 python3 detector.py")


if __name__ == "__main__":
    main()
