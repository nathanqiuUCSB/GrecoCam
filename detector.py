import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os
import time
from collections import deque

from audio import configure, announce
from camera import open_camera, is_jetson
from face_prep import prepare_face_crop, face_too_blown_out, brighten_face_in_frame
from visit_log import VisitLogger

# ── Config ────────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
CACHE_FILE = "face_cache.npz"

def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")

def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default

def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default

# Higher = more lenient matching (0.32 strict, 0.45 loose).
RECOGNITION_THRESHOLD = _env_float("RECOGNITION_THRESHOLD", 0.45 if is_jetson() else 0.32)
FACE_UPSCALE_MIN = _env_int("FACE_UPSCALE_MIN", 200)
# How often to run DeepFace (every N YOLO passes). 1 = fastest announce, more CPU.
RUN_RECOGNITION_EVERY = _env_int("RUN_RECOGNITION_EVERY", 1 if is_jetson() else 4)
YOLO_CONF = _env_float("YOLO_CONF", 0.30)
MIN_FACE_SIZE = _env_int("MIN_FACE_SIZE", 40)  # smaller = detect farther away
# Known faces: votes needed in the recent window before announcing.
VOTES_REQUIRED = _env_int("VOTES_REQUIRED", 2)
VOTE_WINDOW = _env_int("VOTE_WINDOW", 4)
UNKNOWN_VOTES_REQUIRED = _env_int("UNKNOWN_VOTES_REQUIRED", 4)  # strangers need more agreement
# One strong match at or above this similarity announces immediately (known faces only).
HIGH_CONF_ANNOUNCE = _env_float("HIGH_CONF_ANNOUNCE", 0.80)
ABSENT_SEC = _env_float("ABSENT_SEC", 2.0)  # how long face must be gone before a new visit
ENABLE_TTS = True
TTS_BACKEND = "edge"  # edge (realistic) | auto | pyttsx3 | espeak
TTS_VOICE = "en-US-JennyNeural"  # try preview_voices.py to pick one
KNOWN_MESSAGE = "Welcome home, {name}!"
UNKNOWN_MESSAGE = "So we don't knock anymore huh"

SHOW_PREVIEW = _env_bool("SHOW_PREVIEW", default=not is_jetson())
STREAM_PREVIEW = _env_bool("STREAM_PREVIEW", default=False)
STREAM_PORT = _env_int("STREAM_PORT", 8080)
MUSIC_MODE = _env_bool("MUSIC_MODE", default=False)
# ──────────────────────────────────────────────────────────────────────────────

configure(enabled=ENABLE_TTS, backend=TTS_BACKEND, voice=TTS_VOICE,
          music_mode=MUSIC_MODE)
if MUSIC_MODE:
    print("Audio: MUSIC MODE (using sounds/<name>music.mp3 clips)\n")

model = YOLO("yolo26n-face.pt")

# ── Face cache ────────────────────────────────────────────────────────────────
def get_file_mtimes(directory):
    mtimes = {}
    for f in os.listdir(directory):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            mtimes[f] = os.path.getmtime(os.path.join(directory, f))
    return mtimes

def load_known_faces(directory, cache_file=CACHE_FILE):
    current_mtimes = get_file_mtimes(directory)

    if os.path.exists(cache_file):
        cache = np.load(cache_file, allow_pickle=True)
        if cache["mtimes"].item() == current_mtimes:
            print("Loading faces from cache...")
            names = cache["names"].tolist()
            embeddings = cache["embeddings"]
            known = {name: embeddings[i] for i, name in enumerate(names)}
            print(f"Loaded {len(known)} face(s) from cache\n")
            return known

    print("Computing embeddings...")
    raw = {}
    for filename in current_mtimes:
        name = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        path = os.path.join(directory, filename)
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend="retinaface"
            )
            raw.setdefault(name, []).append(np.array(result[0]["embedding"]))
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ Skipped {filename}: {e}")

    known = {}
    for name, embeddings in raw.items():
        avg = np.mean(embeddings, axis=0)
        known[name] = avg / np.linalg.norm(avg)
        print(f"  → {name}: {len(embeddings)} photos averaged")

    names = list(known.keys())
    np.savez(cache_file,
             names=names,
             embeddings=np.stack([known[n] for n in names]),
             mtimes=np.array(current_mtimes, dtype=object))
    print(f"Cache saved to {cache_file}\n")
    return known

print("Loading known faces...")
known_faces = load_known_faces(KNOWN_FACES_DIR)
print(f"Loaded {len(known_faces)} face(s)")
print(f"Recognition threshold: {RECOGNITION_THRESHOLD} "
      f"| known votes={VOTES_REQUIRED}/{VOTE_WINDOW} instant>={HIGH_CONF_ANNOUNCE:.0%} "
      f"| unknown votes={UNKNOWN_VOTES_REQUIRED} "
      f"| every={RUN_RECOGNITION_EVERY} yolo passes\n")

# ── Recognition ───────────────────────────────────────────────────────────────
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_face(face_crop):
    try:
        if face_too_blown_out(face_crop):
            return None, ""  # don't count as Unknown — wait for a better frame

        face_crop = prepare_face_crop(face_crop)
        result = DeepFace.represent(
            img_path=face_crop,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="skip"
        )
        embedding = np.array(result[0]["embedding"])

        best_name, best_dist = "Unknown", float("inf")
        for name, known_emb in known_faces.items():
            dist = cosine_distance(embedding, known_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        similarity = f"{(1 - best_dist) * 100:.0f}%"
        if best_dist < RECOGNITION_THRESHOLD:
            return best_name.capitalize(), similarity
        return "Unknown", similarity

    except Exception:
        return "Unknown", ""

# ── Announcements ─────────────────────────────────────────────────────────────
class VisitAnnouncer:
    """Announce once per visit after enough matching votes (majority-style)."""

    def __init__(self, votes_required, vote_window, absent_sec, unknown_votes, high_conf,
                 visit_logger=None):
        self.votes_required = votes_required
        self.unknown_votes = unknown_votes
        self.high_conf = high_conf
        self.absent_sec = absent_sec
        self.visit_logger = visit_logger
        self.recent_votes = deque(maxlen=vote_window)
        self.last_face_time = None
        self.visit_active = False
        self.announced_this_visit = False
        self.visit_start_time = None

    def _votes_needed(self, name, conf):
        if name == "Unknown":
            return self.unknown_votes
        if conf:
            try:
                pct = float(str(conf).rstrip("%")) / 100.0
                if pct >= self.high_conf:
                    return 1
            except ValueError:
                pass
        return self.votes_required

    def _is_high_conf(self, name, conf):
        if name == "Unknown" or not conf:
            return False
        try:
            return float(str(conf).rstrip("%")) / 100.0 >= self.high_conf
        except ValueError:
            return False

    def _end_visit(self):
        if self.visit_active and not self.announced_this_visit and self.visit_logger:
            elapsed = time.time() - self.visit_start_time if self.visit_start_time else 0
            self.visit_logger.save_no_announce(elapsed)
        self.visit_active = False
        self.announced_this_visit = False
        self.visit_start_time = None
        self.recent_votes.clear()

    def _start_visit(self):
        self.visit_active = True
        self.announced_this_visit = False
        self.visit_start_time = time.time()
        self.recent_votes.clear()
        if self.visit_logger:
            self.visit_logger.on_visit_start()
        print(f"[{time.strftime('%H:%M:%S')}] Visit started")

    def record(self, name, conf="", snapshot=None):
        now = time.time()

        if name is not None:
            self.last_face_time = now
            if not self.visit_active:
                self._start_visit()
            if self.announced_this_visit:
                return

            if snapshot and self.visit_logger:
                self.visit_logger.note_recognition(
                    snapshot["face_crop"], snapshot["frame"],
                    snapshot["box"], name, conf,
                )

            label = f"{name} {conf}".strip()
            print(f"[{time.strftime('%H:%M:%S')}] Recognized: {label}")

            self.recent_votes.append(name)
            if self.visit_logger:
                self.visit_logger.note_vote(name)
            needed = self._votes_needed(name, conf)
            count = list(self.recent_votes).count(name)
            instant = self._is_high_conf(name, conf)
            tag = " instant" if instant and needed == 1 else ""
            print(f"  votes: {name}={count}/{needed}{tag}  window={list(self.recent_votes)}")
            if count < needed:
                return

            if name == "Unknown":
                message = UNKNOWN_MESSAGE
                clip_name = "unknown"
            else:
                message = KNOWN_MESSAGE.format(name=name)
                clip_name = name

            elapsed = now - self.visit_start_time if self.visit_start_time else 0
            print(f"[{time.strftime('%H:%M:%S')}] ANNOUNCE ({elapsed:.2f}s since visit): {message}")
            self.announced_this_visit = True
            self.recent_votes.clear()
            if self.visit_logger:
                self.visit_logger.save_announced(name, conf, elapsed)
            announce(clip_name, fallback_text=message)
            return

        if self.last_face_time and (now - self.last_face_time) >= self.absent_sec:
            self._end_visit()

visit_logger = VisitLogger()
announcer = VisitAnnouncer(
    VOTES_REQUIRED, VOTE_WINDOW, ABSENT_SEC, UNKNOWN_VOTES_REQUIRED,
    HIGH_CONF_ANNOUNCE, visit_logger=visit_logger,
)
if visit_logger.enabled:
    print(f"Visit log: {visit_logger.directory}/ (set SAVE_VISITS=0 to disable)\n")

def largest_face(boxes, frame):
    best = None
    best_area = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        if float(box.conf[0]) < YOLO_CONF:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    if best is None:
        return None, None
    x1, y1, x2, y2 = best
    return best, frame[y1:y2, x1:x2]

# ── Camera ────────────────────────────────────────────────────────────────────
cap = open_camera()

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

if STREAM_PREVIEW:
    import stream
    stream.start_server(STREAM_PORT)

if SHOW_PREVIEW:
    print("Camera opened. Press Q to quit.")
else:
    print("Camera opened (headless). Press Ctrl+C to quit.")

frame_count = 0
last_detections = []
MAX_READ_FAILURES = 10

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # CSI can drop a frame or two after Argus hiccups; retry before quitting.
            failures = 1
            while failures < MAX_READ_FAILURES:
                print(f"Camera read failed (attempt {failures}/{MAX_READ_FAILURES}), retrying...")
                time.sleep(0.2)
                ret, frame = cap.read()
                if ret:
                    break
                failures += 1
            if not ret:
                print("Camera stopped producing frames. Exiting.")
                print("  Tip: reseat ribbon if this keeps happening, or restart nvargus-daemon:")
                print("  sudo systemctl restart nvargus-daemon")
                break

        if frame_count % 2 == 0:
            face_results = model(frame, verbose=False, imgsz=640)

            if frame_count % RUN_RECOGNITION_EVERY == 0:
                last_detections = []
                all_boxes = []
                for result in face_results:
                    all_boxes.extend(result.boxes)

                primary_box, face_crop = largest_face(all_boxes, frame)
                primary_name, primary_conf = "Unknown", ""

                if primary_box is not None and face_crop.size > 0:
                    primary_name, primary_conf = identify_face(face_crop)

                snapshot = None
                if primary_box is not None and face_crop.size > 0:
                    snapshot = {
                        "face_crop": face_crop,
                        "frame": frame,
                        "box": primary_box,
                    }

                if primary_box is not None and primary_name is not None:
                    announcer.record(primary_name, primary_conf, snapshot=snapshot)
                else:
                    # No face, or blown-out glare frame — don't vote Unknown
                    announcer.record(None)

                for box in all_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue
                    if float(box.conf[0]) < YOLO_CONF:
                        continue

                    if primary_box == (x1, y1, x2, y2):
                        if primary_name is None:
                            name, face_conf = "Glare", ""
                        else:
                            name, face_conf = primary_name, primary_conf
                    else:
                        name, face_conf = "Face", ""

                    color = (0, 220, 0) if name not in ("Unknown", "Face", "Glare") else (0, 140, 255)
                    label = f"{name} {face_conf}" if face_conf else name

                    last_detections.append({
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "color": color
                    })

        if SHOW_PREVIEW or STREAM_PREVIEW:
            display = frame
            if last_detections:
                primary = max(
                    last_detections,
                    key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
                )
                display = brighten_face_in_frame(frame, primary["box"])
            for det in last_detections:
                x1, y1, x2, y2 = det["box"]
                color = det["color"]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, det["label"], (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            known_count = sum(1 for d in last_detections if d["color"] == (0, 220, 0))
            cv2.putText(display, f"Known: {known_count}  Total: {len(last_detections)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame = display

        if STREAM_PREVIEW:
            stream.update_frame(frame)

        if SHOW_PREVIEW:
            cv2.imshow("Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"\nUnexpected error: {type(e).__name__}: {e}")
    raise

cap.release()
if SHOW_PREVIEW:
    cv2.destroyAllWindows()