import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os

# ── Config ────────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
RECOGNITION_THRESHOLD = 0.32      # ← tightened: Facenet512 works best at 0.30–0.35
MIN_FACE_SIZE = 48                 # ← NEW: skip crops smaller than this (px)
RUN_RECOGNITION_EVERY = 8          # ← slightly more frequent for fresher labels
# ──────────────────────────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")

# ── Load known face embeddings ────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
CACHE_FILE = "face_cache.npz"

def get_file_mtimes(directory):
    """Return a dict of filename -> mtime for all images in the directory."""
    mtimes = {}
    for f in os.listdir(directory):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(directory, f)
            mtimes[f] = os.path.getmtime(path)
    return mtimes

def load_known_faces(directory, cache_file=CACHE_FILE):
    current_mtimes = get_file_mtimes(directory)

    # Try loading cache
    if os.path.exists(cache_file):
        cache = np.load(cache_file, allow_pickle=True)
        cached_mtimes = cache["mtimes"].item()  # dict stored as 0-d object array

        if cached_mtimes == current_mtimes:
            print("Loading faces from cache...")
            names = cache["names"].tolist()
            embeddings = cache["embeddings"]
            known = {name: embeddings[i] for i, name in enumerate(names)}
            print(f"Loaded {len(known)} face(s) from cache\n")
            return known

    # Cache miss or photos changed — recompute
    print("Computing embeddings (this only runs when photos change)...")
    raw = {}
    for filename, _ in current_mtimes.items():
        name = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        path = os.path.join(directory, filename)
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend="retinaface"
            )
            embedding = np.array(result[0]["embedding"])
            raw.setdefault(name, []).append(embedding)
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ Skipped {filename}: {e}")

    known = {}
    for name, embeddings in raw.items():
        avg = np.mean(embeddings, axis=0)
        known[name] = avg / np.linalg.norm(avg)
        print(f"  → {name}: {len(embeddings)} photos averaged")

    # Save cache
    names = list(known.keys())
    embeddings_array = np.stack([known[n] for n in names])
    np.savez(cache_file,
             names=names,
             embeddings=embeddings_array,
             mtimes=np.array(current_mtimes, dtype=object))
    print(f"Cache saved to {cache_file}\n")

    return known

print("Loading known faces...")
known_faces = load_known_faces(KNOWN_FACES_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_face_crop(frame, x1, y1, x2, y2):
    """
    Extract a square-ish face crop from the upper portion of a person box.
    
    Strategy:
      - Estimate face height as ~25% of person box height
      - Make the crop square (width == height) centered horizontally
      - Add 20% padding around the estimated face
      - Clamp to frame bounds
    
    This is far more reliable than a flat 40% strip.
    """
    box_h = y2 - y1
    box_w = x2 - x1

    # Face occupies roughly the top 22% of a standing person
    face_h = int(box_h * 0.22)
    face_h = max(face_h, MIN_FACE_SIZE)

    # Add padding
    pad = int(face_h * 0.20)
    face_h_padded = face_h + 2 * pad

    # Square crop centered on the box top-center
    cx = (x1 + x2) // 2
    half = face_h_padded // 2

    fx1 = max(cx - half, 0)
    fx2 = min(cx + half, frame.shape[1])
    fy1 = max(y1 - pad, 0)
    fy2 = min(fy1 + face_h_padded, frame.shape[0])

    crop = frame[fy1:fy2, fx1:fx2]
    return crop

def is_crop_usable(crop):
    """Reject crops that are too small or too dark/blurry to encode reliably."""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return False
    # Blur check: Laplacian variance — blurry images score very low
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
        return False
    return True

def identify_face(face_crop):
    """Returns (name, confidence_str) or ('Unknown', '')"""
    try:
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

        if best_dist < RECOGNITION_THRESHOLD:
            confidence = f"{(1 - best_dist) * 100:.0f}%"
            return best_name.capitalize(), confidence
        return "Unknown", ""

    except Exception:
        return "Unknown", ""

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened. Press Q to quit.")

frame_count = 0
last_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 2 == 0:
        yolo_results = model(frame, classes=[0], verbose=False, imgsz=640)

        if frame_count % RUN_RECOGNITION_EVERY == 0:
            last_detections = []
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    # ── Better crop ──────────────────────────────────────
                    face_crop = extract_face_crop(frame, x1, y1, x2, y2)

                    # ── Quality gate ─────────────────────────────────────
                    if not is_crop_usable(face_crop):
                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "label": "Unknown",
                            "color": (100, 100, 100)   # gray = skipped
                        })
                        continue

                    name, face_conf = identify_face(face_crop)
                    color = (0, 220, 0) if name != "Unknown" else (0, 140, 255)
                    label = f"{name} {face_conf}" if face_conf else "Unknown"

                    last_detections.append({
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "color": color
                    })

    # ── Draw ──────────────────────────────────────────────────────────────
    for det in last_detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, det["label"], (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    known_count = sum(1 for d in last_detections if d["color"] == (0, 220, 0))
    cv2.putText(frame, f"Known: {known_count}  Total: {len(last_detections)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognizer", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()