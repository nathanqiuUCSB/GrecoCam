import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os

# ── Config ────────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"   # folder with one jpg per person
RECOGNITION_THRESHOLD = 0.40     # cosine distance — lower = stricter match
RUN_RECOGNITION_EVERY = 10       # run face recognition every N frames (perf)
# ──────────────────────────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")

# ── Load known face embeddings on startup ─────────────────────────────────────
def load_known_faces(directory):
    known = {}
    raw = {}  # name → list of embeddings

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # "ryan_1.jpg" → "ryan",  "sarah_2.jpg" → "sarah"
            name = os.path.splitext(filename)[0]  # "ryan_1"
            name = name.rsplit("_", 1)[0]         # "ryan"
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

    # Average all embeddings per person into one
    for name, embeddings in raw.items():
        avg = np.mean(embeddings, axis=0)
        known[name] = avg / np.linalg.norm(avg)  # normalize
        print(f"  → {name}: {len(embeddings)} photos averaged")

    return known


print("Loading known faces...")
known_faces = load_known_faces(KNOWN_FACES_DIR)
print(f"Loaded {len(known_faces)} face(s)\n")

# ── Cosine similarity helper ───────────────────────────────────────────────────
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify_face(face_crop):
    """Returns (name, confidence_str) or ('Unknown', '')"""
    try:
        result = DeepFace.represent(
            img_path=face_crop,
            model_name="Facenet512",
            enforce_detection=False,  # already cropped, skip re-detection
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
last_detections = []   # list of dicts: {box, label, color}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Run YOLO every other frame ─────────────────────────────────────────
    if frame_count % 2 == 0:
        yolo_results = model(frame, classes=[0], verbose=False, imgsz=640)

        # ── Run face recognition every N frames ────────────────────────────
        if frame_count % RUN_RECOGNITION_EVERY == 0:
            last_detections = []
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    # Crop the person bounding box for face recognition
                    # Take upper 40% — that's where the face usually is
                    face_y2 = y1 + int((y2 - y1) * 0.40)
                    face_crop = frame[y1:face_y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    name, face_conf = identify_face(face_crop)
                    color = (0, 255, 0) if name != "Unknown" else (0, 140, 255)
                    label = f"{name} {face_conf}" if face_conf else "Unknown"

                    last_detections.append({
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "color": color
                    })

    # ── Draw ───────────────────────────────────────────────────────────────
    for det in last_detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, det["label"], (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    known_count = sum(1 for d in last_detections if d["color"] == (0, 255, 0))
    cv2.putText(frame, f"Known: {known_count}  Total: {len(last_detections)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognizer", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

## How it works
