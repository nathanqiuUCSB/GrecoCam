import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os

# ── Config ────────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
CACHE_FILE = "face_cache.npz"
RECOGNITION_THRESHOLD = 0.32
RUN_RECOGNITION_EVERY = 8
# ──────────────────────────────────────────────────────────────────────────────

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
print(f"Loaded {len(known_faces)} face(s)\n")

# ── Recognition ───────────────────────────────────────────────────────────────
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify_face(face_crop):
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
            return best_name.capitalize(), f"{(1 - best_dist) * 100:.0f}%"
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
        face_results = model(frame, verbose=False, imgsz=640)

        if frame_count % RUN_RECOGNITION_EVERY == 0:
            last_detections = []
            for result in face_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    name, face_conf = identify_face(face_crop)
                    color = (0, 220, 0) if name != "Unknown" else (0, 140, 255)
                    label = f"{name} {face_conf}" if face_conf else "Unknown"

                    last_detections.append({
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "color": color
                    })

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