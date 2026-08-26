"""Save visit records with face + scene photos for later accuracy review."""

import json
import os
import time

import cv2

VISITS_DIR = os.environ.get("VISITS_DIR", "visits")


def _enabled():
    v = os.environ.get("SAVE_VISITS")
    if v is None:
        return True
    return v.lower() in ("1", "true", "yes")


class VisitLogger:
    def __init__(self, directory=VISITS_DIR):
        self.directory = directory
        self.jsonl_path = os.path.join(directory, "visits.jsonl")
        self.enabled = _enabled()
        self._visit_id = None
        self._best_area = 0
        self._best_face = None
        self._best_scene = None
        self._best_box = None
        self._best_name = None
        self._best_conf = None
        self._vote_history = []

    def on_visit_start(self):
        if not self.enabled:
            return
        self._visit_id = time.strftime("%Y%m%d_%H%M%S")
        self._best_area = 0
        self._best_face = None
        self._best_scene = None
        self._best_box = None
        self._best_name = None
        self._best_conf = None
        self._vote_history = []

    def note_recognition(self, face_crop, frame, box, name, conf):
        """Keep the largest face snapshot from this visit."""
        if not self.enabled or face_crop is None or face_crop.size == 0:
            return
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= self._best_area:
            return
        self._best_area = area
        self._best_face = face_crop.copy()
        self._best_box = box
        self._best_name = name
        self._best_conf = conf
        scene = frame.copy()
        color = (0, 220, 0) if name not in ("Unknown", None) else (0, 140, 255)
        cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf}".strip() if name else ""
        if label:
            cv2.putText(scene, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self._best_scene = scene

    def note_vote(self, name):
        if self.enabled and name:
            self._vote_history.append(name)

    def _write(self, outcome, name, conf, elapsed_sec, announced):
        if not self.enabled or self._best_face is None:
            return
        visit_id = self._visit_id or time.strftime("%Y%m%d_%H%M%S")
        detected = name or self._best_name or "none"
        name_slug = detected.lower().replace(" ", "_")
        folder = os.path.join(self.directory, f"{visit_id}_{name_slug}")
        os.makedirs(folder, exist_ok=True)

        face_path = os.path.join(folder, "face.jpg")
        scene_path = os.path.join(folder, "scene.jpg")
        cv2.imwrite(face_path, self._best_face)
        if self._best_scene is not None:
            cv2.imwrite(scene_path, self._best_scene)

        record = {
            "id": visit_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "outcome": outcome,
            "announced": announced,
            "name": detected if detected != "none" else None,
            "confidence": conf or self._best_conf,
            "elapsed_sec": round(elapsed_sec, 2),
            "votes": list(self._vote_history),
            "face": "face.jpg",
            "scene": "scene.jpg" if self._best_scene is not None else None,
        }
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        record["face"] = os.path.relpath(face_path, self.directory)
        if self._best_scene is not None:
            record["scene"] = os.path.relpath(scene_path, self.directory)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] Visit saved → {folder}/")
        self._reset_snapshots()

    def save_announced(self, name, conf, elapsed_sec):
        self._write("announced", name, conf, elapsed_sec, announced=True)

    def save_no_announce(self, elapsed_sec):
        if not self.enabled or self._best_face is None:
            return
        # Summarize what the camera saw even if it never triggered audio.
        votes = self._vote_history
        best_guess = max(set(votes), key=votes.count) if votes else None
        self._write("no_announce", best_guess, self._best_conf, elapsed_sec, announced=False)

    def _reset_snapshots(self):
        self._best_area = 0
        self._best_face = None
        self._best_scene = None
        self._best_box = None
