# GrecoCam

Smart door camera for the **NVIDIA Jetson Orin Nano**. Detects faces with YOLO, recognizes people with DeepFace (Facenet512), and plays personalized audio greetings.

## Features

- Real-time face detection + recognition on CSI camera (IMX219)
- Interactive enrollment with live browser preview
- Confidence-based + temporal voting (fast announce for strong matches)
- Face enhancement for backlit / doorway lighting
- MJPEG preview in the browser (`http://<orin-ip>:8080/`)
- Regular or music-mode audio clips
- Visit logging with face + scene photos

## Hardware

- NVIDIA Jetson Orin Nano Super
- Arducam / Raspberry Pi Camera Module (IMX219) on CSI
- Speakers (Bluetooth or USB)

## Project layout

| File | Role |
|------|------|
| `detector.py` | Main recognition loop + announcements |
| `enroll.py` | Capture enrollment photos from the camera |
| `camera.py` | Jetson CSI / USB camera helpers |
| `audio.py` | Play `sounds/<name>.mp3` (or music clips) |
| `face_prep.py` | Face crop enhancement (CLAHE + brightness) |
| `stream.py` | MJPEG browser preview server |
| `visit_log.py` | Save visits under `visits/` |
| `known_faces/` | Enrollment photos (`name_1.jpg`, …) |
| `sounds/` | Greeting clips (`name.mp3`, `namemusic.mp3`) |

## Setup (Jetson Orin)

**Do not** `pip install opencv-python` on the Orin — it breaks CSI. Use system OpenCV:

```bash
sudo apt install python3-opencv espeak
cd ~/GrecoCam
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install numpy ultralytics deepface edge-tts pygame pyttsx3
```

Enable the IMX219 overlay with Jetson-IO if needed, and use the **super** DTB for Orin Nano Super.

## Run detector

```bash
cd ~/GrecoCam && source venv/bin/activate
export JETSON_WIDTH=1280 JETSON_HEIGHT=720 JETSON_FPS=60

STREAM_PREVIEW=1 python3 detector.py
```

Open the live feed: `http://192.168.0.26:8080/` (use your Orin’s IP).

**Music mode** (plays `sounds/<name>music.mp3` instead of `<name>.mp3`):

```bash
MUSIC_MODE=1 STREAM_PREVIEW=1 python3 detector.py
```

### Keep running after SSH disconnect

```bash
tmux new -s grecocam
# start detector inside tmux
# detach: Ctrl+B, then D
# reattach later: tmux attach -t grecocam
```

## Enroll a person

Stop the detector first (camera + port 8080).

```bash
cd ~/GrecoCam && source venv/bin/activate
export JETSON_WIDTH=1280 JETSON_HEIGHT=720 JETSON_FPS=60

python3 enroll.py name --clear
# Browser: http://<orin-ip>:8080/
# Press Enter in SSH for each shot

rm -f face_cache.npz
STREAM_PREVIEW=1 python3 detector.py
```

Tips:

- Enroll from the **IMX219 at the door**, in poses you use in real life
- Name must be alphanumeric (`nathan`, `ethan`, `mattt`, …)
- Add `sounds/<name>.mp3` (and optional `<name>music.mp3`) for greetings
- `.wav` clips also work

## Audio test

```bash
python3 -c "from audio import announce; announce('nathan')"

# Music mode
python3 -c "from audio import configure, announce; configure(music_mode=True); announce('nathan')"
```

## Visit logs

With `SAVE_VISITS=1` (default), each visit is saved under `visits/`:

```
visits/
  visits.jsonl
  20260806_164530_nathan/
    face.jpg
    scene.jpg
    meta.json
```

Disable with `SAVE_VISITS=0`.

## Useful environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `STREAM_PREVIEW` | `0` | Browser MJPEG preview |
| `MUSIC_MODE` | `0` | Prefer `*music` audio clips |
| `JETSON_WIDTH` / `HEIGHT` / `FPS` | `1280` / `720` / `30` | Capture size / rate |
| `JETSON_EXPOSURE_COMP` | `-1.0` | Exposure bias (`-2`…`2`) |
| `JETSON_AE_REGION` | `auto` | Center-weighted AE (set `off` for full frame) |
| `RECOGNITION_THRESHOLD` | `0.45` (Jetson) | Match distance (lower = stricter) |
| `VOTES_REQUIRED` | `2` | Votes needed for known faces |
| `HIGH_CONF_ANNOUNCE` | `0.80` | Instant announce if similarity ≥ 80% |
| `UNKNOWN_VOTES_REQUIRED` | `4` | Votes needed for Unknown |
| `SAVE_VISITS` | `1` | Log visits with photos |

## PC development

```bash
pip install -r requirements.txt
python detector.py   # USB webcam; local OpenCV window by default
```

## License

MIT (or your preferred license).
