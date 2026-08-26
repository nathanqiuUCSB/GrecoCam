"""Camera helpers for PC (USB) and Jetson Orin Nano (CSI IMX219)."""

import os
import cv2

CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND")  # "jetson" | "usb" | unset = auto

JETSON_SENSOR_ID = int(os.environ.get("JETSON_SENSOR_ID", "0"))
JETSON_WIDTH = int(os.environ.get("JETSON_WIDTH", "1280"))
JETSON_HEIGHT = int(os.environ.get("JETSON_HEIGHT", "720"))
JETSON_FPS = int(os.environ.get("JETSON_FPS", "30"))
JETSON_FLIP = int(os.environ.get("JETSON_FLIP", "0"))
# Negative = darker image (helps when door opens and outdoor light blows out faces).
# Typical door-cam range: -1.5 to -0.5. Set JETSON_EXPOSURE_COMP=0 for stock AE.
JETSON_EXPOSURE_COMP = float(os.environ.get("JETSON_EXPOSURE_COMP", "-1.0"))
JETSON_AE_LOCK = os.environ.get("JETSON_AE_LOCK", "").lower() in ("1", "true", "yes")
# Auto-exposure region: "auto" = center-weighted (ignores bright windows at edges),
# "off" = full-frame AE, or "left top right bottom weight" in pixels.
JETSON_AE_REGION = os.environ.get("JETSON_AE_REGION", "auto")

USB_INDEX = int(os.environ.get("USB_CAMERA_INDEX", "0"))
USB_WIDTH = int(os.environ.get("USB_WIDTH", "640"))
USB_HEIGHT = int(os.environ.get("USB_HEIGHT", "480"))
USB_FPS = int(os.environ.get("USB_FPS", "30"))


def jetson_model():
    """Return Jetson board name, or None on a PC."""
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        with open(model_path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip("\x00").strip()
    return None


def is_jetson():
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    model = jetson_model()
    return model is not None and "jetson" in model.lower()


def use_jetson_camera():
    if CAMERA_BACKEND == "jetson":
        return True
    if CAMERA_BACKEND == "usb":
        return False
    return is_jetson()


def ae_region_string():
    """Return nvarguscamerasrc aeregion value, or None to use full-frame AE."""
    setting = JETSON_AE_REGION.strip()
    if not setting or setting.lower() in ("off", "0", "false", "no"):
        return None
    if setting.lower() == "auto":
        # Center 50% x 70% — meter on the person at the door, not side windows.
        left = int(JETSON_WIDTH * 0.25)
        top = int(JETSON_HEIGHT * 0.15)
        right = int(JETSON_WIDTH * 0.75)
        bottom = int(JETSON_HEIGHT * 0.85)
        return f"{left} {top} {right} {bottom} 1.0"
    return setting


def gstreamer_pipeline(
    sensor_id=JETSON_SENSOR_ID,
    capture_width=JETSON_WIDTH,
    capture_height=JETSON_HEIGHT,
    display_width=JETSON_WIDTH,
    display_height=JETSON_HEIGHT,
    framerate=JETSON_FPS,
    flip_method=JETSON_FLIP,
    exposure_comp=JETSON_EXPOSURE_COMP,
    ae_lock=JETSON_AE_LOCK,
    ae_region=None,
):
    if ae_region is None:
        ae_region = ae_region_string()
    src = f"nvarguscamerasrc sensor-id={sensor_id}"
    # exposurecompensation: -2..2 — pull down so bright doorways don't wipe faces
    if exposure_comp != 0:
        src += f" exposurecompensation={exposure_comp}"
    if ae_lock:
        src += " aelock=true"
    if ae_region:
        src += f' aeregion="{ae_region}"'
    return (
        f"{src} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def opencv_has_gstreamer():
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if line.strip().startswith("GStreamer:"):
            return "YES" in line
    return False


def open_camera(exposure_comp=None):
    if use_jetson_camera():
        model = jetson_model() or "Jetson"
        print(f"Detected {model}")

        if not opencv_has_gstreamer():
            print("WARNING: OpenCV was built without GStreamer support.")
            print("  On Orin Nano, do NOT use: pip install opencv-python")
            print("  Use system OpenCV instead: sudo apt install python3-opencv")
            print("  Then run with the system python or a venv with --system-site-packages")

        if exposure_comp is None:
            exposure_comp = JETSON_EXPOSURE_COMP
        ae_region = ae_region_string()
        pipeline = gstreamer_pipeline(exposure_comp=exposure_comp, ae_region=ae_region)
        ae_msg = ae_region if ae_region else "full frame"
        print(f"Opening CSI camera (sensor {JETSON_SENSOR_ID}, "
              f"{JETSON_WIDTH}x{JETSON_HEIGHT}, "
              f"exposure_comp={exposure_comp}, ae_region={ae_msg})...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print(f"Opening USB camera (index {USB_INDEX}, {USB_WIDTH}x{USB_HEIGHT})...")
        cap = cv2.VideoCapture(USB_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, USB_FPS)
    return cap
