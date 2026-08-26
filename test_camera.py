"""Grab one frame and save test.jpg (for SSH / headless testing on Orin Nano)."""

import sys

import cv2

from camera import open_camera, opencv_has_gstreamer, use_jetson_camera

WARMUP_FRAMES = 10
OUTPUT_FILE = "test.jpg"


def main():
    if use_jetson_camera():
        print(f"OpenCV GStreamer support: {'yes' if opencv_has_gstreamer() else 'NO'}")

    cap = open_camera()
    if not cap.isOpened():
        print("Error: Could not open camera")
        if use_jetson_camera():
            print("Orin Nano tips:")
            print("  - Enable the camera: sudo /opt/nvidia/jetson-io/jetson-io.py")
            print("  - Check ribbon cable orientation")
            print("  - Test: gst-launch-1.0 nvarguscamerasrc ! fakesink")
        sys.exit(1)

    for _ in range(WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read a frame")
        sys.exit(1)

    cv2.imwrite(OUTPUT_FILE, frame)
    h, w = frame.shape[:2]
    print(f"Saved {OUTPUT_FILE} ({w}x{h})")
    print("Copy to your PC: scp user@orin-ip:~/GrecoCam/test.jpg .")


if __name__ == "__main__":
    main()
