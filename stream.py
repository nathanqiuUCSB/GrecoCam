"""Lightweight MJPEG preview server for viewing the feed over the network/SSH.

Enable in detector.py with STREAM_PREVIEW=1. Open in a browser:
  http://<orin-ip>:8080/
"""

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_latest_jpeg = None
_lock = threading.Lock()
_boundary = b"frame"


def update_frame(frame, quality=80):
    """Encode the given BGR frame as JPEG and publish it to connected viewers."""
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return
    with _lock:
        _latest_jpeg = buf.tobytes()


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_index()
        elif path in ("/stream", "/stream.mjpg", "/video"):
            self._serve_mjpeg()
        elif path == "/snapshot.jpg":
            self._serve_snapshot()
        else:
            self.send_error(404)

    def _serve_index(self):
        html = b"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>GrecoCam</title>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
  h1 { font-size: 1rem; font-weight: normal; padding: 8px 12px; margin: 0; }
  img { display: block; width: 100%; max-width: 1280px; margin: 0 auto; }
</style>
</head><body>
<h1>GrecoCam live preview</h1>
<img src="/stream" alt="camera stream">
</body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_snapshot(self):
        with _lock:
            jpeg = _latest_jpeg
        if not jpeg:
            self.send_error(503, "No frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_boundary.decode()}",
        )
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        last_sent = None
        try:
            while True:
                with _lock:
                    jpeg = _latest_jpeg
                if jpeg is None or jpeg is last_sent:
                    time.sleep(0.03)
                    continue
                last_sent = jpeg
                part = (
                    b"--" + _boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                self.wfile.write(part)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def start_server(port=8080):
    """Start the MJPEG server in a background daemon thread."""
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ip = _local_ip()
    print(f"MJPEG preview: http://{ip}:{port}/")
    print(f"  snapshot:    http://{ip}:{port}/snapshot.jpg")
    return server
