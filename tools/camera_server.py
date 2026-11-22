"""Serve a local camera as a single-client MJPEG stream on a trusted network."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import cv2

from config import CaptureConfig
from ingestion.capture import CameraCapture, MAX_CONSECUTIVE_FAILURES
from ingestion.exceptions import CameraError


def handler_for(config: CaptureConfig):
    class CameraStream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if urlsplit(self.path).path != "/stream":
                self.send_error(404)
                return

            self.connection.settimeout(5)
            started = False
            try:
                with CameraCapture(config) as camera:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    started = True
                    failures = 0
                    while failures < MAX_CONSECUTIVE_FAILURES:
                        frame = camera.read()
                        if frame is None:
                            failures += 1
                            continue
                        failures = 0
                        ok, jpeg = cv2.imencode(".jpg", frame)
                        if not ok:
                            break
                        data = jpeg.tobytes()
                        self.wfile.write(
                            f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
            except (CameraError, cv2.error):
                if not started:
                    self.send_error(503, "Camera unavailable")
            except OSError:
                # The viewer disconnected or stopped reading.
                pass

    return CameraStream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", help="Listening address (default: loopback).")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        config = CaptureConfig(camera_index=args.camera)
        with HTTPServer((args.bind, args.port), handler_for(config)) as server:
            print(f"Camera stream listening on {args.bind}:{args.port}/stream", flush=True)
            server.serve_forever()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Camera server error: {exc}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
