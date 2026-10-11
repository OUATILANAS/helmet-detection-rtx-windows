"""OpenCV reader for network CCTV streams using RTSP or RTSPS."""

from __future__ import annotations

import cv2


class RTSPCamera:
    """Read an RTSP stream with bounded FFmpeg connection/read timeouts."""

    def __init__(self, url: str, open_timeout_ms: int = 5000, read_timeout_ms: int = 5000):
        self.capture = None
        try:
            timeout_params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms,
            ]
            self.capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, timeout_params)
        except (AttributeError, cv2.error, TypeError):
            # Older OpenCV builds may not expose FFmpeg timeout parameters.
            self.capture = cv2.VideoCapture(url)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "Could not open the RTSP stream. Check the URL, credentials, and camera reachability."
            )
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.capture.read()
        return ok, frame if ok else None

    def release(self) -> None:
        self.capture.release()
