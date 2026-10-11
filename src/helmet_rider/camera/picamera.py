"""Optional Raspberry Pi Camera Module adapter using system Picamera2."""

from __future__ import annotations


class PiCameraSource:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 25) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not available. Install it with apt and use a "
                "system-site-packages virtual environment; see the Raspberry Pi README."
            ) from exc
        self.camera = Picamera2()
        configuration = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": fps},
            buffer_count=2,
        )
        self.camera.configure(configuration)
        self.camera.start()

    def read(self):
        try:
            frame = self.camera.capture_array("main")
        except Exception:
            return False, None
        return frame is not None, frame

    def release(self) -> None:
        self.camera.stop()
