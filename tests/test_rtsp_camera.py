import threading

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import api


class FakeRTSPCamera:
    def __init__(self, url):
        self.url = url
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        if self.reads == 1:
            return True, np.zeros((24, 32, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_rtsp_url_validation_accepts_loopback_and_rejects_public_host(monkeypatch):
    api._validate_rtsp_url("rtsp://127.0.0.1:8554/live")

    monkeypatch.setattr(
        api.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("8.8.8.8", 554))],
    )
    with pytest.raises(HTTPException) as error:
        api._validate_rtsp_url("rtsp://camera.example/live")
    assert "local network" in error.value.detail


def test_rtsp_url_validation_rejects_invalid_scheme_and_port():
    with pytest.raises(HTTPException) as error:
        api._validate_rtsp_url("https://192.168.1.20/live")
    assert "rtsp://" in error.value.detail
    with pytest.raises(HTTPException) as error:
        api._validate_rtsp_url("rtsp://192.168.1.20:0/live")
    assert "port" in error.value.detail


def test_rtsp_session_streams_and_releases_camera(monkeypatch):
    opened = []

    def open_camera(url):
        camera = FakeRTSPCamera(url)
        opened.append(camera)
        return camera

    monkeypatch.setattr(api, "RTSPCamera", open_camera)
    monkeypatch.setattr(api, "get_pipeline", lambda: (_ for _ in ()).throw(FileNotFoundError("no model")))

    with TestClient(api.app) as client:
        response = client.post(
            "/api/camera/rtsp/session",
            json={"url": "rtsp://127.0.0.1:8554/live"},
        )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        assert "127.0.0.1" not in session_id

        stream = client.get(f"/api/camera/rtsp/stream/{session_id}")
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert b"Content-Type: image/jpeg" in stream.content
        assert opened[0].released


def test_stop_endpoint_signals_active_camera_stream(monkeypatch):
    stop_event = threading.Event()
    monkeypatch.setattr(api, "_camera_stream_sessions", {"active-test": stop_event})
    monkeypatch.setattr(api, "_active_camera_stream_session_id", "active-test")
    monkeypatch.setattr(
        api,
        "_camera_stats",
        {"active": True, "preview_only": False, "summary": {}, "timings": {}},
    )

    with TestClient(api.app) as client:
        response = client.post("/api/camera/stop", json={"session_id": "active-test"})

    assert response.status_code == 200
    assert response.json()["stopped"] is True
    assert stop_event.is_set()
    assert api._camera_stats["active"] is False


def test_stop_endpoint_releases_pending_rtsp_session(monkeypatch):
    camera = FakeRTSPCamera("rtsp://127.0.0.1:8554/live")
    timer = threading.Timer(45, lambda: None)
    monkeypatch.setattr(api, "_rtsp_sessions", {"pending-test": (camera, timer)})
    monkeypatch.setattr(api, "_camera_stream_sessions", {})
    monkeypatch.setattr(api, "_active_camera_stream_session_id", None)
    monkeypatch.setattr(api, "_camera_stats", {"active": False, "summary": {}, "timings": {}})

    with TestClient(api.app) as client:
        response = client.post("/api/camera/stop", json={"session_id": "pending-test"})

    assert response.status_code == 200
    assert response.json()["stopped"] is True
    assert camera.released


def test_web_pipeline_resets_between_stills_and_stream_sessions(monkeypatch):
    class CountingPipeline:
        def __init__(self):
            self.resets = 0
            self.frames = 0

        def reset(self):
            self.resets += 1

        def process(self, frame, *, temporal_consistency=False):
            self.frames += 1
            return self.frames

    pipeline = CountingPipeline()
    monkeypatch.setattr(api, "_active_analysis_session_id", None)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    api.process_for_web(pipeline, frame)
    api.process_for_web(pipeline, frame)
    assert pipeline.resets == 2

    api.process_for_web(pipeline, frame, "video-1")
    api.process_for_web(pipeline, frame, "video-1")
    assert pipeline.resets == 3

    api.process_for_web(pipeline, frame, "camera-1")
    assert pipeline.resets == 4
