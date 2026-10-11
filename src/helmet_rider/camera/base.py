"""Small common interface for frame sources."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CameraSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
