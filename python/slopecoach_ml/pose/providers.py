from __future__ import annotations

from typing import Protocol

from .contracts import FrameGeometry, PoseFrame


class PoseProvider(Protocol):
    name: str

    def estimate(self, frame: object, geometry: FrameGeometry) -> PoseFrame: ...


class MockPoseProvider:
    name = "mock-pose"

    def __init__(self, pose_frame: PoseFrame) -> None:
        pose_frame.validate()
        self._pose_frame = pose_frame

    def estimate(self, frame: object, geometry: FrameGeometry) -> PoseFrame:
        geometry.validate()
        if geometry != self._pose_frame.geometry:
            raise ValueError("mock pose geometry does not match requested frame geometry")
        return self._pose_frame
