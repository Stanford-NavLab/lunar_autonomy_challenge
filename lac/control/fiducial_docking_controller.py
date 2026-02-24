"""Image-based visual servo controller for docking to a fiducial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import apriltag


@dataclass
class FiducialDockingConfig:
    target_tag_id: int = 69
    max_v: float = 0.18
    max_w: float = 0.50
    k_v: float = 1.6
    k_w: float = 1.2
    target_area_ratio: float = 0.04
    x_tol_norm: float = 0.06
    area_tol_ratio: float = 0.010
    aligned_hold_frames: int = 8
    search_w: float = 0.15
    allow_reverse: bool = False
    yaw_sign: float = -1.0


class FiducialDockingController:
    """Simple tag-centering + approach controller from a single grayscale camera."""

    def __init__(self, config: FiducialDockingConfig | None = None):
        self.cfg = config or FiducialDockingConfig()
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        self._aligned_count = 0

    def reset(self) -> None:
        self._aligned_count = 0

    def _select_target_detection(self, detections: list[Any]) -> Any | None:
        tagged = [d for d in detections if int(getattr(d, "tag_id", -1)) == self.cfg.target_tag_id]
        if not tagged:
            return None
        if len(tagged) == 1:
            return tagged[0]
        return max(tagged, key=lambda d: cv2.contourArea(np.asarray(d.corners, dtype=np.float32)))

    def step(self, gray_img: np.ndarray | None) -> tuple[float, float, bool, dict[str, Any]]:
        """Return (v_cmd, w_cmd, aligned, debug)."""
        if gray_img is None:
            self._aligned_count = 0
            return 0.0, self.cfg.search_w, False, {"target_found": False, "reason": "no_image"}

        detections = self.detector.detect(gray_img)
        det = self._select_target_detection(detections)
        if det is None:
            self._aligned_count = 0
            return (
                0.0,
                self.cfg.search_w,
                False,
                {"target_found": False, "num_detections": len(detections)},
            )

        h, w = gray_img.shape[:2]
        cx = float(det.center[0])
        center_x = 0.5 * float(w)
        x_err_norm = (cx - center_x) / max(center_x, 1.0)

        area_px = float(cv2.contourArea(np.asarray(det.corners, dtype=np.float32)))
        area_ratio = area_px / max(float(w * h), 1.0)
        area_err = self.cfg.target_area_ratio - area_ratio

        w_cmd = float(
            np.clip(self.cfg.yaw_sign * self.cfg.k_w * x_err_norm, -self.cfg.max_w, self.cfg.max_w)
        )
        v_cmd = float(self.cfg.k_v * area_err)
        if not self.cfg.allow_reverse:
            v_cmd = max(v_cmd, 0.0)
        v_cmd = float(np.clip(v_cmd, -self.cfg.max_v, self.cfg.max_v))

        is_centered = abs(x_err_norm) <= self.cfg.x_tol_norm
        is_at_distance = abs(area_err) <= self.cfg.area_tol_ratio
        if is_centered and is_at_distance:
            self._aligned_count += 1
        else:
            self._aligned_count = 0
        aligned = self._aligned_count >= self.cfg.aligned_hold_frames
        if aligned:
            v_cmd, w_cmd = 0.0, 0.0

        debug = {
            "target_found": True,
            "tag_id": int(det.tag_id),
            "x_err_norm": float(x_err_norm),
            "area_ratio": float(area_ratio),
            "area_err": float(area_err),
            "aligned_count": int(self._aligned_count),
            "aligned": bool(aligned),
            "num_detections": len(detections),
        }
        return v_cmd, w_cmd, aligned, debug
