"""Lightweight online lateral drift bias estimator for docking approach."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class LateralDriftCompensatorConfig:
    enabled: bool = False
    beta: float = 0.08
    max_bias_m: float = 0.35
    x_window_m: float = 0.70
    min_speed_mps: float = 0.03
    gain: float = 0.8
    decay: float = 0.98


class LateralDriftCompensator:
    """Estimate persistent lateral bias and compensate e_y for control."""

    def __init__(self, cfg: LateralDriftCompensatorConfig | None = None):
        self.cfg = cfg or LateralDriftCompensatorConfig()
        self.bias_y_m = 0.0

    def reset(self) -> None:
        self.bias_y_m = 0.0

    def compensate(self, e_y_m: float) -> float:
        if not self.cfg.enabled:
            return float(e_y_m)
        # Add same-sign bias to amplify correction against persistent drift.
        return float(e_y_m + self.cfg.gain * self.bias_y_m)

    def update(
        self, stage: str, ex_m: float, ey_m: float, v_ref_mps: float, meas_valid: bool
    ) -> None:
        if not self.cfg.enabled:
            return

        # No reliable measurement or not in approach: slowly forget.
        if (not meas_valid) or stage != "approach":
            self.bias_y_m = float(self.bias_y_m * self.cfg.decay)
            return

        # Learn only when close enough in range and actually moving.
        if abs(ex_m) <= self.cfg.x_window_m and abs(v_ref_mps) >= self.cfg.min_speed_mps:
            target = _clamp(float(ey_m), -self.cfg.max_bias_m, self.cfg.max_bias_m)
            self.bias_y_m = float((1.0 - self.cfg.beta) * self.bias_y_m + self.cfg.beta * target)
        else:
            self.bias_y_m = float(self.bias_y_m * self.cfg.decay)
