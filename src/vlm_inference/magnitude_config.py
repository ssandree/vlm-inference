"""Settings for within-tube motion-magnitude FPS. No sampling imports."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Absolute MU frame-mean cuts. Equal to the 33rd and 67th percentiles of
# positive Stage3 frame means on the three dry-run VIRAT clips
# (010000_07, 000200_00, 000200_01). Override per experiment.
DEFAULT_MAGNITUDE_LOW = 2.0227
DEFAULT_MAGNITUDE_HIGH = 5.0992


@dataclass(frozen=True)
class MagnitudeAdaptiveConfig:
    """Within-tube FPS from the stored Stage3 magnitude series.

    ``absolute`` compares each frame mean to ``low_threshold`` / ``high_threshold``.
    ``tube_quantile`` uses that tube's own positive-magnitude percentiles instead.
    """

    low_threshold: float = DEFAULT_MAGNITUDE_LOW
    high_threshold: float = DEFAULT_MAGNITUDE_HIGH
    min_fps: float = 0.5
    mid_fps: float = 1.0
    max_fps: float = 2.0
    threshold_source: str = "absolute"
    low_quantile: float = 100.0 / 3.0
    high_quantile: float = 200.0 / 3.0

    def __post_init__(self) -> None:
        rates = (self.min_fps, self.mid_fps, self.max_fps)
        if not all(math.isfinite(v) and v > 0 for v in rates):
            raise ValueError("magnitude FPS values must be finite and positive")
        if not self.min_fps <= self.mid_fps <= self.max_fps:
            raise ValueError("require min_fps <= mid_fps <= max_fps")
        source = str(self.threshold_source)
        if source not in {"absolute", "tube_quantile"}:
            raise ValueError("threshold_source must be absolute or tube_quantile")
        if source == "absolute":
            if not math.isfinite(self.low_threshold) or not math.isfinite(self.high_threshold):
                raise ValueError("magnitude thresholds must be finite")
            if self.low_threshold > self.high_threshold:
                raise ValueError("low_threshold must be <= high_threshold")
        elif not 0 <= self.low_quantile < self.high_quantile <= 100:
            raise ValueError("require 0 <= low_quantile < high_quantile <= 100")
