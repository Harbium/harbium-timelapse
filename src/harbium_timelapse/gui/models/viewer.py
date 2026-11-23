"""The sequence viewer's frames: what the big picture above the timeline shows, and plays.

Each frame's preview (1,600 px, linear light, saved while measuring) is reduced to ``WIDTH`` pixels and kept as an
8-bit picture, all at one brightness scale so frames compare honestly. "After" is made from "Before" through a
256-entry lookup table per frame: showing is a multiplication of linear light, so the frame's correction (in
stops) maps each 8-bit level exactly. A millisecond or two a frame, so playback at 24 fps has room to spare.

At the Grade step these are the RAW files with Prepare's corrections: RapidRAW's grade can't be rendered live.
"""

from __future__ import annotations

import numpy as np

WIDTH = 640                      # about 0.8 MB a frame at 3:2; 229 frames ~190 MB
MODES = ("after", "before", "wipe")


def _srgb_levels() -> np.ndarray:
    from ...finish import _srgb_to_linear
    return _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0)


_LINEAR = None


def _lut(stops: float) -> np.ndarray:
    """8-bit sRGB level -> the same level with ``stops`` of light added, exactly."""
    global _LINEAR
    from ...finish import _linear_to_srgb
    if _LINEAR is None:
        _LINEAR = _srgb_levels()
    return np.round(_linear_to_srgb(np.clip(_LINEAR * 2.0 ** float(stops), 0, 1)) * 255).astype(np.uint8)


def reduce(linear: np.ndarray, scale: float, width: int = WIDTH) -> np.ndarray:
    """A preview (linear light) as a small 8-bit picture at one brightness scale."""
    from PIL import Image

    from ...timeslice import to_display
    picture = to_display(linear, scale)
    h, w = picture.shape[:2]
    if w > width:
        picture = np.asarray(Image.fromarray(picture).resize((width, max(1, round(h * width / w))),
                                                              Image.BILINEAR))
    return np.ascontiguousarray(picture)


class SequenceFrames:
    """The frames the viewer shows, and each one's correction (stops)."""

    def __init__(self, frames: list[np.ndarray], gains: np.ndarray | None = None):
        self.frames = frames
        self.gains = np.zeros(len(frames)) if gains is None else np.asarray(gains, dtype=np.float64)
        self._tables: dict[float, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h

    def set_gains(self, gains) -> None:
        g = np.zeros(len(self.frames)) if gains is None else np.asarray(gains, dtype=np.float64)
        self.gains = np.where(np.isfinite(g), g, 0.0)

    def _table(self, stops: float) -> np.ndarray:
        key = round(float(stops), 4)
        table = self._tables.get(key)
        if table is None:
            if len(self._tables) > 512:
                self._tables.clear()
            table = self._tables[key] = _lut(key)
        return table

    def image(self, i: int, mode: str = "after", wipe: float = 0.5) -> np.ndarray:
        """Frame ``i`` as shown: corrected (after), as measured (before), or before left of the wipe line and
        after right of it."""
        i = int(min(max(i, 0), len(self.frames) - 1))
        before = self.frames[i]
        if mode == "before" or self.gains[i] == 0:
            after = before
        else:
            after = self._table(self.gains[i])[before]
        if mode == "after":
            return after
        if mode == "before":
            return before
        cut = int(round(min(max(wipe, 0.0), 1.0) * before.shape[1]))
        out = after.copy()
        out[:, :cut] = before[:, :cut]
        if 0 < cut < before.shape[1]:
            out[:, max(0, cut - 1):cut + 1] = 255                  # the wipe line
        return out
