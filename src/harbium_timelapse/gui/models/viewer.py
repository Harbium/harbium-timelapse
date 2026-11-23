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
MODES = ("after", "before", "wipe", "flicker")
EXAGGERATE = 5                   # Flicker x5: each pixel's difference from its neighbours in time, magnified
AROUND = 2                       # ...those neighbours: this many frames either side


def _srgb_levels() -> np.ndarray:
    from ...finish import _srgb_to_linear
    return _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0)


_LINEAR = None


def _linear() -> np.ndarray:
    global _LINEAR
    if _LINEAR is None:
        _LINEAR = _srgb_levels()
    return _LINEAR


def _lut(stops: float, tone=None) -> np.ndarray:
    """8-bit sRGB level -> the same level with ``stops`` of light added, exactly; with ``tone`` (Finish's
    tone-aware shares) each level gets its own share, as Finish applies it."""
    from ...finish import _linear_to_srgb
    lin = _linear()
    share = tone.at(lin) if tone is not None else 1.0
    exact = _linear_to_srgb(np.clip(lin * 2.0 ** (float(stops) * share), 0, 1)) * 255
    if tone is not None:
        exact = np.maximum.accumulate(exact)
    return np.round(exact).astype(np.uint8)


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

    def __init__(self, frames: list[np.ndarray], gains: np.ndarray | None = None, tone=None):
        self.frames = frames
        self.tone = tone
        self._tables: dict[float, np.ndarray] = {}
        self.set_gains(gains)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h

    def set_gains(self, gains) -> None:
        """One correction per frame (stops), or one per frame and channel (frames x 3: Finish's colour too)."""
        g = np.zeros(len(self.frames)) if gains is None else np.asarray(gains, dtype=np.float64)
        self.gains = np.where(np.isfinite(g), g, 0.0)

    def _table(self, stops: float) -> np.ndarray:
        key = round(float(stops), 4)
        table = self._tables.get(key)
        if table is None:
            if len(self._tables) > 1024:
                self._tables.clear()
            table = self._tables[key] = _lut(key, self.tone)
        return table

    def _after(self, i: int) -> np.ndarray:
        before, g = self.frames[i], self.gains[i]
        if np.ndim(g) == 0:
            return before if g == 0 else self._table(g)[before]
        out = np.empty_like(before)
        for c in range(3):                                         # Finish corrects each channel
            out[..., c] = self._table(g[c])[before[..., c]]
        return out

    def flicker(self, i: int) -> np.ndarray:
        """Flicker x5: each pixel against the same pixel in the frames around it (corrected), the difference
        magnified. Leftover flicker becomes an obvious shimmer; still parts that are steady stay put."""
        lin = _linear().astype(np.float32)
        n = len(self.frames)
        around = [j for j in range(i - AROUND, i + AROUND + 1) if 0 <= j < n and j != i]
        here = lin[self._after(i)]
        if not around:
            return self._after(i)
        local = np.mean([lin[self._after(j)] for j in around], axis=0)
        ratio = np.where(local > 1e-5, here / np.maximum(local, 1e-5), 1.0)
        out = np.clip(local * ratio ** EXAGGERATE, 0, 1)
        from ...finish import _linear_to_srgb
        return np.round(_linear_to_srgb(out) * 255).astype(np.uint8)

    def image(self, i: int, mode: str = "after", wipe: float = 0.5) -> np.ndarray:
        """Frame ``i`` as shown: corrected (after), as measured (before), or before left of the wipe line and
        after right of it."""
        i = int(min(max(i, 0), len(self.frames) - 1))
        before = self.frames[i]
        if mode == "flicker":
            return self.flicker(i)
        after = before if mode == "before" else self._after(i)
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
