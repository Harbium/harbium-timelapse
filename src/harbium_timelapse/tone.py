"""Tone-aware correction: how strongly each brightness level carries the frame's flicker.

Why: a grade that stretches some levels (strong contrast, an S-curve, AgX) stretches the flicker there too. On
James's sunset (R6, RapidRAW, contrast 54 and an S-curve) the highlights carried about twice the frame's flicker,
so one correction per frame left the sky flickering (0.023 stops, in step across the whole sky) while the street
was steady.

So each brightness level gets its own share of the frame's correction: 1 is the frame's own, 2 twice it. Shares
are learnt once per sequence, from tiny samples kept while measuring (no extra reading), against finish's own
measurement (the regional vote). They are used only when the evidence is clear; otherwise nothing changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGES = (5, 25, 45, 65, 85, 98)       # percentiles of the still pixels' levels: five bands, shadows to highlights
NAMES = ("the deepest shadows", "the shadows", "the midtones", "the bright midtones", "the brightest highlights")
MIN_EFFECT = 0.15                     # a share must differ from 1 by more than this to be worth acting on
MOST, LEAST = 3.0, 0.25               # never more than three times, or less than a quarter of, the frame's correction
SAMPLE_EDGE = 128                     # the tiny brightness copy kept per frame


@dataclass(frozen=True)
class ToneShares:
    """Each band's share of the frame's correction, at the band's typical level (log2 of linear light)."""
    levels: tuple[float, ...]
    shares: tuple[float, ...]
    bands: tuple[int, ...] = ()       # which of the five bands each level is (they can be skipped), for naming

    def at(self, linear: np.ndarray) -> np.ndarray:
        """The share for each linear level: blended smoothly between bands, and past the end bands continuing
        their trend (a grade keeps steepening towards white), within LEAST to MOST of the frame's correction."""
        lv = np.log2(np.maximum(np.asarray(linear, dtype=np.float64), 1e-6))
        x, y = np.array(self.levels), np.array(self.shares)
        out = np.interp(lv, x, y)
        low, high = (y[1] - y[0]) / (x[1] - x[0]), (y[-1] - y[-2]) / (x[-1] - x[-2])
        out = np.where(lv < x[0], y[0] + low * (lv - x[0]), out)
        out = np.where(lv > x[-1], y[-1] + high * (lv - x[-1]), out)
        return np.clip(out, LEAST, MOST)

    def describe(self) -> str:
        """Which level got the most correction and which the least: a grade isn't always a simple ramp (on
        James's sunset the bright midtones of the sky glow needed most, the brightest highlights held down)."""
        bands = self.bands if len(self.bands) == len(self.shares) else tuple(range(len(self.shares)))
        most, least = int(np.argmax(self.shares)), int(np.argmin(self.shares))
        name = lambda i: NAMES[bands[i]] if bands[i] < len(NAMES) else f"level {i + 1}"    # noqa: E731
        ratio = self.shares[most] / max(self.shares[least], 1e-6)
        first = name(most)
        return f"{first[0].upper()}{first[1:]} got {ratio:.1f}x the correction of {name(least)}."


def tiny(small_linear_rgb: np.ndarray) -> np.ndarray:
    """A frame's tiny brightness copy, by averaging blocks (grain averages out)."""
    from .diagnose import LUMA
    luma = small_linear_rgb[..., :3] @ LUMA
    step = max(1, max(luma.shape) // SAMPLE_EDGE)
    h, w = (luma.shape[0] // step) * step, (luma.shape[1] // step) * step
    return luma[:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3)).astype(np.float32)


def tone_shares(tinies: list[np.ndarray], reference: np.ndarray) -> ToneShares | None:
    """Each band's share of the measured flicker, or None when every level carries it about equally (or there
    isn't enough to judge): then the correction stays one gain per frame, exactly as before.

    ``reference`` is the frame's measured brightness in stops (finish's regional vote), one per frame."""
    from .diagnose import _still, share
    if len(tinies) < 12 or len({t.shape for t in tinies}) != 1:
        return None
    b = np.log2(np.maximum(np.stack([t.reshape(-1) for t in tinies]).astype(np.float64), 1e-6))
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    if still.sum() < 50:
        return None
    edges = np.percentile(ref[still], EDGES)
    levels, shares, bands, clear = [], [], [], False
    for band, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        mask = still & (ref >= lo) & (ref < hi)
        if mask.sum() < 10:
            continue
        k, se = share(np.median(b[:, mask], axis=1), reference)
        if not np.isfinite(k) or se > 0.25:
            continue                                         # too much else moving at this level to tell
        shrunk = 1 + (k - 1) * (k - 1) ** 2 / ((k - 1) ** 2 + se ** 2)      # unsure estimates lean towards 1
        clear |= abs(shrunk - 1) > MIN_EFFECT and abs(k - 1) > 2 * se
        levels.append(float(np.median(ref[mask])))
        shares.append(float(np.clip(shrunk, LEAST, MOST)))
        bands.append(band)
    if not clear or len(levels) < 2:
        return None
    return ToneShares(tuple(levels), tuple(shares), tuple(bands))
