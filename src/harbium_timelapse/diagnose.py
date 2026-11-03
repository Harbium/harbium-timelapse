"""What's actually going on in a sequence: plain-English findings from an export.

While finish writes, every frame leaves a tiny brightness sample (about 160
points across, each the average of a small patch of pixels, so grain is about
6x lower than single pixels and it still costs only milliseconds), before and
after correction. From those and the plan, four checks tell real change from flicker
from editing side effects:

- sections: how fast the light really changes at the start, middle and end,
  and how much flicker was left, so steep real change isn't mistaken for flicker;
- tones: whether shadows, midtones and highlights had the same flicker removed,
  judged on still pixels only (leaves, walkers and lights set aside), so a tone
  curve that compressed one level would show as over-correction there;
- regions: areas that didn't share the frame's flicker (lit by steady lamps,
  say), where a whole-frame correction adds small steps; marked on a thumbnail;
- blends: places where the light's rate of change switches abruptly, the usual
  sign of a keyframe or blend in the edit.

Each finding says what it means, is marked fine ("ok"), worth knowing ("note")
or worth a look ("look"), and has a "what to do" line: concrete steps that are
known to work, or plainly "nothing to fix" when it's real change. When the data
can't tell, it says so rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TINY = 160                      # long edge of each frame's brightness sample, in pixels
PATCH = 6                       # each sample point averages a PATCH x PATCH patch: grain about 6x lower
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
BANDS = (("shadows", 10, 40), ("midtones", 40, 75), ("highlights", 75, 97))
GRID = (6, 8)                   # rows, columns of regions


@dataclass
class Finding:
    kind: str                                   # "ok", "note" or "look"
    title: str
    detail: str
    fix: str = ""                               # what to do about it, in plain steps
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)   # (top, left, bottom, right), 0-1
    frames: list[int] = field(default_factory=list)                                 # 0-based


# --- gathering, while the export runs -------------------------------------------------------------

def sample(img: np.ndarray) -> np.ndarray:
    """A frame's tiny brightness sample (linear light): about TINY points across, each the average of a PATCH x
    PATCH patch of pixels, so grain is much lower than taking single pixels. A few milliseconds."""
    from .finish import _levels
    from .measure import _srgb_to_linear
    h, w = img.shape[:2]
    step = max(1, max(h, w) // TINY)
    size = min(PATCH, step)
    nh, nw = (h - size) // step + 1, (w - size) // step + 1
    lut = _srgb_to_linear(np.arange(_levels(img) + 1, dtype=np.float64) / _levels(img)).astype(np.float32)
    total = np.zeros((nh, nw), np.float32)
    for dy in range(size):
        for dx in range(size):
            patch = img[dy:dy + (nh - 1) * step + 1:step, dx:dx + (nw - 1) * step + 1:step, :3]
            total += np.take(lut, patch) @ LUMA
    return total / (size * size)


class Recorder:
    """Collects the samples while finish writes (pass as ``apply_finish(..., diagnose=recorder)``)."""

    def __init__(self):
        self.before: list[np.ndarray] = []
        self.after: list[np.ndarray] = []
        self.thumbnail: np.ndarray | None = None          # the middle frame in colour, for marking regions

    def add(self, before_img: np.ndarray, after_img: np.ndarray, index: int, total: int) -> None:
        self.before.append(sample(before_img))
        self.after.append(sample(after_img))
        if index == total // 2:
            from .timeslice import to_8bit
            step = max(1, max(before_img.shape[:2]) // 320)
            self.thumbnail = to_8bit(after_img[::step, ::step, :3]).copy()


# --- the checks -------------------------------------------------------------------------------------

def jitter(values: np.ndarray) -> float:
    """Frame-to-frame wobble with the real trend taken out (second differences), in stops."""
    d = np.diff(np.asarray(values, dtype=np.float64), 2)
    d = d[np.isfinite(d)]
    return float(np.std(d) / np.sqrt(6)) if d.size > 2 else 0.0


def _logs(stack: list[np.ndarray]) -> np.ndarray:
    return np.log2(np.maximum(np.stack([s.ravel() for s in stack]), 1e-6))         # frames x pixels


def _still(before: np.ndarray, usable: np.ndarray, bins: int = 12) -> np.ndarray:
    """Pixels that don't change chaotically: set aside swaying leaves, walkers and lights switching.

    Each pixel is judged on its own frame-to-frame wobble, against what grain alone gives a pixel of the same
    brightness (learnt from the footage: dark pixels are grainier). So noisy shadows stay in, and only pixels
    wobbling well beyond their grain are set aside. (Not judged on how a pixel differs from the frame: areas
    that don't share the frame's flicker, like lamp-lit ones, are exactly what the checks need to see.)"""
    wobble = np.std(np.diff(before, 2, axis=0), axis=0)
    level = np.median(before, axis=0)                       # each pixel's typical brightness, log2
    edges = np.percentile(level[usable], np.linspace(0, 100, bins + 1))
    which = np.clip(np.searchsorted(edges, level, side="right") - 1, 0, bins - 1)
    expected = np.zeros(bins)
    for i in range(bins):
        in_bin = usable & (which == i)
        expected[i] = np.median(wobble[in_bin]) if in_bin.any() else np.median(wobble[usable])
    return usable & (wobble <= 2.5 * expected[which] + 1e-4)


def _flicker(before: np.ndarray, still: np.ndarray) -> float:
    return jitter(np.median(before[:, still], axis=1))


def _removed(jb: float, ja: float) -> tuple[float, float]:
    """Flicker removed and jitter added: independent wobbles add as squares."""
    return float(np.sqrt(max(jb * jb - ja * ja, 0))), float(np.sqrt(max(ja * ja - jb * jb, 0)))


def check_tones(rec: Recorder) -> Finding:
    b, a = _logs(rec.before), _logs(rec.after)
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    flicker = _flicker(b, still)
    if flicker < 0.004:
        return Finding("ok", "Tones: almost no flicker to judge",
                       f"The still parts of the frame wobbled by only {flicker:.3f} stops before correction.")
    edges = np.percentile(ref[still], [p for _, lo, hi in BANDS for p in (lo, hi)])
    rows = []
    for i, (name, _, _) in enumerate(BANDS):
        mask = still & (ref >= edges[2 * i]) & (ref < edges[2 * i + 1])
        if mask.sum() < 20:
            continue
        jb, ja = jitter(np.median(b[:, mask], axis=1)), jitter(np.median(a[:, mask], axis=1))
        rows.append((name, *_removed(jb, ja), jb))
    worse = [r for r in rows if r[2] > max(0.004, 0.3 * flicker)]
    short = [r for r in rows if r[1] < 0.6 * flicker and not r in worse]   # noqa: E713
    words = ", ".join(f"{n} {rem:.3f}" for n, rem, _, _ in rows)
    if worse:
        names = " and ".join(r[0] for r in worse)
        return Finding("look", f"Tones: the {names} picked up flicker",
                       f"Flicker removed per level (stops): {words}. One correction for every brightness level "
                       f"over-corrected the {names}, most likely because the grade compressed them.",
                       fix=f"In your editor, ease off whatever compresses the {names} (a strong curve shoulder, or a "
                           "big Highlights or Shadows move), export again and finish again. A tone-aware correction "
                           "in harbium-timelapse would also fix it; it isn't built yet.")
    if short:
        names = " and ".join(r[0] for r in short)
        return Finding("note", f"Tones: the {names} had less flicker removed",
                       f"Flicker removed per level (stops): {words}, against {flicker:.3f} for the frame. "
                       "The grade may have compressed them a little; nothing was made worse.",
                       fix="Nothing to fix.")
    return Finding("ok", "Tones: every brightness level had the same flicker removed",
                   f"Flicker removed per level (stops): {words}, matching the frame's {flicker:.3f}. "
                   "One correction suits every level in this sequence.")


def flicker_share(before: np.ndarray, still: np.ndarray) -> np.ndarray:
    """For every pixel, how much of the frame's flicker it shares: 1 = all of it, 0 = none (lit by a steady lamp,
    say). Each pixel's frame-to-frame changes, compared with the frame's own."""
    g = np.diff(np.median(before[:, still], axis=1), 2)
    p = np.diff(before, 2, axis=0)
    return (g @ p) / max(float(g @ g), 1e-12)


def check_regions(rec: Recorder) -> Finding:
    b = _logs(rec.before)
    h, w = rec.before[0].shape
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    flicker = _flicker(b, still)
    if flicker < 0.004:
        return Finding("ok", "Regions: almost no flicker to judge", "")
    share = flicker_share(b, still)
    rows, cols = GRID
    yy, xx = np.divmod(np.arange(h * w), w)
    boxes, worst = [], 0.0
    for r in range(rows):
        for c in range(cols):
            cell = still & (yy * rows // h == r) & (xx * cols // w == c)
            if cell.sum() < 15:
                continue
            k = float(np.mean(share[cell]))
            if k < 0.6:                                   # most of the cell kept out of the flicker
                boxes.append((r / rows, c / cols, (r + 1) / rows, (c + 1) / cols))
                worst = max(worst, (1 - k) * flicker)
    if not boxes:
        return Finding("ok", "Regions: every area shared the flicker",
                       "The correction steadied every part of the frame it could measure.")
    return Finding("look", f"Regions: {len(boxes)} area{'s' if len(boxes) > 1 else ''} picked up small steps",
                   "These areas didn't share the frame's flicker (often because steady lamps light them while the "
                   f"daylight wobbled), so correcting the whole frame added steps of up to about {worst:.3f} stops "
                   "there.", boxes=boxes,
                   fix="In your editor, mask the marked areas and give them the opposite of the correction (or simply "
                       "exclude them from your deflicker if your editor allows it), then export and finish again. If the "
                       "steps are hard to see in the film, it's fine to leave them.")


def check_sections(measured: np.ndarray, target: np.ndarray, after: np.ndarray) -> Finding:
    n = len(measured)
    if n < 30:
        return Finding("ok", "Sections: too short to compare", "")
    thirds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    change = [abs(float(np.mean(np.diff(target[a:b])))) for a, b in thirds]
    left = [jitter(after[a:b]) for a, b in thirds]
    words = (f"Real change per frame: {change[0]:.3f}, {change[1]:.3f} and {change[2]:.3f} stops at the start, "
             f"middle and end; flicker left: {left[0]:.3f}, {left[1]:.3f}, {left[2]:.3f}.")
    fast = max(change) / max(min(change), 1e-4)
    if fast >= 3:
        where = ("start", "middle", "end")[int(np.argmax(change))]
        return Finding("note", f"Sections: the light changes {fast:.0f}x faster at the {where}",
                       words + f" Steps between frames at the {where} of the time slice are mostly real change, "
                       "not flicker: what matters is that they stay even.",
                       fix="Nothing to fix. To make the change look gentler, ease the exposure across those frames in "
                           "your edit (that's a creative choice, not a correction).")
    return Finding("ok", "Sections: the light changes at a similar pace throughout", words)


def check_blends(measured: np.ndarray, window: int = 8) -> Finding:
    """Where the rate of change switches abruptly: the slope over the frames before against the frames after."""
    v = np.asarray(measured, dtype=np.float64)
    n = len(v)
    if n < 4 * window:
        return Finding("ok", "Blends: too short to check", "")
    x = np.arange(window, dtype=np.float64)
    slopes = np.array([np.polyfit(x, v[i:i + window], 1)[0] for i in range(n - window + 1)])
    kinks = np.array([slopes[i] - slopes[i - window] for i in range(window, len(slopes))])   # at frame i
    noise = float(np.median(np.abs(kinks - np.median(kinks)))) * 1.4826 + 1e-6
    found = []
    for j in np.argsort(-np.abs(kinks)):
        if abs(kinks[j]) < max(0.012, 5 * noise) or len(found) == 3:
            break
        frame = int(j + window)
        if all(abs(frame - f) > window for f in found):
            found.append(frame)
    if not found:
        return Finding("ok", "Blends: the light's pace changes smoothly",
                       "No sudden switches in how fast the light changes, as an editing keyframe or blend can cause.")
    names = ", ".join(str(f + 1) for f in sorted(found))
    return Finding("note", f"Blends: the pace of change switches abruptly near frame {names}",
                   "The light's rate of change jumps there, which is the usual sign of a keyframe or a blend "
                   "between edits.", frames=sorted(found),
                   fix=f"Open frame {names} in your editor. If a keyframe sits there, move it or soften the change "
                       "between the keyframes either side (with RapidRAW, timelapse-rapidraw refine blends edits "
                       "smoothly between keyframes). Or raise the smoothness in Finish to round the switch off.")


def diagnose(rec: Recorder, measured: np.ndarray, target: np.ndarray, after: np.ndarray) -> list[Finding]:
    """Every check, in order. ``measured``, ``target`` and ``after`` are per-frame brightness in stops."""
    out = [check_sections(measured, target, after), check_blends(measured)]
    if len(rec.before) >= 10:
        out += [check_tones(rec), check_regions(rec)]
    return out
