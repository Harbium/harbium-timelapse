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


def share(series: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """How much of the measured flicker a series carries (1 = all of it, 0 = none), and how sure that is (the
    standard error). Frame-to-frame changes of the series against the reference's: anything else moving there
    (leaves in the wind, walkers) makes the estimate less sure, never a false answer."""
    g = np.diff(np.asarray(reference, dtype=np.float64), 2)
    p = np.diff(np.asarray(series, dtype=np.float64), 2)
    ok = np.isfinite(g) & np.isfinite(p)
    if ok.sum() >= 8:
        # Leave out outsized jumps: a light event (headlights into the lens, a flash) is many times the size of
        # ordinary flicker and would otherwise decide the whole estimate on its own.
        spread = 1.4826 * np.median(np.abs(g[ok] - np.median(g[ok])))
        if spread > 0:
            ok &= np.abs(g - np.median(g[ok])) <= 5 * spread
    g, p = g[ok], p[ok]
    if g.size < 8 or float(g @ g) <= 0:
        return 1.0, 1.0
    k = float(g @ p) / float(g @ g)
    resid = p - k * g
    se = float(np.std(resid) / (np.std(g) * np.sqrt(g.size)))
    return k, se


def _reference(rec: "Recorder", reference: np.ndarray | None, before: np.ndarray, still: np.ndarray) -> np.ndarray:
    """The frame's measured brightness, in stops: finish's own measurement (the regional vote, built to resist
    moving content) when given; otherwise the middle brightness of the still pixels."""
    if reference is not None and len(reference) == len(before):
        ref = np.asarray(reference, dtype=np.float64)
        if np.isfinite(ref).sum() >= 8:
            idx = np.arange(len(ref))
            good = np.isfinite(ref)
            return np.interp(idx, idx[good], ref[good])
    return np.median(before[:, still], axis=1)


def check_tones(rec: Recorder, reference: np.ndarray | None = None) -> Finding:
    b = _logs(rec.before)
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    frame = _reference(rec, reference, b, still)
    flicker = jitter(frame)
    if flicker < 0.004:
        return Finding("ok", "Tones: no flicker to speak of", f"{flicker:.3f} stops before correction.")
    edges = np.percentile(ref[still], [p for _, lo, hi in BANDS for p in (lo, hi)])
    rows, unsure = [], []
    for i, (name, _, _) in enumerate(BANDS):
        mask = still & (ref >= edges[2 * i]) & (ref < edges[2 * i + 1])
        if mask.sum() < 20:
            continue
        k, se = share(np.median(b[:, mask], axis=1), frame)
        (unsure if se > 0.25 else rows).append((name, k, se))
    words = ", ".join(f"{n} {k * 100:.0f}%" for n, k, _ in rows)
    low = [(n, k, se) for n, k, se in rows if k + 2 * se < 0.7]
    high = [(n, k, se) for n, k, se in rows if k - 2 * se > 1.4]
    caveat = (f" Too much movement in the {' and '.join(n for n, _, _ in unsure)} to judge." if unsure else "")
    if low:
        names = " and ".join(n for n, _, _ in low)
        worst = max((1 - k) * flicker for _, k, _ in low)
        return Finding("look", f"Tones: the {names} are over-corrected",
                       f"They flicker less than the rest of the frame, so the correction adds up to {worst:.3f} stops "
                       "there. Usually a grade that compresses them." + caveat,
                       fix=f"Ease off whatever compresses the {names} (a strong curve shoulder, Highlights or "
                           "Shadows), then export and finish again.")
    if high:
        names = " and ".join(n for n, _, _ in high)
        worst = max((k - 1) * flicker for _, k, _ in high)
        return Finding("look", f"Tones: the {names} keep some flicker",
                       f"Your grade stretches the {names}, so they flicker more than the rest of the frame. Up to "
                       f"{worst:.3f} stops is left there." + caveat,
                       fix=f"Ease the contrast or the curve over the {names}, then export and finish again.")
    if not rows:
        return Finding("ok", "Tones: too much movement to judge", "")
    return Finding("ok", "Tones: every level steadies alike", f"Share of the flicker by level: {words}." + caveat)


def check_tones_after(rec: Recorder, tone) -> Finding:
    """After a tone-aware correction: the flicker left at each level, judged on the corrected frames."""
    b, a = _logs(rec.before), _logs(rec.after)
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    edges = np.percentile(ref[still], [p for _, lo, hi in BANDS for p in (lo, hi)])
    words = []
    for i, (name, _, _) in enumerate(BANDS):
        mask = still & (ref >= edges[2 * i]) & (ref < edges[2 * i + 1])
        if mask.sum() >= 20:
            # Average the light, then take stops: dither (tiny noise added before rounding, zero on average) is a
            # big relative change at levels of 1-3 out of 255, and a middle value of per-pixel stops read it as
            # flicker (James's sunset: shadows "0.132 -> 0.211" while the film's shadows wobbled 0.014-0.026).
            level = lambda x: np.log2(np.mean(2.0 ** x[:, mask], axis=1))     # noqa: E731
            if float(np.mean(2.0 ** a[:, mask])) < 0.002:     # about 0-8 of 255: too little light to measure
                words.append(f"{name} too dark to judge")
                continue
            words.append(f"{name} {jitter(level(b)):.3f} → {jitter(level(a)):.3f}")
    return Finding("ok", "Tones: corrected level by level",
                   tone.describe() + (" Flicker by level, before → after (stops): " + ", ".join(words) + "."
                                      if words else ""))


def check_regions(rec: Recorder, reference: np.ndarray | None = None) -> Finding:
    b = _logs(rec.before)
    h, w = rec.before[0].shape
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    frame = _reference(rec, reference, b, still)
    flicker = jitter(frame)
    if flicker < 0.004:
        return Finding("ok", "Regions: no flicker to speak of", "")
    rows, cols = GRID
    yy, xx = np.divmod(np.arange(h * w), w)
    boxes, worst, unsure = [], 0.0, 0
    for r in range(rows):
        for c in range(cols):
            cell = usable & (yy * rows // h == r) & (xx * cols // w == c)
            calm = cell & still
            pick = calm if calm.sum() >= 15 else cell
            if pick.sum() < 15:
                continue
            k, se = share(np.median(b[:, pick], axis=1), frame)
            if se > 0.25:
                unsure += 1                              # too much else moving here to tell
            elif k + 2 * se < 0.6:                       # confidently carries much less of the flicker
                boxes.append((r / rows, c / cols, (r + 1) / rows, (c + 1) / cols))
                worst = max(worst, (1 - k) * flicker)
    note = f"{unsure} area{'s' if unsure != 1 else ''} had too much movement to judge." if unsure else ""
    if not boxes:
        return Finding("ok", "Regions: every area steadies alike", note)
    return Finding("look", f"Regions: {len(boxes)} area{'s' if len(boxes) > 1 else ''} may show small steps",
                   "These areas flicker less than the frame (often lit by steady lamps), so the correction adds up to "
                   f"{worst:.3f} stops there." + (" " + note if note else ""), boxes=boxes,
                   fix="If you can see it in the film, mask these areas in your editor and reduce the correction there. "
                       "Otherwise, leave it.")


def check_sections(measured: np.ndarray, target: np.ndarray, after: np.ndarray) -> Finding:
    n = len(measured)
    if n < 30:
        return Finding("ok", "Sections: too short to compare", "")
    thirds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    change = [abs(float(np.mean(np.diff(target[a:b])))) for a, b in thirds]
    left = [jitter(after[a:b]) for a, b in thirds]
    words = (f"Change per frame at the start, middle and end: {change[0]:.3f}, {change[1]:.3f}, {change[2]:.3f} stops. "
             f"Flicker left: {left[0]:.3f}, {left[1]:.3f}, {left[2]:.3f}.")
    fast = max(change) / max(min(change), 1e-4)
    if fast >= 3:
        where = ("start", "middle", "end")[int(np.argmax(change))]
        return Finding("note", f"Sections: the light changes {fast:.0f}x faster at the {where}",
                       words + " That's the light itself, not flicker.")
    return Finding("ok", "Sections: the light changes at an even pace", words)


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
        return Finding("ok", "Blends: no sudden changes of pace", "")
    names = ", ".join(str(f + 1) for f in sorted(found))
    return Finding("note", f"Blends: a sudden change of pace near frame {names}",
                   "Usually a keyframe or an edit that changes abruptly.", frames=sorted(found),
                   fix=f"Check frame {names} in your editor: move or soften the keyframe there, or raise Smoothness.")


def diagnose(rec: Recorder, measured: np.ndarray, target: np.ndarray, after: np.ndarray,
             tone=None) -> list[Finding]:
    """Every check, in order. ``measured``, ``target`` and ``after`` are per-frame brightness in stops; the
    tone and region checks judge everything against ``measured`` (finish's regional vote)."""
    out = [check_sections(measured, target, after), check_blends(measured)]
    if len(rec.before) >= 10:
        out += [check_tones(rec, measured) if tone is None else check_tones_after(rec, tone),
                check_regions(rec, measured)]
    return out
