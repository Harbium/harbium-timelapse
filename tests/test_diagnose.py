"""Each check finds its own problem, says what to do, and stays quiet when there's nothing to find."""

import numpy as np

from harbium_timelapse.diagnose import Recorder, check_blends, check_regions, check_sections, check_tones

rng = np.random.default_rng(11)
H, W, N = 40, 60, 80
scene = np.concatenate([np.full((14, W), 0.02), np.full((13, W), 0.12), np.full((13, W), 0.5)])
scene = scene * np.exp(rng.normal(0, 0.15, (H, W)))
flicker = rng.normal(0, 0.05, N)


def recorder(tone=lambda x: x, lamp=False, foliage=True):
    rec = Recorder()
    for i, f in enumerate(flicker):
        frame = scene * 2 ** f
        if foliage:                                    # leaves swaying: wild pixel-by-pixel change, not flicker
            frame[:, :8] = scene[:, :8] * np.exp(rng.normal(0, 0.6, (H, 8)))
        if lamp:                                       # lit by a steady lamp: no flicker at all
            frame[4:14, 40:56] = scene[4:14, 40:56]
        before = np.clip(tone(frame), 0, 1)
        rec.before.append(before.astype(np.float32))
        rec.after.append(np.clip(before * 2 ** (-f), 0, 1).astype(np.float32))    # finish's whole-frame correction
    return rec


def test_clean_footage_gets_a_clean_bill():
    rec = recorder()
    assert check_tones(rec).kind == "ok" and check_regions(rec).kind == "ok"


def test_a_steady_lamp_lit_area_is_found_and_marked():
    f = check_regions(recorder(lamp=True))
    assert f.kind == "look" and f.boxes and "mask" in f.fix
    top, left, bottom, right = f.boxes[0]
    assert left >= 0.5 and top < 0.4                     # in the top right, where the lamp area is
    assert all(b[1] >= 0.5 for b in f.boxes)             # and nowhere else: not the swaying foliage on the left


def test_a_tone_curve_compressing_highlights_is_found():
    curve = lambda x: (1 - np.exp(-3 * x)) / (1 - np.exp(-3))   # noqa: E731 - a strong shoulder
    rec = Recorder()
    for f in flicker:                                     # the export carries the flicker through the curve...
        before = curve(scene * 2 ** f)
        rec.before.append(before.astype(np.float32))
        lum_shift = np.log2(np.median(before)) - np.log2(np.median(curve(scene)))
        rec.after.append(np.clip(before * 2 ** (-lum_shift), 0, 1).astype(np.float32))   # ...finish corrects it whole
    f = check_tones(rec)
    assert f.kind in ("look", "note") and "highlights" in f.title and f.fix


def test_an_abrupt_change_of_pace_is_found_near_its_frame():
    t = np.arange(120, dtype=float)
    smooth = -0.01 * t
    kinked = np.where(t < 70, -0.01 * t, -0.7 - 0.05 * (t - 70)) + rng.normal(0, 0.002, 120)
    assert check_blends(smooth + rng.normal(0, 0.002, 120)).kind == "ok"
    f = check_blends(kinked)
    assert f.kind == "note" and any(abs(fr - 70) <= 4 for fr in f.frames) and "keyframe" in f.fix


def test_steep_real_change_is_explained_not_blamed():
    t = np.arange(90, dtype=float)
    target = -0.005 * t - 0.0006 * np.maximum(t - 60, 0) ** 2
    f = check_sections(target, target, target)
    assert f.kind == "note" and "end" in f.title and "light itself" in f.detail and not f.fix


def test_levels_carrying_more_of_the_flicker_are_flagged_as_under_corrected():
    """Regression: highlights carried 206% of the frame's flicker (a contrasty grade) and were called fine."""
    steep = lambda x: np.clip(x ** 2.2 * 4, 0, 1)                 # noqa: E731 - a grade that stretches brighter levels
    rec = Recorder()
    for f in flicker:
        before = steep(scene * 2 ** f).astype(np.float32)
        rec.before.append(before)
        rec.after.append(before)
    finding = check_tones(rec, flicker)
    assert finding.kind == "look" and "keep some flicker" in finding.title and "contrast" in finding.fix


def test_dither_in_deep_shadows_is_not_read_as_flicker_after_correction():
    """Regression: after a tone-aware finish, the check reported shadows 0.132 -> 0.211 stops while the film's
    shadows wobbled only 0.014-0.026: dither (+-1 level) at levels of 1-3 out of 255 read as flicker."""
    from harbium_timelapse.diagnose import check_tones_after
    from harbium_timelapse.tone import ToneShares
    rng2 = np.random.default_rng(4)
    dark = np.exp(rng2.uniform(np.log(0.0006), np.log(0.5), scene.shape))   # down to 1-3 levels of 255 in sRGB
    rec = Recorder()
    for f in flicker:
        rec.before.append((dark * 2 ** f).astype(np.float32))
        srgb = np.clip(dark ** (1 / 2.2) * 255 + rng2.triangular(-1, 0, 1, dark.shape), 0, 255).round() / 255
        rec.after.append((srgb ** 2.2).astype(np.float32))              # steady, then dithered and rounded
    finding = check_tones_after(rec, ToneShares((-8.0, -1.0), (1.2, 1.0), (0, 4)))
    pairs = [part.split(" ")[-3:] for part in finding.detail.split(": ", 2)[-1].rstrip(".").split(", ")]
    assert all(float(after) < float(before) for before, _, after in pairs), finding.detail
