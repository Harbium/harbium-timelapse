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
    assert f.kind == "note" and "end" in f.title and f.fix.startswith("Nothing to fix")
