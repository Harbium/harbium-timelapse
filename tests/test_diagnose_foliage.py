"""A frame mostly full of trees: no false alarm, but a real lamp-lit area is still found.

Regression: on James's dusk sequence (trees fill much of the frame, and their overall brightness shifts with
the wind) the checks estimated the frame's flicker at 0.121 stops against a measured 0.041, and outlined 33
of 48 areas. They now judge everything against finish's own measurement, and say "can't tell" where too much
else moves."""

import numpy as np

from harbium_timelapse.diagnose import Recorder, check_regions, check_tones

rng = np.random.default_rng(5)
H, W, N = 40, 60, 80
scene = np.concatenate([np.full((14, W), 0.02), np.full((13, W), 0.12), np.full((13, W), 0.5)])
scene = scene * np.exp(rng.normal(0, 0.15, (H, W)))
flicker = rng.normal(0, 0.04, N)
wind = rng.normal(0, 0.15, N)


def trees(lamp: bool) -> Recorder:
    rec = Recorder()
    for i, f in enumerate(flicker):
        frame = scene * 2 ** f
        leaves = np.kron(rng.normal(0, 0.4, (H // 4 + 1, 9)), np.ones((4, 4)))[:H, :36]
        frame[:, :36] = scene[:, :36] * 2 ** f * np.exp(leaves + wind[i])          # trees: 60% of the frame
        if lamp:
            frame[0:14, 40:56] = scene[0:14, 40:56]                                # lit by a steady lamp
        before = np.clip(frame, 0, 1).astype(np.float32)
        rec.before.append(before)
        rec.after.append(np.clip(before * 2 ** (-f), 0, 1).astype(np.float32))
    return rec


def test_a_frame_full_of_moving_trees_raises_no_false_alarm():
    rec = trees(lamp=False)
    regions, tones = check_regions(rec, flicker), check_tones(rec, flicker)
    assert regions.kind == "ok" and "too much movement to judge" in regions.detail
    assert tones.kind == "ok"


def test_a_lamp_lit_area_among_the_trees_is_still_found():
    f = check_regions(trees(lamp=True), flicker)
    assert f.kind == "look" and f.boxes
    assert all(left >= 0.6 and top < 0.4 for top, left, _, _ in f.boxes)        # only the lamp area, top right
