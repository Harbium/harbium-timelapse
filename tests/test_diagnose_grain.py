"""High grain: the checks give the same answers as on clean footage, and noisy shadows aren't thrown away."""

import numpy as np

from harbium_timelapse.diagnose import Recorder, _logs, _still, check_regions, check_tones
from harbium_timelapse.finish import _linear_to_srgb

rng = np.random.default_rng(21)
H, W, N = 720, 1080, 60                    # samples every 6 pixels, so each point averages a full 6x6 patch
rows = np.repeat([0.02, 0.12, 0.5], H // 3)[:, None] * np.ones((1, W))
scene = rows * np.exp(rng.normal(0, 0.1, (H, W)))
flicker = rng.normal(0, 0.05, N)


def encode(linear):
    return np.round(_linear_to_srgb(np.clip(linear, 0, 1)) * 65535).astype(np.uint16)[..., None].repeat(3, axis=2)


def grainy(lamp: bool) -> Recorder:
    rec = Recorder()
    for i, f in enumerate(flicker):
        light = scene * 2 ** f
        leaves = np.kron(rng.normal(0, 0.6, (H // 24 + 1, W // 192 + 1)), np.ones((24, 24)))[:H, : W // 8]
        light[:, : W // 8] = scene[:, : W // 8] * np.exp(leaves)       # swaying foliage: whole leaves move, in blobs
        if lamp:
            light[H // 20: H // 4, int(W * 0.7): int(W * 0.9)] = scene[H // 20: H // 4, int(W * 0.7): int(W * 0.9)]
        iso = 1 + 2 * i / (N - 1)                                                             # ISO rising 3x
        grain = rng.normal(0, 1, (H, W)) * np.sqrt(0.0004 * light + 1e-6) * iso              # shot + read noise
        before = np.clip(light + grain, 0, 1)
        rec.add(encode(before), encode(before * 2 ** (-f)), i, N)                             # finish's correction
    return rec


def test_high_grain_gives_the_same_answers_as_clean_footage():
    rec = grainy(lamp=False)
    assert check_tones(rec).kind == "ok", check_tones(rec).detail
    assert check_regions(rec).kind == "ok"


def test_the_lamp_lit_area_is_still_found_through_heavy_grain():
    f = check_regions(grainy(lamp=True))
    assert f.kind == "look" and f.boxes
    assert all(left >= 0.5 and top < 0.4 for top, left, _, _ in f.boxes)        # only where the lamp area is


def test_noisy_shadows_are_kept_not_mistaken_for_movement():
    rec = grainy(lamp=False)
    b = _logs(rec.before)
    ref = b[len(b) // 2]
    usable = (ref > np.log2(1 / 1000)) & (ref < np.log2(0.97))
    still = _still(b, usable)
    h, w = rec.before[0].shape
    shadows, foliage = np.zeros((h, w), bool), np.zeros((h, w), bool)
    shadows[: h // 3, w // 4:] = True                         # the dark third, away from the foliage
    foliage[:, : w // 10] = True
    assert still.reshape(h, w)[shadows].mean() > 0.8          # grainy shadows stay in
    assert still.reshape(h, w)[foliage].mean() < 0.2          # swaying leaves are still set aside
