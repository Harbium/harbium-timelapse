"""Tone-aware correction, proven with known answers.

The camera's flicker is the same at every level, but a grade that stretches bright levels (James's sunset:
contrast 54, an S-curve, AgX) stretches their flicker too: one gain per frame then leaves the sky flickering."""

import numpy as np

from harbium_timelapse.finish import Meter, _linear_to_srgb, _small_linear, correct, plan_finish

rng = np.random.default_rng(8)
H, W, N = 72, 108, 120
scene = np.kron(np.exp(rng.uniform(np.log(0.006), np.log(0.5), (H // 6, W // 6))), np.ones((6, 6)))
flicker = rng.normal(0, 0.04, N)
BRIGHT, DARK = scene > 0.25, scene < 0.03


def encode(linear):
    rgb = np.repeat(np.clip(linear, 0, 1)[..., None], 3, axis=-1)
    return np.round(_linear_to_srgb(rgb) * 65535).astype(np.uint16)


def contrasty(x):
    return x * (1 + 3 * x ** 2)          # dark levels untouched, bright ones stretched (about 2.5x at the top)


def run(grade):
    frames = [encode(grade(scene * 2 ** f)) for f in flicker]
    meter = Meter()
    for img in frames:
        meter.add(_small_linear(img))
    measured = meter.series()
    paths = [f"f{i:03d}.png" for i in range(N)]
    tone = meter.tone(measured)
    aware, flat = plan_finish(paths, measured=measured, tone=tone), plan_finish(paths, measured=measured)

    def wobble(plan, mask):
        lin = [np.log2(_small_linear(correct(img, plan.gains[i], tone=plan.tone)).mean(-1)[mask[::1, ::1]].mean())
               for i, img in enumerate(frames)]
        return float(np.std(np.diff(lin, 2)) / np.sqrt(6))
    return tone, aware, flat, wobble


def test_a_contrasty_grade_is_corrected_level_by_level():
    tone, aware, flat, wobble = run(contrasty)
    # Shares are relative to the frame's measured flicker: what matters is that bright levels need clearly more.
    assert tone is not None and tone.shares[-1] > 1.3 * tone.shares[0]
    for mask in (BRIGHT, DARK):                                  # one gain left both off; tone-aware fixes both
        assert wobble(aware, mask) < 0.35 * wobble(flat, mask)


def test_an_ungraded_sequence_is_corrected_exactly_as_before():
    tone, *_ = run(lambda x: x)
    assert tone is None
