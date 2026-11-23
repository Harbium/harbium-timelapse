"""Light events (headlights into the lens, a flash, lights switching on) are left alone, not corrected as flicker.

Regression: on James's sunset, headlights flooding most of the frame for two frames were measured 1.9 stops too
bright, so finishing darkened the whole frame to match, sky included (it jumped 1.7-2.3 stops)."""

import numpy as np
import pytest

from harbium_timelapse.measure import SteadyChain, steady_input

rng = np.random.default_rng(3)
H, W, N = 96, 144, 40
scene = np.exp(rng.uniform(np.log(0.01), np.log(0.4), (H, W)))
flicker = rng.normal(0, 0.04, N)


def measure(change):
    chain = SteadyChain()
    for i, f in enumerate(flicker):
        img = change(i, scene * 2 ** f)
        chain.add(steady_input(np.clip(img * np.exp(rng.normal(0, 0.01, img.shape)), 0, 0.97)))
    m = chain.series(np.zeros(N))
    return (m - m[0]) - (flicker - flicker[0])


@pytest.mark.parametrize("share", [0.3, 0.55, 0.7])
def test_a_light_flooding_part_of_the_frame_is_left_alone(share):
    def headlights(i, img):
        if i in (20, 21):
            img = img.copy()
            img[:, int((1 - share) * W):] *= 2 ** 2.0
        return img
    assert np.abs(measure(headlights)).max() < 0.01          # every frame measured as the camera saw it


def test_a_real_exposure_step_that_hits_regions_unevenly_is_still_measured():
    """A vari-ND turned by a stop darkens the sky a little more than the street (polarised skylight): that's
    still the camera's exposure changing, and must be measured as a step, not dismissed as an event."""
    def nd_turned(i, img):
        if i >= 20:
            img = img * 2 ** -1.0
            img[: H // 3] *= 2 ** -0.3                          # the sky darkens a little more
        return img
    err = measure(nd_turned)                                  # the test's truth leaves the ND turn out, so...
    for later in (20, 25):
        step = err[later] - err[19]
        assert -1.35 < step < -0.95, step                     # ...it shows as the step: about a stop, never 0
