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


def structured_scene():
    """A scene like a street at dusk: bright sky, mid-tone houses, a dark street, textured in patches."""
    g = np.random.default_rng(3)
    yy = np.linspace(0, 1, H)[:, None]
    base = np.where(yy < 0.33, 0.45, np.where(yy < 0.6, 0.06, 0.006)) * np.ones((1, W))
    return base * np.exp(g.normal(0, 0.3, (H // 8, W // 8))).repeat(8, 0).repeat(8, 1)


@pytest.mark.parametrize("flare", [0.005, 0.02, 0.06, 0.15])
def test_veiling_flare_across_the_whole_frame_is_left_alone(flare):
    """Regression (James's sunset): headlights into the lens added light everywhere, the dark street +3-4
    stops, the sky barely; measured as over-exposure, the whole frame was darkened and the sky came out 2-3
    stops too dark. Added light moves dark regions far more than bright ones; exposure moves them all alike.
    Flare here brightens the dark street by 0.6 to 4.1 stops."""
    street = structured_scene()
    chain = SteadyChain()
    for i, f in enumerate(flicker):
        img = street * 2 ** f
        if i in (20, 21):
            img = img + flare * np.linspace(0.3, 1.0, W)[None, :]      # stronger towards the light
        chain.add(steady_input(np.clip(img * np.exp(rng.normal(0, 0.01, img.shape)), 0, 0.97)))
    m = chain.series(np.zeros(N))
    err = (m - m[0]) - (flicker - flicker[0])
    assert np.abs(err[20:22]).max() < 0.1                       # during: within a tenth of a stop
    assert np.abs(err[22:]).max() < 0.02                        # after: nothing carried forward


def test_uneven_flare_reaching_every_region_like_james_headlights_is_bridged():
    """Regression with the shape of James's real frames (diagnose_votes on 181A0960-0962): headlights brightened
    every region, the sky included, by 0.4 to 5.5 stops, regions of the same brightness differing by stops.
    Chained frame to frame, the frames after it were left 0.94 stops off and the event frames darkened 1.5."""
    street = structured_scene()
    g = np.random.default_rng(11)
    patches = np.exp(g.normal(0, 1.0, (H // 8, W // 8))).repeat(8, 0).repeat(8, 1)        # very uneven
    flare = 0.05 * patches * np.linspace(0.4, 1.6, W)[None, :]
    chain = SteadyChain()
    for i, f in enumerate(flicker):
        img = street * 2 ** f
        if i in (20, 21):
            img = img + flare * (1.0 if i == 20 else 0.7)
        chain.add(steady_input(np.clip(img * np.exp(rng.normal(0, 0.01, img.shape)), 0, 0.97)))
    m = chain.series(np.zeros(N))
    err = (m - m[0]) - (flicker - flicker[0])
    assert np.abs(err[22:]).max() < 0.02                       # nothing carried into the frames after it
    assert np.abs(err[20:22]).max() < 0.1                       # the event frames left as shot


def test_people_close_to_the_camera_are_not_a_light_event_and_their_frames_flicker_is_corrected():
    """Regression (James's dusk sequence, frames 64-66): people close to the camera pushed a few regions far out;
    the 10th-90th spread crossed 1 stop, the frames were treated as light events and left as shot, and frame
    64's flicker (an even 0.11-stop dip) stayed in the film as a flash."""
    street = structured_scene()
    g = np.random.default_rng(21)
    chain = SteadyChain()
    rows, cols = 6, 8
    for i, f in enumerate(flicker):
        img = street * 2 ** f
        if 15 <= i <= 25:                                       # figures filling a few regions near the camera
            for r, c in zip(g.integers(3, rows, 6), g.integers(0, cols, 6), strict=True):
                ys, xs = slice(r * H // rows, (r + 1) * H // rows), slice(c * W // cols, (c + 1) * W // cols)
                img[ys, xs] = img[ys, xs] * g.choice([0.15, 6.0])
        chain.add(steady_input(np.clip(img * np.exp(rng.normal(0, 0.01, img.shape)), 0, 0.97)))
    assert not any(e is not None and e[2] for e in chain.steps)          # never treated as a light event
    m = chain.series(np.zeros(N))
    err = (m - m[0]) - (flicker - flicker[0])
    assert np.abs(err).max() < 0.03                                     # the flicker measured, so corrected
