"""The sequence viewer: frames above the timeline, keyframes on it, keys and clicks."""

import types
from pathlib import Path

import numpy as np
import pytest

from harbium_timelapse.gui.models.timeline import PAD, Theme, render_timeline
from harbium_timelapse.gui.models.viewer import SequenceFrames, reduce
from harbium_timelapse.timeslice import to_display

rng = np.random.default_rng(6)
LINEAR = [np.clip(np.exp(rng.uniform(np.log(0.003), np.log(0.6), (40, 60, 3))), 0, 1) for _ in range(5)]


def test_after_is_the_correction_applied_exactly():
    frames = SequenceFrames([reduce(x, 1.0) for x in LINEAR], gains=[0.0, 0.3, -0.25, 0.0, 1.0])
    for i, g in enumerate(frames.gains):
        exact = to_display(LINEAR[i] * 2.0 ** g, 1.0).astype(int)
        assert np.abs(frames.image(i, "after").astype(int) - exact).max() <= 1      # within one level of 255
    assert np.array_equal(frames.image(1, "before"), reduce(LINEAR[1], 1.0))


def test_the_wipe_shows_before_left_and_after_right():
    frames = SequenceFrames([reduce(x, 1.0) for x in LINEAR], gains=[0.5] * 5)
    wiped, before, after = frames.image(2, "wipe", 0.5), frames.image(2, "before"), frames.image(2, "after")
    assert np.array_equal(wiped[:, :28], before[:, :28]) and np.array_equal(wiped[:, 32:], after[:, 32:])


def test_suggested_keyframes_are_outlines_and_labelled_ones_solid():
    line = np.linspace(0, -2, 50)
    solid = render_timeline([type("S", (), {"values": line, "color": (200, 200, 200), "width": 1})()], 200, 80,
                            Theme(), flags=(25,))
    ghost = render_timeline([type("S", (), {"values": line, "color": (200, 200, 200), "width": 1})()], 200, 80,
                            Theme(), ghosts=(25,))
    x = int(round(PAD + 25 * (200 - 2 * PAD) / 49))
    centre = (11, x)                                            # the diamond's middle: filled vs hollow
    assert tuple(solid[centre]) == Theme().mark and tuple(ghost[centre]) != Theme().mark


@pytest.fixture
def panel():
    pytest.importorskip("flet")                  # the interface: the bare job (no Flet) skips these
    from harbium_timelapse.gui.theme import LIGHT
    from harbium_timelapse.gui.views.app import UI, SequenceView
    seq = [types.SimpleNamespace(name=f"181A{834 + i:04d}.CR3", path=Path(f"/x/{i}")) for i in range(130)]
    calls = []
    m = types.SimpleNamespace(
        opened=types.SimpleNamespace(seq=seq), view=SequenceFrames([reduce(LINEAR[0], 1.0)] * 130),
        cursor=0, view_mode="after", wipe=0.5, view_status="", keyframes=[{"frame": f, "reason": ""} for f in (0, 68, 129)],
        keyframes_state="labelled", prepare_series=(np.linspace(0, -3, 130), np.linspace(0, -3, 130)),
        measured=None, exposure_steps=(), load_view=lambda: None)
    m.set_cursor = lambda f: setattr(m, "cursor", int(min(max(f, 0), 129)))
    m.view_image = lambda frame=None: m.view.image(m.cursor if frame is None else frame, m.view_mode, m.wipe)
    m.toggle_keyframe_here = lambda: calls.append(m.cursor)
    view = SequenceView(types.SimpleNamespace(m=m, ui=UI(LIGHT), view=None))
    view.control()
    view.refresh()
    return view, m, calls


def test_keys_step_jump_and_toggle(panel):
    view, m, calls = panel
    key = lambda k, shift=False: types.SimpleNamespace(key=k, shift=shift)     # noqa: E731
    view.on_key(key("Arrow Right"))
    view.on_key(key("Arrow Right", shift=True))
    assert m.cursor == 11
    view.on_key(key("Arrow Left"))
    view.on_key(key("K"))
    assert m.cursor == 10 and calls == [10]


def test_a_click_on_the_diamonds_snaps_to_the_nearest_keyframe(panel):
    view, m, _ = panel
    at = lambda frame, y: types.SimpleNamespace(local_position=types.SimpleNamespace(            # noqa: E731
        x=PAD + frame * (view.WIDTH - 2 * PAD) / 129, y=y))
    view._tapped(at(69, 8))                                     # a frame off the diamond for frame 69 (~5 px)
    assert m.cursor == 68
    view._tapped(at(90, 60))                                    # on the curve: exactly that frame
    assert m.cursor == 90
    assert view.line_image().shape == (view.LINE_H, view.WIDTH, 3)


def test_on_prepare_there_are_no_keyframe_controls(panel):
    from harbium_timelapse.gui.views.app import SequenceView
    view, m, calls = panel
    plain = SequenceView(view.screen, keyframes=False)
    plain.control()
    plain.refresh()
    words = [getattr(c, "content", "") for c in plain.controls.controls]
    assert not any("keyframe" in str(w).lower() for w in words)
    plain.on_key(types.SimpleNamespace(key="K", shift=False))
    assert calls == [] and plain._keys() == ((), ())


def test_flicker_x5_magnifies_leftover_flicker_and_leaves_steady_frames_alone():
    steady = [reduce(LINEAR[0], 1.0)] * 7
    frames = SequenceFrames(steady)
    assert np.abs(frames.image(3, "flicker").astype(int) - steady[3].astype(int)).max() <= 1
    jumpy = list(steady)
    jumpy[3] = reduce(LINEAR[0] * 2 ** 0.1, 1.0)                      # 0.1 stops too bright
    shown = frames.__class__(jumpy).image(3, "flicker")
    from harbium_timelapse.timeslice import from_display
    mid = (LINEAR[0] > 0.01) & (LINEAR[0] < 0.08)                       # mid levels, clear of clipping and black
    stops = np.median(np.log2(from_display(shown)[mid] / from_display(steady[3])[mid]))
    assert 0.4 < stops < 0.6                                             # about five times the 0.1


def test_the_review_viewer_matches_what_finish_writes():
    from harbium_timelapse.finish import correct
    from harbium_timelapse.tone import ToneShares
    tone = ToneShares((-7.0, -4.0, -1.0), (0.8, 1.3, 2.0), (0, 2, 4))
    exported = [reduce(x, 1.0) for x in LINEAR]
    gains = np.array([[0.1, 0.05, -0.02], [-0.2, -0.25, -0.15], [0, 0, 0], [0.3, 0.3, 0.3], [-0.05, 0, 0.05]])
    frames = SequenceFrames(exported, gains=gains, tone=tone)
    for i in range(5):
        written = correct(exported[i], gains[i], tone=tone)
        assert np.abs(frames.image(i, "after").astype(int) - written.astype(int)).max() <= 1


def test_the_scrub_bar_moves_the_playhead(panel):
    view, m, _ = panel
    drag = types.SimpleNamespace(control=types.SimpleNamespace(value=57.4))
    view._scrubbing(drag)
    assert m.cursor == 57 and "frame 58 of 130" in view.label.value
    view._scrubbed(types.SimpleNamespace(control=types.SimpleNamespace(value=90)))
    assert m.cursor == 90 and view.scrub.value == 90


def test_flicker_x5_keeps_grain_calm():
    """Regression: grain in the shadows came out as vivid blue, green and purple speckles, read as flicker."""
    g = np.random.default_rng(9)
    base = np.clip(np.exp(g.uniform(np.log(0.0005), np.log(0.5), (60, 90, 1))) * np.ones((1, 1, 3)), 0, 1)
    grainy = [reduce(np.clip(base * np.exp(g.normal(0, 0.25, base.shape)), 0, 1), 1.0) for _ in range(7)]
    frames = SequenceFrames(grainy)
    shown = frames.image(3, "flicker").astype(int)
    local = np.mean([f.astype(float) for k, f in enumerate(grainy) if k != 3 and 1 <= k <= 5], axis=0)
    assert np.abs(shown - local).mean() < 3                       # close to the steady picture, no speckles
    tint_shown, tint_local = shown[..., 0] - shown[..., 2], local[..., 0] - local[..., 2]
    assert np.abs(tint_shown - tint_local).mean() < 2               # magnifying adds no colour of its own


def test_the_flicker_map_is_even_for_a_steady_film_and_marks_a_jump_where_it_happens():
    from harbium_timelapse.gui.models.viewer import flicker_map
    steady = [reduce(LINEAR[0], 1.0)] * 12
    even = flicker_map(SequenceFrames(steady)).astype(int)
    assert np.abs(even - steady[0].astype(int)).mean() < 2                 # steady: just the picture
    jumpy = list(steady)
    jumpy[6] = reduce(LINEAR[0] * 2 ** 0.15, 1.0)                          # frame 7 jumps 0.15 stops
    marked = flicker_map(SequenceFrames(jumpy)).astype(float).mean(axis=(0, 2))
    plain = steady[0].astype(float).mean(axis=(0, 2))
    edges = np.linspace(0, marked.size, 13).round().astype(int)
    ratio = [marked[edges[i]:edges[i + 1]].mean() / plain[edges[i]:edges[i + 1]].mean() for i in range(12)]
    assert ratio[6] > 1.2                                                  # a clear stripe at frame 7
    assert all(abs(ratio[i] - 1) < 0.1 for i in range(12) if i != 6)      # its neighbours barely move
    assert all(abs(ratio[i] - 1) < 0.02 for i in range(12) if abs(i - 6) > 2)   # the rest: steady
