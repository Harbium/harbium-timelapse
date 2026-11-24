"""Every screen builds at a wide and a narrow window, against the real model: two columns when there's room,
one when there isn't (a narrow window had wrapped words letter by letter)."""

import types

import numpy as np
import pytest

pytest.importorskip("flet")       # the interface: these tests need it, the bare job skips them

from test_app_model import folder, opened, services, until, bare_raws  # noqa: F401  (fixtures)
from harbium_timelapse.gui.models.timeline import Theme
from harbium_timelapse.gui.theme import LIGHT
from harbium_timelapse.gui.views import app as views


def stub_view(m, width, height=1000):
    ui = views.UI(LIGHT)
    page = types.SimpleNamespace(width=width, height=height, run_task=lambda f, *a: None, update=lambda: None,
                                 overlay=[])
    view = types.SimpleNamespace(page=page, model=m, ui=ui, chart_theme=Theme(), screen=None,
                                 content_width=max(360, width - 64), content_height=height,
                                 ask=None, MARGIN=64)
    ui.scale = min(1.3, max(1.12, 1.12 * width / 1600))
    return view


@pytest.mark.parametrize("width", [1600, 700])
def test_every_screen_builds_at_this_width(bare_raws, monkeypatch, width):  # noqa: F811
    from harbium_timelapse import previews
    monkeypatch.setattr(previews, "load_small", lambda path, longest: np.full((40, 60, 3), 0.1))
    m = opened(bare_raws)
    built = {}
    for name in ("start", "folder"):
        built[name] = views.SCREENS[name](stub_view(m, width)).build()
    m.choose_editor("rapidraw")
    m.wait()
    m.go("prepare")
    m.wait()
    until(lambda: m.preview is not None and not m.previewing)
    screen = views.SCREENS["prepare"](stub_view(m, width))
    built["prepare"] = screen.build()
    two = isinstance(built["prepare"], views.ft.Row)
    assert two == (width == 1600)                                  # two columns only with room for them
    m.go("grade")
    screen = views.SCREENS["grade"](stub_view(m, width))
    built["grade"] = screen.build()
    assert isinstance(built["grade"], views.ft.Row) == (width == 1600)
    assert screen.seq.WIDTH <= max(360, width - 64)                # the viewer fits the window
    full = views.FullView(stub_view(m, width), True, screen.seq.src)
    full.open()
    assert full.seq.full and full.seq.WIDTH <= width
    full.close()


def test_refreshing_reuses_pictures_instead_of_reloading_them(bare_raws, monkeypatch):  # noqa: F811
    """Regression: every refresh (many a second during playback or a job) made new picture controls for the
    keyframe list, and each new one flashed white while it loaded."""
    from harbium_timelapse import previews
    monkeypatch.setattr(previews, "load_small", lambda path, longest: np.full((40, 60, 3), 0.1))
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    m.go("prepare")
    m.wait()
    m.go("grade")
    m.wait()
    m.add_keyframe(0)
    screen = views.SCREENS["grade"](stub_view(m, 1600))
    screen.build()
    panel = screen.keys
    before = list(panel.body.controls)
    for _ in range(5):
        screen.refresh()
    assert panel.body.controls == before and all(a is b for a, b in zip(panel.body.controls, before))
    calls = []
    first = screen.picture("k", lambda: calls.append(1) or object())
    assert screen.picture("k", lambda: calls.append(1) or object()) is first and calls == [1]
