"""The app's view model, driven the way the screens drive it, without a window."""

import hashlib
import shutil
import time
import types

import numpy as np
import pytest
from PIL import Image

import harbium_timelapse.editors.lightroom.adapter as lr_adapter
from harbium_timelapse.app import session
from harbium_timelapse.gui.models import AppModel, frame_at, render_timeline, Series
from harbium_timelapse.gui.models.app import STOPPED

N = 20
RAW = np.linspace(0, -0.8, N) + np.random.default_rng(2).normal(0, 0.06, N)


def xmp(name, exposure):
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
            f'crs:Exposure2012="{exposure}" crs:RawFileName="{name}.CR2" crs:HasSettings="True"/>'
            '</rdf:RDF></x:xmpmeta>').encode()


@pytest.fixture
def folder(tmp_path, monkeypatch):
    raw = tmp_path / "timelapse photos"
    raw.mkdir()
    for i in range(N):
        (raw / f"IMG_{1000 + i}.CR2").write_bytes(b"")
        (raw / f"IMG_{1000 + i}.xmp").write_bytes(xmp(f"IMG_{1000 + i}", "0.00"))
    monkeypatch.setattr(lr_adapter, "running_programs", lambda: [])
    return raw


def services():
    """The real operations, except measuring (the stand-in photos are empty files) and the folder
    check's complaints about their missing camera settings."""
    def open_folder(folder, tracker=None):
        opened = session.open_folder(folder, tracker)
        opened.problems = [p for p in opened.problems if p.kind not in ("unknown_settings", "missing_time")]
        return opened
    return types.SimpleNamespace(**{**vars(session), "open_folder": open_folder,
                                    "measure": lambda seq, tracker=None: RAW.copy()})


def until(condition, timeout=20):
    end = time.time() + timeout
    while not condition():
        assert time.time() < end, "timed out"
        time.sleep(0.02)


def digest(folder):
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.xmp"))}


def opened(folder):
    m = AppModel(services=services())
    m.open_folder(folder)
    m.wait()
    assert m.screen == "folder", m.message
    return m


def test_opening_a_folder_shows_its_facts_and_remembers_it(folder):
    m = opened(folder)
    assert dict(m.facts)["Frames"] == str(N) and dict(m.facts)["Editor"] == "Lightroom"
    assert m.next_step == ("prepare", "Next: preview corrections")
    assert m.recent[0]["folder"].endswith("timelapse photos")


def test_the_stage_bar_only_opens_what_makes_sense(folder):
    m = AppModel(services=services())
    assert not m.can_go("prepare") and not m.can_go("finish")
    m.open_folder(folder)
    m.wait()
    states = {s: state for s, _, state in m.stage_items()}
    assert states["folder"] == "current" and states["prepare"] == "open"
    assert states["grade"] == "locked" and states["done"] == "locked"


def test_prepare_measures_once_previews_then_applies_and_undoes(folder):
    before = digest(folder)
    m = opened(folder)
    m.go("prepare")
    until(lambda: m.preview is not None and not m.previewing)
    measured, curve = m.prepare_series
    assert len(measured) == len(curve) == N and "Flicker" in m.prepare_summary
    assert digest(folder) == before                                   # previewing writes nothing
    for w in (9, 15, 31, 45):                                          # a burst of slider moves
        m.set_window(w)
    until(lambda: not m.previewing)
    assert m.window == 45 and m.preview is not None
    assert m.needs_metadata_saved and not m.can_apply                  # Lightroom: save metadata first
    m.confirm_metadata_saved(True)
    m.apply()
    m.wait()
    assert m.screen == "grade" and m.project.stage == "prepared" and "Corrections applied" in m.message
    assert digest(folder) != before
    m.undo()
    m.wait()
    assert m.screen == "folder" and m.project.stage == "opened" and digest(folder) == before


def test_cancel_and_failures_are_explained(folder):
    m = opened(folder)
    def slow(tracker):
        hook = tracker.stage("Writing")
        for i in range(100_000):
            time.sleep(0.001)
            hook(i, 100_000)
    m._run("Writing", slow, lambda r: None)
    until(lambda: m.job is not None and m.job.total > 0)
    m.cancel()
    m.wait()
    assert m.job is None and m.message == STOPPED
    m.open_folder(folder / "not-here")
    assert m.message_kind == "error" and "isn't a folder" in m.message


def sky(folder, n=N):
    folder.mkdir()
    rng = np.random.default_rng(2)
    base = np.exp(rng.normal(-1.6, 0.5, (48, 64)))[..., None] * np.array([1.0, 0.95, 0.85])
    for i in range(n):
        img = np.clip(base * 2 ** rng.normal(0, 0.08), 0, 1) ** (1 / 2.2) * 255
        Image.fromarray(img.astype(np.uint8)).save(folder / f"f{i:02d}.png")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_the_export_is_noticed_finished_and_reviewed(folder, tmp_path):
    m = opened(folder)
    m.project.reach("prepared")
    m.go("grade")
    m.export_since = time.time() - 1
    m.poll_export()
    assert m.export.state == "waiting"
    sky(folder.parent / "final-export")
    for i, f in enumerate(sorted((folder.parent / "final-export").iterdir())):   # named after the RAW frames
        f.rename(f.with_name(f"IMG_{1000 + i}{f.suffix}"))
    m.poll_export()
    assert m.export.state == "arriving" and m.export.folder.name == "final-export"
    m.use_export()
    assert m.screen == "finish" and m.video_path.suffix == ".mp4"
    m.choose_preset("edit")
    assert m.video_path.suffix == ".mov"
    m.choose_preset("share")
    m.set_video_path(tmp_path / "film.mp4")
    m.start_finish()
    m.wait()
    assert m.screen == "done" and (tmp_path / "film.mp4").exists(), m.message
    assert "Brightness wobble" in m.result_lines[0] and m.project.stage == "finished"
    for name, _ in m.worth_a_look[:1]:
        m.looks_fine(name)
        assert name not in [n for n, _ in m.worth_a_look]


def test_the_timeline_draws_and_clicks_map_to_frames():
    img = render_timeline([Series(RAW, (132, 127, 117)), Series(RAW * 0.5, (232, 228, 219), 3)], 600, 160,
                          steps=(5,), flags=(9,), current=12, cant_judge=(0, 1, N - 2, N - 1))
    assert img.shape == (160, 600, 3) and img.std() > 0
    assert frame_at(0, 600, N) == 0 and frame_at(600, 600, N) == N - 1 and frame_at(300, 600, N) in (9, 10)


def test_a_folder_of_exported_frames_goes_straight_to_finishing(tmp_path):
    sky(tmp_path / "export")
    m = AppModel()
    m.open_folder(tmp_path / "export")
    m.wait()
    assert m.next_step == ("finish", "Next: finish an export")
    m.go("finish")
    assert m.screen == "finish" and m.project.export == str(tmp_path / "export") and m.can_finish


def test_presets_never_enlarge_small_frames(tmp_path):
    sky(tmp_path / "export")                                   # 64 pixels wide
    assert AppModel._preset_width(tmp_path / "export", 3840) == 64
    assert AppModel._preset_width(tmp_path / "export", None) is None


def test_an_export_only_folder_never_shows_prepare_or_grade_as_done(tmp_path):
    sky(tmp_path / "export")
    m = AppModel()
    m.open_folder(tmp_path / "export")
    m.wait()
    m.project.reach("finished")
    states = {s: state for s, _, state in m.stage_items()}
    assert states["prepare"] == "locked" and states["grade"] == "locked"


def test_raw_files_without_edit_files_explain_what_to_do(tmp_path):
    raw = tmp_path / "sunset"
    raw.mkdir()
    for i in range(6):
        (raw / f"181A{880 + i}.CR3").write_bytes(b"")
    m = AppModel(services=services())
    m.open_folder(raw)
    m.wait()
    target, label = m.next_step
    assert m.raw_without_edits and label == "Choose your editor first" and not m.can_go(target)
    card = m.problems[0]                                   # never sent to Finish with RAW files
    assert card["serious"] and card.get("choose_editor") and "Save Metadata to Files" in card["fix"]


# --- keyframes in the app (RapidRAW) ------------------------------------------------------------------------

@pytest.fixture
def bare_raws(tmp_path, monkeypatch):
    """RAW files with no edit files at all, as RapidRAW 1.6.3+ leaves an unedited folder."""
    from harbium_timelapse.editors.rapidraw import adapter as rr_adapter
    raw = tmp_path / "sunset"
    raw.mkdir()
    for i in range(N):
        (raw / f"181A{834 + i:04d}.CR3").write_bytes(b"")
    monkeypatch.setattr(rr_adapter, "rapidraw_running", lambda: False)
    monkeypatch.setattr(session, "ensure_previews", lambda seq, tracker=None: 0)   # stand-ins have no previews
    return raw


def test_a_raw_folder_with_no_edit_files_asks_for_the_editor_then_opens_with_it(bare_raws):
    m = opened(bare_raws)
    card = m.problems[0]
    assert card.get("choose_editor") and m.next_step[1] == "Choose your editor first"
    m.choose_editor("rapidraw")
    m.wait()
    assert m.opened.editor is not None and m.opened.editor.name == "rapidraw"
    assert m.next_step == ("prepare", "Next: preview corrections") and m.can_go("prepare")
    assert not any(c["serious"] for c in m.problems)                 # missing files are a note, not a stop


def test_keyframes_suggest_adjust_label_and_blend_from_the_app(bare_raws):
    from harbium_timelapse.editors.rapidraw.sidecar import read_sidecar
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    m.suggest_keyframes()
    m.wait()
    assert m.keyframes and m.keyframes_state == "chosen"
    assert m.keyframes[0]["frame"] == 0 and m.keyframes[-1]["frame"] == N - 1     # first and last at least
    m.add_keyframe(7)
    m.remove_keyframe(N - 1)
    m.add_keyframe(N - 1)
    chosen = [k["frame"] for k in m.keyframes]
    assert 7 in chosen
    m.label_keyframes()
    m.wait()
    assert m.keyframes_state == "labelled", m.message
    seq = m.opened.seq
    purple = [i for i, f in enumerate(seq) if "color:purple" in ((read_sidecar(f.path) or {}).get("tags") or [])]
    assert purple == chosen                                           # exactly the chosen frames
    m.blend_keyframes()
    m.wait()
    assert m.keyframes_state == "blended", m.message
    assert all(read_sidecar(f.path) is not None for f in seq)          # every frame has an edit file now
    m.remove_keyframe(7)                                               # change of mind: unlabel it
    m.label_keyframes()
    m.wait()
    assert "color:purple" not in (read_sidecar(seq[7].path).get("tags") or [])


def test_keyframes_already_labelled_in_rapidraw_are_picked_up(bare_raws):
    from harbium_timelapse.editors.rapidraw.keyframes import label_keyframes
    from harbium_timelapse.sequencer import load_sequence
    label_keyframes(load_sequence(bare_raws), [0, 9, N - 1])
    m = opened(bare_raws)
    m.load_keyframes()
    assert [k["frame"] for k in m.keyframes] == [0, 9, N - 1] and m.keyframes_state == "labelled"


def test_export_with_rapidraw_goes_straight_on_to_finish(bare_raws, tmp_path, monkeypatch):

    import importlib.util
    from pathlib import Path as _Path
    spec = importlib.util.spec_from_file_location(
        "rapidraw_stand_in", _Path(__file__).parent.parent / "app" / "test_rapidraw_export.py")
    stand_in = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stand_in)
    sys_path = stand_in.make_stand_in(tmp_path / "bin")
    monkeypatch.setattr(session, "find_rapidraw", lambda: sys_path)
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    assert m.rapidraw_program == sys_path
    m.export_with_rapidraw()
    m.wait()
    assert m.screen == "finish", m.message
    assert m.project.export.endswith("sunset export")


def test_keyframes_edited_after_blending_are_caught_before_exporting(bare_raws, tmp_path, monkeypatch):
    """Regression: glow and halation removed from every keyframe still showed in the export: the frames between
    them held the previous blend, and the Grade screen had forgotten the keyframes, so nothing said so."""
    import importlib.util

    from pathlib import Path as _Path
    from harbium_timelapse.editors.rapidraw.sidecar import read_sidecar, write_sidecar
    spec = importlib.util.spec_from_file_location(
        "rapidraw_stand_in", _Path(__file__).parent.parent / "app" / "test_rapidraw_export.py")
    stand_in = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stand_in)
    program = stand_in.make_stand_in(tmp_path / "bin")
    monkeypatch.setattr(session, "find_rapidraw", lambda: program)

    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    for f in (0, N - 1):
        m.add_keyframe(f)
    m.label_keyframes()
    m.wait()
    seq = m.opened.seq
    for f in (0, N - 1):                                   # graded with glow, then blended
        data = read_sidecar(seq[f].path)
        data["adjustments"] = {"glowAmount": 40, "halationAmount": 25}
        write_sidecar(seq[f].path, data)
    m.blend_keyframes()
    m.wait()
    assert m.keyframes_state == "blended"
    for f in (0, N - 1):                                   # glow and halation taken off the keyframes...
        data = read_sidecar(seq[f].path)
        data["adjustments"].update(glowAmount=0, halationAmount=0)
        write_sidecar(seq[f].path, data)

    fresh = AppModel(services=m.s)                         # ...and back to the app later
    fresh.open_folder(bare_raws)
    fresh.wait()
    fresh.load_keyframes()
    assert [k["frame"] for k in fresh.keyframes] == [0, N - 1] and fresh.keyframes_state == "labelled"
    assert "glow amount" in fresh.keyframes_out_of_date and "halation amount" in fresh.keyframes_out_of_date
    fresh.screen = "grade"
    fresh.export_with_rapidraw()                           # first press: a warning, nothing exported
    assert fresh.screen == "grade" and "Blend again" in fresh.message
    fresh.blend_keyframes()
    fresh.wait()
    assert fresh.keyframes_state == "blended" and not fresh.keyframes_out_of_date
    assert read_sidecar(seq[5].path)["adjustments"]["glowAmount"] == 0   # the frames between follow again
    fresh.export_with_rapidraw()
    fresh.wait()
    assert fresh.screen == "finish", fresh.message


def test_the_viewer_loads_from_previews_and_follows_the_playhead(bare_raws, monkeypatch):
    from harbium_timelapse import previews
    rng = np.random.default_rng(1)
    monkeypatch.setattr(previews, "load_small", lambda path, longest: rng.uniform(0.01, 0.4, (40, 60, 3)))
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    m.load_view()
    until(lambda: m.view is not None)
    assert len(m.view) == N and m.view_image().shape == (40, 60, 3)
    m.set_cursor(N + 5)
    assert m.cursor == N - 1                                       # kept within the sequence
    m.set_cursor(4)
    m.toggle_keyframe_here()
    assert [k["frame"] for k in m.keyframes] == [4]
    m.toggle_keyframe_here()
    assert m.keyframes == []


def test_on_prepare_the_viewer_follows_the_smoothness_slider(folder, monkeypatch):
    from harbium_timelapse import previews
    monkeypatch.setattr(previews, "load_small", lambda path, longest: np.full((40, 60, 3), 0.1))
    monkeypatch.setattr(session, "ensure_previews", lambda seq, tracker=None: 0)   # stand-ins have no previews
    m = opened(folder)
    m.go("prepare")
    m.wait()
    until(lambda: m.preview is not None and not m.previewing)
    m.load_view()
    until(lambda: m.view is not None)
    measured, curve = m.prepare_series
    assert np.allclose(m.view.gains, curve - measured)                # After = the preview's corrections
    before = m.view.gains.copy()
    m.set_window(7)                                                    # drag Smoothness towards "follows the light"
    until(lambda: m.preview is not None and not m.previewing and not np.allclose(m.view.gains, before))
    measured, curve = m.prepare_series
    assert np.allclose(m.view.gains, curve - measured)                 # ...and After follows at once


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_the_review_viewer_loads_the_exported_frames(folder, tmp_path):
    m = opened(folder)
    m.project.reach("prepared")
    sky(folder.parent / "final-export")
    m.use_export(folder.parent / "final-export")
    m.choose_preset("share")
    m.set_video_path(tmp_path / "film.mp4")
    m.start_finish()
    m.wait()
    assert m.screen == "done", m.message
    m.load_export_view()
    until(lambda: m.export_view is not None)
    assert len(m.export_view) == len(m.planned.frames) and m.export_view.gains.shape == (len(m.planned.frames), 3)
    m.set_export_cursor(3)
    assert m.export_view_image().ndim == 3


def test_only_the_viewer_on_screen_keeps_its_frames(bare_raws, monkeypatch):
    """Grade's RAW frames and Review's exported frames are each a few hundred MB for a long sequence."""
    from harbium_timelapse.gui.models.viewer import SequenceFrames
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    m.view, m.export_view = SequenceFrames([np.zeros((4, 6, 3), np.uint8)]), SequenceFrames([np.zeros((4, 6, 3), np.uint8)])
    monkeypatch.setattr(m, "can_go", lambda screen: True)
    m.go("done")
    assert m.view is None and m.export_view is not None
    m.export_view = SequenceFrames([np.zeros((4, 6, 3), np.uint8)])
    m.go("grade")
    assert m.export_view is None


def _stand_in(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path as _Path
    spec = importlib.util.spec_from_file_location(
        "rapidraw_stand_in", _Path(__file__).parent.parent / "app" / "test_rapidraw_export.py")
    stand_in = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stand_in)
    program = stand_in.make_stand_in(tmp_path / "bin")
    monkeypatch.setattr(session, "find_rapidraw", lambda: program)
    return program


def test_the_render_format_follows_the_preset():
    from harbium_timelapse.gui.models.export import apply_preset, ExportSettings
    deep = {p: AppModel.renders_deep(apply_preset(ExportSettings(), p)) for p in ("share", "master", "edit")}
    assert deep == {"share": False, "master": True, "edit": True}        # 8-bit H.264 gains nothing from 16-bit


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_export_and_finish_renders_measures_corrects_and_writes_in_one_run(bare_raws, tmp_path, monkeypatch):
    _stand_in(tmp_path, monkeypatch)
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    m.out.video_path = str(tmp_path / "film.mp4")
    m.export_and_finish("share")
    m.wait()
    assert m.screen == "done", m.message
    assert m.project.export.endswith("sunset render") and (tmp_path / "film.mp4").stat().st_size > 0
    asked = (bare_raws.parent / "rapidraw-args.txt").read_text()
    assert "--format jpeg" in asked
    m.screen = "grade"
    m.export_and_finish("share")                                        # nothing changed: nothing rendered
    m.wait()
    assert m.screen == "done", m.message


def test_export_and_finish_refuses_without_the_disk_space(bare_raws, tmp_path, monkeypatch):
    import shutil as _shutil
    _stand_in(tmp_path, monkeypatch)
    m = opened(bare_raws)
    m.choose_editor("rapidraw")
    m.wait()
    monkeypatch.setattr(_shutil, "disk_usage", lambda path: type("U", (), {"free": 10_000_000})())
    m.screen = "grade"
    m.export_and_finish("master")                                         # 16-bit: ~120 MB a frame
    assert m.screen == "grade" and m.job is None and "GB" in m.message and "Share" in m.message
