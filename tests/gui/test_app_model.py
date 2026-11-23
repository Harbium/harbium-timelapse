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
    assert len(measured) == len(curve) == N and "wobble" in m.prepare_summary
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
