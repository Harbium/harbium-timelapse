"""The export window's model: presets, advanced options, checks, estimates and frames-only exports."""

import shutil

import numpy as np
import pytest
from PIL import Image

from harbium_timelapse.gui.models import AppModel
from harbium_timelapse.gui.models.export import user_presets


def opened_export(tmp_path, n=12):
    folder = tmp_path / "final-export"
    folder.mkdir()
    rng = np.random.default_rng(3)
    base = np.linspace(30, 220, 160)[None, :, None] * np.ones((100, 1, 3))
    for i in range(n):
        Image.fromarray(np.clip(base * 2 ** rng.normal(0, 0.06), 0, 255).astype(np.uint8)).save(folder / f"f{i:02d}.jpg")
    m = AppModel()
    m.open_folder(folder)
    m.wait()
    m.go("finish")
    m.set_video_path(tmp_path / "film.mp4")
    return m


def test_presets_are_matched_not_flagged(tmp_path):
    m = opened_export(tmp_path)
    assert m.preset == "share" and m.out.codec == "h264"
    m.choose_preset("edit")
    assert m.out.codec == "prores" and m.video_path.suffix == ".mov"      # the file follows the format
    for change in ({"output": "both"}, {"fps": 25.0}, {"first": 2, "last": 9}):
        m.set_export(**change)
        assert m.preset == "edit"                                          # not part of any preset
    m.set_export(prores_profile="4444")
    assert m.preset == "custom"
    m.set_export(prores_profile="hq")
    assert m.preset == "edit"                                              # changed back: the preset is back
    m.choose_preset("master")
    m.set_export(bit_depth=10)                                             # Master's own default, said explicitly
    assert m.preset == "master"


def test_your_own_presets_are_saved_and_can_be_deleted(tmp_path):
    m = opened_export(tmp_path)
    m.choose_preset("master")
    m.set_export(quality=14, fps=25.0)
    m.save_preset("My 4K delivery")
    assert "My 4K delivery" in user_presets() and ("My 4K delivery", "My 4K delivery") in m.presets
    m.choose_preset("share")
    m.choose_preset("My 4K delivery")
    assert (m.out.codec, m.out.quality, m.out.fps) == ("h265", 14, 25.0)
    m.save_preset("share")
    assert m.message_kind == "error" and "built-in" in m.message
    m.delete_preset("My 4K delivery")
    assert "My 4K delivery" not in user_presets()


def test_problems_block_the_export_and_say_why(tmp_path):
    m = opened_export(tmp_path)
    assert m.can_finish and "frames at" in m.export_check["estimate"] and "free" in m.export_check["estimate"]
    m.set_export(first=9, last=4)
    assert not m.can_finish and any("frame range" in p for p in m.export_check["problems"])
    m.set_export(first=None, last=None, codec="prores", chroma="420")
    assert any("always 4:2:2" in p for p in m.export_check["problems"])
    m.set_export(codec="h264", chroma=None)
    m.set_video_path("F:/\u2026/film.mp4")
    assert any("shortened example" in p for p in m.export_check["problems"])
    m.set_video_path(tmp_path / "film.mp4")
    m.set_export(output="frames")
    m.set_frames_folder(tmp_path / "final-export")
    assert any("their own folder" in p for p in m.export_check["problems"])
    assert any("HEVC" in n for n in (m.set_export(output="video", codec="h265") or m.export_check["notes"]))


def test_a_frames_only_export_and_the_time_estimate_afterwards(tmp_path):
    m = opened_export(tmp_path)
    m.set_export(output="frames", jpeg_quality=98)
    m.set_frames_folder(tmp_path / "series")
    assert "first export of this kind" in m.export_check["estimate"]
    m.start_finish()
    m.wait()
    assert m.screen == "done", m.message
    assert len(list((tmp_path / "series").glob("*.jpg"))) == 12 and not (tmp_path / "film.mp4").exists()
    assert "s a frame" in " ".join(m.result_lines)
    m.go("finish")
    assert "min" in m.export_check["estimate"]                           # now from this computer's real speed


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_trimmed_video_with_advanced_options(tmp_path):
    m = opened_export(tmp_path)
    m.set_export(codec="h264", chroma="444", quality=20, resolution="native", first=3, last=10)
    m.start_finish()
    m.wait()
    assert m.screen == "done" and (tmp_path / "film.mp4").exists(), m.message
    assert len(m.finished.check) == 8


def test_choices_are_explained_in_plain_words(tmp_path):
    from harbium_timelapse.gui.models.export import ExportSettings, explain
    assert "plays anywhere" in explain(ExportSettings(codec="h264"))
    text = explain(ExportSettings(codec="h265"))
    assert "smaller files" in text and "gradients" in text and "plays everywhere" in text
    assert "for editing" in explain(ExportSettings(codec="h264", chroma="444"))
    assert "for editing" in explain(ExportSettings(codec="prores"))


def test_the_visual_trim_uses_cached_thumbnails(tmp_path):
    from harbium_timelapse.app.thumbs import thumbnail
    m = opened_export(tmp_path, n=30)
    import time
    end = time.time() + 20
    while not m.trim_strip and time.time() < end:
        time.sleep(0.05)
    assert m.trim_frames == 30 and len(m.trim_strip) == 12 and all(p.exists() for _, p in m.trim_strip)
    first = sorted((tmp_path / "final-export").iterdir())[0]
    assert thumbnail(first) == thumbnail(first)                            # made once, then reused
    m.set_trim(5, 20)
    assert (m.out.first, m.out.last, m.preset) == (5, 20, "share")
    m.set_trim(1, 30)
    assert (m.out.first, m.out.last) == (None, None)                       # the whole sequence again
    assert m.frame_time(25) == "0:01.0"


def test_up_to_4k_fits_a_4k_frame_not_just_its_width():
    """Regression: 3:2 frames came out 3840 x 2560 at '4K', H.264 level 6, which Windows Media Player decoded
    with blocky patches. They now fit inside 3840 x 2160."""
    from harbium_timelapse.gui.models.export import ExportSettings, notes, output_size
    s = ExportSettings(resolution="4K")
    assert output_size(s, 5472, 3648) == (3240, 2160)
    assert output_size(s, 6000, 3376) == (3838, 2160) or output_size(s, 6000, 3376)[1] <= 2160
    assert output_size(ExportSettings(resolution="4K"), 1920, 1280) == (1920, 1280)        # never enlarged
    big = ExportSettings(resolution="custom", custom_width=3840)
    assert any("may not play smoothly" in n for n in notes(big, 3840, 2560))
    assert not any("may not play smoothly" in n for n in notes(s, 3240, 2160))
