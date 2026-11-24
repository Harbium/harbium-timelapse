"""Exporting through RapidRAW's command line (RapidRAW 1.6+), against a stand-in that behaves like the real one
on James's PC (RapidRAW 1.7.1): same lines printed, files named <frame>_edited.jpeg, the photo's own date kept
as the modified time, exit code 0 on success and 1 with "Headless export failed: ..." otherwise."""

import stat
import sys
import time

import pytest

from harbium_timelapse.app import session
from harbium_timelapse.sequencer import load_sequence

FAKE = r'''#!{python}
import sys
from pathlib import Path
args = sys.argv[1:]
print("Starting headless export...", flush=True)
src, out = Path(args[1]), Path(args[args.index("--output") + 1])
(out.parent / "rapidraw-args.txt").write_text(" ".join(args[2:]))       # what it was asked, for the tests
if not src.is_dir() or "fail" in src.name:
    print(f"Headless export failed: Source path does not exist: {{src}}", flush=True)
    sys.exit(1)
fmt = args[args.index("--format") + 1] if "--format" in args else "jpeg"
skip_existing = "--on-conflict" in args and args[args.index("--on-conflict") + 1] == "overwrite-if-changed"
raws = sorted(p for p in src.iterdir() if p.suffix.upper() == ".CR3")
print(f"Found {{len(raws)}} images to export. Processing...", flush=True)
out.mkdir(parents=True, exist_ok=True)
import os
import numpy as np
skipped = 0
for i, p in enumerate(raws):
    target = out / (p.stem + ("_edited.tiff" if fmt == "tiff" else "_edited.jpeg"))
    if skip_existing and target.exists():
        skipped += 1
        continue
    level = 0.25 * 2 ** (0.05 * np.sin(i))                                # a little flicker to correct
    picture = np.clip(np.full((40, 60, 3), level) * np.linspace(0.6, 1.4, 60)[None, :, None], 0, 1)
    if fmt == "tiff":
        import tifffile
        tifffile.imwrite(target, (picture * 65535).astype(np.uint16))
    else:
        from PIL import Image
        Image.fromarray((picture * 255).astype(np.uint8)).save(target, quality=95)
    os.utime(target, (1726946460, 1726946460))      # the photo's own date, as RapidRAW keeps it
    print("[INFO] [process_image_for_export] 5472x3648 processed (ROI: 5472x3648) on GPU in 275ms", flush=True)
if skipped:
    print(f"Skipped {{skipped}} of {{len(raws)}} images because their output files already exist.", flush=True)
print("Headless export completed successfully.", flush=True)
'''


def make_stand_in(folder):
    """The stand-in as something the system can run: on Linux a script with a #! line; on Windows (which
    ignores #! and won't run a script named .exe) the script beside a .cmd that calls Python with it."""
    import os
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "rapidraw_stand_in.py"
    script.write_text(FAKE.format(python=sys.executable), encoding="utf-8")
    if os.name == "nt":
        program = folder / "RapidRAW.cmd"
        program.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return program
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.fixture
def fake_rapidraw(tmp_path):
    return make_stand_in(tmp_path / "bin")


def raws(folder, n=6):
    folder.mkdir()
    for i in range(n):
        (folder / f"181A{834 + i:04d}.CR3").write_bytes(b"")
    return load_sequence(folder)


class Steps:
    def __init__(self):
        self.seen = []
        self.token = type("T", (), {"check": lambda self: None})()

    def stage(self, name):
        return lambda done, total: self.seen.append((name, done, total))


def test_rapidraw_exports_every_frame_beside_the_raw_folder(tmp_path, fake_rapidraw):
    seq = raws(tmp_path / "sunset")
    steps = Steps()
    out = session.export_with_rapidraw(seq, tracker=steps, program=fake_rapidraw)
    assert out == tmp_path / "sunset export" and len(list(out.glob("*_edited.jpeg"))) == 6
    assert steps.seen[-1] == ("Frames", 6, 6)                                 # progress from RapidRAW's own lines
    assert session.export_with_rapidraw(seq, program=fake_rapidraw) == tmp_path / "sunset export 2"   # never mixed


def test_a_failed_export_says_why(tmp_path, fake_rapidraw):
    seq = raws(tmp_path / "fail-me")
    with pytest.raises(ValueError, match="Headless export failed"):
        session.export_with_rapidraw(seq, program=fake_rapidraw)


def test_the_watcher_spots_rapidraw_style_files_with_their_old_dates(tmp_path, fake_rapidraw):
    """Named 181A0834_edited.jpeg and dated September: the watcher wanted exact names and recent dates."""
    seq = raws(tmp_path / "sunset")
    since = time.time() - 1
    out = session.export_with_rapidraw(seq, program=fake_rapidraw)
    s = session.export_status(seq.folder, 6, since)
    assert (s.state, s.folder, s.frames) == ("arriving", out, 6)


def test_exporting_again_replaces_the_frames_and_nothing_else(tmp_path, fake_rapidraw):
    """After blending again: the old frames go, fresh ones arrive in the same folder, other files stay."""
    seq = raws(tmp_path / "sunset")
    out = session.export_with_rapidraw(seq, program=fake_rapidraw)
    (out / "notes.txt").write_text("mine")
    (out / "poster.jpg").write_bytes(b"not a frame")
    stale = out / "181A0834_edited.jpeg"
    stale.write_bytes(b"old grade")
    again = session.export_with_rapidraw(seq, out=out, program=fake_rapidraw, replace=True)
    assert again == out and len(list(out.glob("*_edited.jpeg"))) == 6
    assert stale.read_bytes()[:2] == b"\xff\xd8"                           # the fresh frame (a JPEG), not the old one
    assert (out / "notes.txt").read_text() == "mine" and (out / "poster.jpg").exists()


def test_rendering_again_only_renders_what_changed_and_counts_the_rest_as_done(tmp_path, fake_rapidraw):
    seq = raws(tmp_path / "sunset")
    steps = Steps()
    out = session.render_with_rapidraw(seq, deep=False, tracker=steps, program=fake_rapidraw)
    assert out == tmp_path / "sunset render" and len(list(out.glob("*_edited.jpeg"))) == 6
    asked = (tmp_path / "rapidraw-args.txt").read_text()
    assert "--on-conflict overwrite-if-changed" in asked and "--color-space srgb" in asked and "--quality 95" in asked
    (out / "181A0836_edited.jpeg").unlink()                                   # one frame's edit changed
    steps = Steps()
    session.render_with_rapidraw(seq, deep=False, tracker=steps, program=fake_rapidraw)
    assert steps.seen[-1] == ("Frames", 6, 6)                                 # 1 rendered + 5 skipped = all done


def test_deep_renders_are_16_bit_tiff_in_their_own_folder(tmp_path, fake_rapidraw):
    pytest.importorskip("tifffile")
    seq = raws(tmp_path / "sunset")
    out = session.render_with_rapidraw(seq, deep=True, program=fake_rapidraw)
    assert out == tmp_path / "sunset render 16-bit" and len(list(out.glob("*.tiff"))) == 6
    assert "--format tiff --tiff-bit-depth 16" in (tmp_path / "rapidraw-args.txt").read_text()
    assert session.render_needs(seq, deep=True) == 0                         # all rendered: nothing more needed
