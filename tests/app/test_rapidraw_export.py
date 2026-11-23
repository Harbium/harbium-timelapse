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
import os, sys
from pathlib import Path
args = sys.argv[1:]
print("Starting headless export...", flush=True)
src, out = Path(args[1]), Path(args[args.index("--output") + 1])
if not src.is_dir() or "fail" in src.name:
    print(f"Headless export failed: Source path does not exist: {{src}}", flush=True)
    sys.exit(1)
raws = sorted(p for p in src.iterdir() if p.suffix.upper() == ".CR3")
print(f"Found {{len(raws)}} images to export. Processing...", flush=True)
out.mkdir(parents=True, exist_ok=True)
for p in raws:
    target = out / (p.stem + "_edited.jpeg")
    target.write_bytes(b"jpeg")
    os.utime(target, (1726946460, 1726946460))      # the photo's own date, as RapidRAW keeps it
    print(f"[INFO] [process_image_for_export] 5472x3648 processed (ROI: 5472x3648) on GPU in 275ms", flush=True)
print("Headless export completed successfully.", flush=True)
'''


@pytest.fixture
def fake_rapidraw(tmp_path):
    program = tmp_path / "bin" / "RapidRAW.exe"
    program.parent.mkdir()
    program.write_text(FAKE.format(python=sys.executable), encoding="utf-8")
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    return program


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
    assert stale.read_bytes() == b"jpeg"                                    # the fresh frame, not the old one
    assert (out / "notes.txt").read_text() == "mine" and (out / "poster.jpg").exists()
