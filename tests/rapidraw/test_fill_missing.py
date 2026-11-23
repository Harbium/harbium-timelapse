"""Prepare on a folder where only some frames have RapidRAW edit files (RapidRAW 1.6.3+ only writes one once a
frame is edited). Regression: Apply refused with '129 frames have no RapidRAW sidecar'. Frames without one get
a minimal file holding only the correction: nobody's grade is copied, since in-between frames get yours when
you blend keyframes."""

import numpy as np
import pytest

from test_rapidraw import make_sequence
from harbium_timelapse.editors.rapidraw import load
from harbium_timelapse.editors.rapidraw.adapter import RapidRAW
from harbium_timelapse.editors.rapidraw.keyframes import blend
from harbium_timelapse.editors.rapidraw.sidecar import read_sidecar, sidecar_path, write_sidecar
from harbium_timelapse.editors.rapidraw.state import read_ledger

LIGHT = np.random.default_rng(1).normal(0, 0.05, 30)


@pytest.fixture
def two_keyframes(tmp_path):
    images = make_sequence(tmp_path, 30)
    for i, contrast in ((0, 10), (29, 50)):
        graded = read_sidecar(images[i])
        graded["adjustments"].update(contrast=contrast, exposure=0.0)
        graded["tags"] = ["color:purple"]
        write_sidecar(images[i], graded)
    for image in images[1:29]:
        sidecar_path(image).unlink()
    return tmp_path, images


def test_apply_creates_minimal_files_holding_only_the_correction(two_keyframes):
    path, images = two_keyframes
    report = RapidRAW().prepare(load(path), window=25, raw_brightness=LIGHT)
    assert report.changed == 30 and any("Created edit files for 28 frames" in n for n in report.notes)
    made = read_sidecar(images[15])
    assert set(made["adjustments"]) <= {"exposure", "whiteBalance"}           # nobody's grade copied
    assert not made.get("tags") and not made.get("rating")


def test_blending_afterwards_brings_your_grade_and_keeps_the_correction(two_keyframes):
    path, images = two_keyframes
    RapidRAW().prepare(load(path), window=25, raw_brightness=LIGHT)
    applied = read_ledger(path)[images[15].name]["applied"]
    blend(load(path))
    made = read_sidecar(images[15])
    assert 10 < made["adjustments"]["contrast"] < 50                           # blended from the keyframes
    assert made["adjustments"]["exposure"] == pytest.approx(applied, abs=1e-4)  # 0 blended + the correction


def test_undo_removes_created_files_but_keeps_one_you_edited_since(two_keyframes):
    path, images = two_keyframes
    editor = RapidRAW()
    editor.prepare(load(path), window=25, raw_brightness=LIGHT)
    mine = read_sidecar(images[7])
    mine["adjustments"]["shadows"] = 12                                        # graded later, in RapidRAW
    write_sidecar(images[7], mine)
    editor.undo(load(path))
    assert read_sidecar(images[0])["adjustments"]["contrast"] == 10            # your keyframe stays
    assert read_sidecar(images[7])["adjustments"]["shadows"] == 12             # so does the one you edited since
    assert all(read_sidecar(images[k]) is None for k in range(1, 29) if k != 7)   # the rest are gone


def test_a_folder_with_no_edits_at_all_can_still_be_prepared(tmp_path):
    images = make_sequence(tmp_path, 30)
    for image in images:
        sidecar_path(image).unlink()
    report = RapidRAW().prepare(load(tmp_path), window=25, raw_brightness=LIGHT)
    assert report.changed == 30 and all(read_sidecar(i) is not None for i in images)
