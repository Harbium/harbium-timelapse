"""Safe writes leave nothing behind, and the caches stay tidy."""

import os
import time

import pytest

from harbium_timelapse.files import keep_within, sweep_temporary, touch, write_atomically


def test_a_failed_write_leaves_no_temporary_file_and_the_original_intact(tmp_path):
    target = tmp_path / "project.json"
    target.write_text("original")

    def broken(temp):
        temp.write_text("half a fi")
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        write_atomically(target, broken)
    assert target.read_text() == "original" and [f.name for f in tmp_path.iterdir()] == ["project.json"]
    write_atomically(target, lambda t: t.write_text("new"))
    assert target.read_text() == "new" and [f.name for f in tmp_path.iterdir()] == ["project.json"]


def test_stale_temporary_files_are_swept_and_fresh_ones_left(tmp_path):
    old, fresh, real = tmp_path / "a.123.tmp.jpg", tmp_path / "b.456.tmp", tmp_path / "c.jpg"
    for f in (old, fresh, real):
        f.write_bytes(b"x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))         # two hours: a run that's gone
    assert sweep_temporary(tmp_path) == 1
    assert not old.exists() and fresh.exists() and real.exists()    # fresh: perhaps a run in progress


def test_the_cache_budget_drops_the_least_recently_used_first(tmp_path):
    files = []
    for i in range(5):
        f = tmp_path / f"{i}.jpg"
        f.write_bytes(b"x" * 1000)
        os.utime(f, (time.time() - 1000 + i, time.time() - 1000 + i))   # 0 oldest ... 4 newest
        files.append(f)
    touch(files[0])                                                       # ...but 0 was just used
    freed = keep_within(tmp_path, 3000)
    assert freed == 2000 and sorted(f.name for f in tmp_path.iterdir()) == ["0.jpg", "3.jpg", "4.jpg"]
