"""tone-check: the tone and region checks on two folders, before and after finish.

The same engine finish runs on every export (``diagnose``), for frames already
on disk: each frame pair leaves a tiny brightness sample (JPEGs decoded at a
reduced size, so it's quick), and the tone and region checks run on those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .diagnose import Finding, Recorder, check_regions, check_tones


def _load(path: Path) -> np.ndarray:
    from PIL import Image
    from .finish import _read
    if path.suffix.lower() in (".jpg", ".jpeg"):
        with Image.open(path) as im:
            im.draft("RGB", (640, 640))                  # decode at a reduced size: plenty for a tiny sample
            return np.asarray(im.convert("RGB"))
    return _read(path)


def _pair(job: tuple[Path, Path]):
    return _load(job[0]), _load(job[1])


def check_folders(before: Path, after: Path, progress=None) -> tuple[list[Finding], int]:
    from .finish import finished_frames
    from .parallel import ordered_map
    a_frames = {p.name: p for p in finished_frames(after)}
    pairs = [(p, a_frames[p.name]) for p in finished_frames(before) if p.name in a_frames]
    if len(pairs) < 10:
        raise ValueError(f"Only {len(pairs)} frames have the same name in both folders: need at least 10.")
    rec = Recorder()
    for i, (b, a) in enumerate(ordered_map(_pair, pairs)):
        if b.shape[:2] != a.shape[:2]:
            raise ValueError("Before and after frames are different sizes, so the same pixels can't be compared.")
        rec.add(b, a, i, len(pairs))
        if progress:
            progress(i + 1, len(pairs))
    return [check_tones(rec), check_regions(rec)], len(pairs)
