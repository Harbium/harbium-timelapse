"""tone-check: the same checks finish runs, on two folders already on disk."""

import numpy as np
from PIL import Image

from harbium_timelapse import cli

rng = np.random.default_rng(7)
H, W, N = 60, 90, 30
scene = np.concatenate([np.full((20, W), 0.02), np.full((20, W), 0.12), np.full((20, W), 0.5)])
scene = (scene * np.exp(rng.normal(0, 0.15, (H, W))))[..., None] * np.ones(3)
flicker = rng.normal(0, 0.06, N)


def save(folder, frames):
    folder.mkdir()
    for i, f in enumerate(frames):
        Image.fromarray((np.clip(f, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)).save(folder / f"{i:03d}.png")


def test_the_command_reports_tones_and_regions(tmp_path, capsys):
    save(tmp_path / "before", [scene * 2 ** f for f in flicker])
    save(tmp_path / "after", [scene for _ in flicker])
    assert cli.main(["tone-check", str(tmp_path / "before"), str(tmp_path / "after")]) == 0
    out = capsys.readouterr().out
    assert f"{N} frames compared" in out and "Tones:" in out and "Regions:" in out


def test_mismatched_folders_are_explained(tmp_path, capsys):
    save(tmp_path / "before", [scene for _ in range(12)])
    (tmp_path / "after").mkdir()
    assert cli.main(["tone-check", str(tmp_path / "before"), str(tmp_path / "after")]) == 1
