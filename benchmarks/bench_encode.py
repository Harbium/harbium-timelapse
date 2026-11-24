"""Make a film from a folder of frames exactly as harbium-timelapse's Share preset would, but with every
correction at zero: same decoding, scaling (to fit 3840 x 2160), dither, H.264 settings and colour labels.
For comparing other tools' frames like for like: the only difference left is their grade and deflicker.

    python benchmarks/bench_encode.py "C:/Users/James/Pictures/sunset_bench" "C:/Users/James/Videos/sunset-rt.mp4"
"""

import sys
from pathlib import Path

import numpy as np

from harbium_timelapse.app import session
from harbium_timelapse.finish import FinishPlan, finished_frames
from harbium_timelapse.gui.models.app import AppModel
from harbium_timelapse.gui.models.export import ExportSettings, apply_preset


def main(folder: Path, video: Path) -> None:
    frames = finished_frames(folder)
    if not frames:
        sys.exit(f"No frames (JPEG, PNG or TIFF) in {folder}.")
    zero = np.zeros((len(frames), 3))
    plan = FinishPlan(tuple(frames), zero, zero.copy(), None)      # target = measured: no correction at all
    assert not np.any(plan.gains), "every correction must be zero"
    planned = session.Planned(list(frames), plan, 0.0)
    share = apply_preset(ExportSettings(), "share")
    width = AppModel._preset_width(folder, share.width_limit())
    print(f"{len(frames)} frames from {folder} -> {video} (Share: H.264 8-bit, {width} px wide, 24 fps, no correction)")
    done = session.write_export(planned, video=video, fps=share.fps, width=width, codec=share.codec,
                                dither=share.dither, options=share.options(), settings={"preset": "share (bench)"})
    print(f"Written: {done.video} ({done.video.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
