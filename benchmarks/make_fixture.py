"""A compact copy of a sequence's rendered frames, for the real-sequence benchmark: every frame downsized to
640 pixels wide (same names, same order), zipped. About 15 MB for 200 frames: small enough to send.

    python benchmarks/make_fixture.py "C:/Users/James/Pictures/sunset-keys render" ~/sunset-fixture.zip
"""

import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from harbium_timelapse.finish import finished_frames

WIDTH = 640


def main(folder: Path, target: Path) -> None:
    frames = finished_frames(folder)
    if not frames:
        sys.exit(f"No frames in {folder}.")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as z:
        for i, f in enumerate(frames, 1):
            with Image.open(f) as im:
                im = im.convert("RGB")
                im.thumbnail((WIDTH, WIDTH * 2), Image.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=92)
            z.writestr(f.stem + ".jpg", buf.getvalue())
            print(f"\r{i} of {len(frames)}", end="", flush=True)
    print(f"\nWritten {target} ({target.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
