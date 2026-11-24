"""Small thumbnails of frames, made once and kept in the user cache folder.

Used for the visual trim (and later the live preview). JPEGs are decoded at a
reduced size directly (Pillow's draft mode), so a thumbnail of a 20-megapixel
frame takes a fraction of a full decode. Keyed by the file's path, size and
modified time: a changed frame gets a new thumbnail, and the file name changes
with it (Flet keeps showing an old image if a file is replaced under the same
name).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

WIDTH = 240


def _folder() -> Path:
    from ..exif import _cache_dir
    return _cache_dir() / "thumbs"


def thumbnail(path: Path, width: int = WIDTH) -> Path:
    path = Path(path)
    st = path.stat()
    key = hashlib.sha256(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{st.st_ctime_ns}|{width}"
                         .encode()).hexdigest()[:24]               # ctime: re-exported frames keep their dates
    target = _folder() / f"{key}.jpg"
    if target.exists():
        from ..files import touch
        touch(target)                                     # just used: kept longest by the cache upkeep
        return target
    from PIL import Image
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (width, width))
        im = im.convert("RGB")
        im.thumbnail((width, width * 2))
        target.parent.mkdir(parents=True, exist_ok=True)
        from ..files import write_atomically
        write_atomically(target, lambda t: im.save(t, format="JPEG", quality=85), ".tmp.jpg")
    return target
