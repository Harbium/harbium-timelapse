"""Writing files safely, and keeping the caches tidy.

``write_atomically`` writes to a temporary file beside the target, then swaps it into place in one step, so a
crash never leaves half a file; if anything fails first, the temporary file is removed, so a full disk or a
locked file never leaves debris behind.

The caches (previews, thumbnails, measurements) live in the user's cache folder and are only ever an
optimisation. ``upkeep`` sweeps temporary files left by a run that was killed, and keeps each cache within a
size budget, removing the least recently used entries first. Entries are "touched" when used, so anything still
in use survives.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

STALE = 3600                        # a temporary file older than an hour belongs to a run that's gone


def write_atomically(path: Path | str, write: Callable[[Path], None], temp_suffix: str = ".tmp") -> None:
    """Call ``write(temp)`` to produce the file, then swap it into place; on any failure the temporary file
    goes and the error is raised as usual."""
    path = Path(path)
    temp = path.with_name(f"{path.name}.{os.getpid()}{temp_suffix}")
    try:
        write(temp)
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def touch(path: Path | str) -> None:
    """Mark a cache entry as just used (its modified time is the cache's notion of recent use)."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def sweep_temporary(folder: Path, now: float | None = None) -> int:
    """Remove temporary files older than an hour (left by a run that was killed). Returns how many."""
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = list(Path(folder).iterdir())
    except OSError:
        return 0
    for f in entries:
        if ".tmp" in f.name and f.is_file():
            try:
                if now - f.stat().st_mtime > STALE:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
    return removed


def keep_within(folder: Path, budget: int) -> int:
    """Keep a cache folder within ``budget`` bytes, removing the least recently used files first. Returns the
    bytes freed."""
    try:
        files = [(f, f.stat()) for f in Path(folder).iterdir() if f.is_file() and ".tmp" not in f.name]
    except OSError:
        return 0
    total = sum(st.st_size for _, st in files)
    freed = 0
    for f, st in sorted(files, key=lambda item: item[1].st_mtime):          # oldest use first
        if total - freed <= budget:
            break
        try:
            f.unlink()
            freed += st.st_size
        except OSError:
            pass
    return freed


BUDGETS = {"previews": 4 * 1024 ** 3, "thumbnails": 256 * 1024 ** 2, "measurements": 256 * 1024 ** 2}


def upkeep() -> dict[str, int]:
    """Tidy every cache: stale temporary files out, each within its budget. Safe to run any time; quick."""
    from . import measure_cache, previews
    from .app import thumbs
    freed = {}
    for name, folder in (("previews", previews._folder()), ("thumbnails", thumbs._folder()),
                         ("measurements", measure_cache._folder())):
        sweep_temporary(folder)
        freed[name] = keep_within(folder, BUDGETS[name])
    return freed
