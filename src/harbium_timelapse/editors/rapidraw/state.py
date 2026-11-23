"""What this tool keeps in ``.timelapse-core/`` beside your photos.

- ``originals/``: a copy of each sidecar, taken the first time a command
  changes its edit (blend or deflicker). ``restore`` puts them all back.
  Purple labels are left as they are.
- ``rapidraw.json``: the deflicker ledger, so a second run replaces the
  first rather than stacking on top of it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..common import (  # noqa: F401  (re-exported: RapidRAW's modules import these from here)
    STATE_DIR, LedgerDamaged, folder_lock, read_json, record_kept_in_step, write_json,
)
from ..common import read_record, write_record
from .sidecar import sidecar_path

LEDGER = "rapidraw.json"


def keep_original(image: Path) -> None:
    """Copy the sidecar aside before its first change. Later calls do nothing."""
    side = sidecar_path(image)
    backup = image.parent / STATE_DIR / "originals" / side.name
    if side.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(side, backup)


def restore(folder: Path | str) -> int:
    """Put every sidecar back exactly as it was before this tool touched it."""
    folder = Path(folder)
    originals = folder / STATE_DIR / "originals"
    count = 0
    with folder_lock(folder):
        if originals.is_dir():
            for backup in originals.iterdir():
                if backup.name.endswith(".tmp"):
                    continue
                shutil.copy2(backup, folder / backup.name)
                count += 1
        clear_ledger(folder)
    return count


def read_ledger(folder: Path) -> dict:
    return read_record(folder, LEDGER)


def write_ledger(folder: Path, frames: dict) -> None:
    write_record(folder, LEDGER, frames)


def ledger_kept_in_step(folder: Path):
    """RapidRAW's record, kept in step with the files (see ``common.record_kept_in_step``)."""
    return record_kept_in_step(folder, LEDGER)


def clear_ledger(folder: Path) -> None:
    """Forget deflicker, and any refine passes built on it."""
    (folder / STATE_DIR / LEDGER).unlink(missing_ok=True)
    (folder / STATE_DIR / "refine.json").unlink(missing_ok=True)



# --- edit files this tool created ----------------------------------------------------------------------------

CREATED = "created.json"


def _created_path(folder: Path) -> Path:
    return Path(folder) / STATE_DIR / CREATED


def read_created(folder: Path | str) -> list[str]:
    """Edit files this tool created (by sidecar name) to hold its corrections."""
    import json
    try:
        return list(json.loads(_created_path(Path(folder)).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return []


def note_created(folder: Path | str, side: Path) -> None:
    import json
    created = read_created(folder)
    if side.name not in created:
        created.append(side.name)
    path = _created_path(Path(folder))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(created, indent=1), encoding="utf-8", newline="")


def holds_nothing_of_yours(data: dict | None) -> bool:
    """True for a file with no rating, no labels, and no settings besides a zero exposure or a white balance
    (what's left of this tool's corrections once they're undone)."""
    import json as _json
    if data is None:
        return True
    adj = data.get("adjustments") or {}
    extra = set(adj) - {"exposure", "whiteBalance"}
    return (not extra and not data.get("rating") and not data.get("tags")
            and float(adj.get("exposure") or 0.0) == 0.0 and _json.dumps(adj.get("whiteBalance")) in ("null",))


def remove_created(folder: Path | str) -> tuple[int, int]:
    """Delete the edit files this tool created, unless they now hold something of yours (a grade, a label, a
    rating, from blending or from RapidRAW). Returns (removed, kept)."""
    from .sidecar import read_sidecar
    folder = Path(folder)
    removed = kept = 0
    for name in read_created(folder):
        side = folder / name
        if not side.exists():
            continue
        image = folder / name[:-len(".rrdata")]
        if holds_nothing_of_yours(read_sidecar(image)):
            side.unlink()
            removed += 1
        else:
            kept += 1
    _created_path(folder).unlink(missing_ok=True)
    return removed, kept
