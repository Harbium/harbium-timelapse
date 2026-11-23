"""RapidRAW, through the Editor contract: the reference adapter."""

from __future__ import annotations

from pathlib import Path

from ...deflicker import DEFAULT_WINDOW
from ..base import Editor, PrepareReport
from .deflicker import apply, check, load, plan, undo
from .recover import rapidraw_running
from .sidecar import SUFFIX
from .whitebalance import FIXED, apply_white_balance, plan_white_balance, undo_white_balance

BLOCKING = {"too_short", "mixed_files", "mixed_cameras"}


class RapidRAW(Editor):
    name = "rapidraw"
    label = "RapidRAW"

    def detect(self, folder: Path) -> bool:
        folder = Path(folder)
        return folder.is_dir() and any(p.name.endswith(SUFFIX) for p in folder.iterdir())

    def is_running(self) -> bool:
        return rapidraw_running()

    def load(self, folder: Path, progress=None):
        return load(folder, progress=progress)

    def prepare(self, seq, *, window: int = DEFAULT_WINDOW, strength: float = 1.0,
                white_balance: bool = True, dry_run: bool = False, force: bool = False,
                progress=None, raw_brightness=None) -> PrepareReport:
        report = PrepareReport(editor=self.label, frames=len(seq), dry_run=dry_run)
        report.problems = check(seq)
        blocking = [p for p in report.problems if p.kind in BLOCKING]
        if blocking and not force and not dry_run:
            raise ValueError("\n".join(str(p) for p in blocking) + "\nNot changing anything until that's "
                             "sorted: the results would be wrong. (If you're sure, add --force.)")

        if not dry_run:
            from .keyframes import fill_missing
            created = fill_missing(seq)
            if created:
                report.notes.append(f"Created edit files for {created} frames that had none, holding only the "
                                    "correction: your grade reaches them when you blend keyframes. Undo removes "
                                    "them again.")
                report.problems = [p for p in report.problems if p.kind != "no_sidecar"]
        if white_balance:
            try:
                wb = plan_white_balance(seq)
                if wb.biggest_jump_before < FIXED:
                    report.notes.append(f"White balance: fixed in the camera (about {wb.as_shot[0]:.0f} K), so there's nothing to steady.")
                else:
                    report.notes.extend(f"White balance: {n}" for n in wb.notes)
                    report.notes.append(
                        f"White balance: steadied (the camera's jumped up to {wb.biggest_jump_before:.0f} K between "
                        "frames). Keyframes and your own white balance are left alone.")
                    if not dry_run:
                        changed, kept = apply_white_balance(seq, wb)
                        if kept:
                            report.notes.append(f"White balance: kept your own on {kept} frames.")
            except ValueError as e:
                report.notes.append(f"White balance left as it is: {e}")

        p = plan(seq, window=window, strength=strength, progress=progress, raw_brightness=raw_brightness)
        report.deflicker = p.flicker
        if not dry_run:
            report.changed = apply(p)
        return report

    def handoff_steps(self) -> list[str]:
        return ["Open the folder in RapidRAW: the corrections are already in place. Grade as you like.",
                "Leave Dehaze at zero: it reacts to exposure changes and can make contrast jump.",
                "Export every frame as JPEG into a new, empty folder next to the RAW files.",
                "Your originals are untouched, and every edit file was backed up before it changed."]

    def undo(self, seq) -> int:
        from .state import remove_created
        n = undo(seq.folder)
        undo_white_balance(seq)
        remove_created(seq.folder)          # files it created go again, unless you've edited them since
        return n
