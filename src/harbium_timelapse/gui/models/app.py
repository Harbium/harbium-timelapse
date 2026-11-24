"""The app's state and commands, screen by screen, in plain Python.

Screens read properties and call commands; they never decide anything. Work
runs in background jobs (``app.jobs``); their events come back through
``post``, which the screen layer sets to "run this on the window's event loop"
(tests use "run it now"). After any change, ``changed()`` tells the screens to
redraw.

The operations themselves come from ``app.session``, passed in so tests can use
quick stand-ins where real RAW decoding isn't possible.
"""

from __future__ import annotations

import re

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...app import config
from ...app import session as default_session
from ...app.jobs import Done, Failed, JobRunner, Stopped
from ...app.project import STAGES, load_project
from ...progress import Progress
from .export import (BUILT_IN, RESOLUTIONS, ExportSettings, apply_preset, delete_user_preset, describe,
                     matching_preset,
                     estimate_bytes, estimate_seconds, free_bytes, notes, output_size, problems, remember_speed,
                     save_user_preset, user_presets)

STAGE_BAR = (("folder", "Folder"), ("prepare", "Prepare"), ("grade", "Grade"), ("finish", "Finish"),
             ("done", "Review"))

PRESETS = {k: {"label": v["label"], "detail": v["detail"], "codec": v["settings"].get("codec", "h264"),
               "width": (RESOLUTIONS.get(v["settings"].get("resolution", "native")) or (None,))[0],
               "suffix": ".mov" if v["settings"].get("codec") == "prores" else ".mp4"}
           for k, v in BUILT_IN.items()}

STRIP = 12                      # thumbnails across the sequence in the visual trim
STOPPED = "Stopped. Everything written so far is consistent: run it again to carry on, or undo."


@dataclass
class JobState:
    name: str
    stage: str = "Starting"
    done: int = 0
    total: int = 0
    cancelling: bool = False

    @property
    def fraction(self) -> float | None:
        return self.done / self.total if self.total else None

    @property
    def text(self) -> str:
        if self.cancelling:
            return "Stopping at the next safe point..."
        return f"{self.stage} {self.done} of {self.total}" if self.total else f"{self.stage}..."


def _wobble(values) -> float:
    d = np.diff(np.asarray(values, dtype=np.float64))
    d = d[np.isfinite(d)]
    return float(np.std(d) / np.sqrt(2)) if d.size > 1 else 0.0


class AppModel:
    def __init__(self, post: Callable[[Callable[[], None]], None] | None = None, runner: JobRunner | None = None,
                 services=None, clock: Callable[[], float] = time.time):
        self.post = post or (lambda fn: fn())
        self.runner = runner or JobRunner()
        self.s = services or default_session
        self.clock = clock
        self._listeners: list[Callable[[], None]] = []
        self._handle = None
        self._last_handle = None
        self.screen = "start"
        self.job: JobState | None = None
        self.message: str | None = None
        self.message_kind = "info"
        self.recent = config.recent_projects()
        self.project = None
        self.opened = None
        self._reset_project_state()

    def _reset_project_state(self) -> None:
        self.preview_images = None            # small linear previews of the RAW frames, for time slices
        self.preview_status = ""
        self.preview_progress = None          # (done, total) while previews are being made or loaded
        self._slice = None
        self.measured = None
        self.preview = None
        self.previewing = False
        self._preview_wanted = None
        self.metadata_saved = False
        self.export_since = None
        self.export = None
        self.out = ExportSettings()
        self.trim_frames, self.trim_strip, self.trim_ends, self.trim_version = 0, [], (), 0
        self.out_version = 0            # goes up whenever an export setting changes (screens rebuild on it)
        self._check_cache = None
        self.planned = None
        self.finished = None
        self.current = None
        self.view = None                      # the viewer's frames (SequenceFrames), loaded in the background
        self.view_status = ""
        self.cursor = 0                       # the playhead: the frame shown above the timeline
        self.view_mode = "after"              # after, before, wipe or flicker
        self.export_view = None               # the Review screen's viewer: the exported (graded) frames
        self.export_view_status = ""
        self.export_cursor = 0
        self.wipe = 0.5
        self.keyframes = None                 # [{"frame": i, "reason": why}], or None before any are chosen
        self.keyframes_state = ""             # "", "chosen" (not labelled yet), "labelled", "blended"

    # --- listeners ---------------------------------------------------------------------------

    def subscribe(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def changed(self) -> None:
        for fn in list(self._listeners):
            fn()

    def _notice(self, text: str, kind: str = "info") -> None:
        self.message, self.message_kind = text, kind

    def dismiss_message(self) -> None:
        self.message = None
        self.changed()

    # --- jobs -----------------------------------------------------------------------------------

    def _run(self, name: str, work, on_done) -> bool:
        if self.job is not None:
            return False                                  # one job at a time
        self.job = JobState(name)
        self.message = None

        def on_event(event):
            self.post(lambda: self._event(event, on_done))
        self._handle = self._last_handle = self.runner.start(name, work, on_event)
        self.changed()
        return True

    def _event(self, event, on_done) -> None:
        if isinstance(event, Progress):
            if self.job is not None:
                self.job.stage, self.job.done, self.job.total = event.stage, event.done, event.total
                self.changed()
            return
        self.job, self._handle = None, None
        if isinstance(event, Done):
            try:
                on_done(event.result)
            except ValueError as e:
                self._notice(str(e), "error")
        elif isinstance(event, Stopped):
            self._notice(STOPPED)
        elif isinstance(event, Failed):
            self._notice(event.message, "error")
        self.changed()

    def cancel(self) -> None:
        if self._handle is not None and self.job is not None:
            self._handle.cancel()
            self.job.cancelling = True
            self.changed()

    def wait(self, timeout: float = 60) -> None:
        """For tests and scripts: until the current job has finished."""
        if self._last_handle is not None:
            self._last_handle.wait(timeout)

    # --- navigation -----------------------------------------------------------------------------

    def _reached(self, stage: str) -> bool:
        return self.project is not None and STAGES.index(self.project.stage) >= STAGES.index(stage)

    def can_go(self, screen: str) -> bool:
        if self.job is not None:
            return False
        if screen == "start":
            return True
        if self.opened is None:
            return False
        return {
            "folder": True,
            "prepare": self.opened.editor is not None and not self.opened.blocking,
            "grade": self._reached("prepared"),
            "finish": True,
            "done": self.finished is not None,
        }.get(screen, False)

    def stage_items(self) -> list[tuple[str, str, str]]:
        """(screen, label, state) for the stage bar; state is current, done, open or locked."""
        done_for = {"folder": "opened", "prepare": "prepared", "grade": "exported", "finish": "finished"}
        items = []
        no_editor = self.opened is not None and self.opened.editor is None
        for screen, label in STAGE_BAR:
            if no_editor and screen in ("prepare", "grade"):
                state = "locked"                       # nothing to prepare: these steps don't apply
            elif screen == self.screen:
                state = "current"
            elif screen in done_for and self._reached(done_for[screen]) and screen != "folder":
                state = "done"
            elif screen == "folder" and self.opened is not None:
                state = "done"
            else:
                state = "open" if self.can_go(screen) else "locked"
            items.append((screen, label, state))
        return items

    def go(self, screen: str) -> None:
        if not self.can_go(screen):
            return
        self.screen = screen
        # Only one viewer is ever on screen: free the other's frames (a few hundred MB for a long sequence);
        # coming back loads them again in a few seconds.
        if screen in ("finish", "done"):
            self.view = None
        elif screen in ("prepare", "grade"):
            self.export_view = None
        if screen == "prepare":
            self._ensure_measured()
        elif screen == "grade":
            if self.export_since is None:
                self.export_since = self.clock()
            self._ensure_measured()                    # instant from the cache: brings back the time slice
            if self.measured is not None:
                self.request_preview()
                self.load_previews()
            self.load_keyframes()
        elif screen == "finish":
            self._check_cache = None                    # free space and remembered speeds may have changed
            self.request_trim_thumbs()
            if self.video_path is None:
                self._default_video_path()
        self.changed()

    # --- start --------------------------------------------------------------------------------

    def open_folder(self, folder: str | Path) -> None:
        folder = Path(folder)
        if not folder.is_dir():
            self._notice(f"Couldn't open {folder}: it isn't a folder.", "error")
            self.changed()
            return
        self._run("Opening", lambda t: self.s.open_folder(folder, t), lambda opened: self._opened(folder, opened))

    def _opened(self, folder: Path, opened) -> None:
        project = load_project(folder)                    # raises ValueError for a newer version: shown as a notice
        self._reset_project_state()
        self.opened, self.project = opened, project
        if project.note:
            self._notice(project.note)
        if self.opened.editor is not None:
            project.editor = self.opened.editor.name
        elif not project.export and not self.raw_without_edits:
            project.export = str(folder)                 # no editor files: these frames are the export
        if project.export:
            self.export_since = 0.0
        config.remember_project(folder, len(opened.seq), project.stage)
        self.recent = config.recent_projects()
        self.screen = "folder"

    # --- folder check ---------------------------------------------------------------------------

    @property
    def facts(self) -> list[tuple[str, str]]:
        if self.opened is None:
            return []
        o = self.opened
        return [("Frames", str(len(o.seq))), ("Camera", o.camera), ("Shot", o.span),
                ("Editor", o.editor.label if o.editor else "none found")]

    @property
    def problems(self) -> list[dict]:
        if self.opened is None:
            return []
        cards, changes = [], [p for p in self.opened.problems if p.kind == "settings_change"]
        if self.raw_without_edits:
            cards.append({"serious": True, "note": False, "choose_editor": True,
                          "message": "Which editor will you grade these RAW files in?",
                          "fix": "RapidRAW: the app creates the edit files it needs. Lightroom: import the folder, "
                                 "select every frame, choose Metadata > Save Metadata to Files, then open the folder "
                                 "here again."})
        if self.opened.editor is not None and self.opened.editor.name == "rapidraw":
            missing = [p for p in self.opened.problems if p.kind == "no_sidecar"]
            if missing:
                cards.append({"serious": False, "note": True, "fix": "",
                              "message": "Some frames have no RapidRAW edit files yet. The app creates them when "
                                         "it needs them."})
        for p in [q for q in self.opened.problems if not (q.kind == "no_sidecar" and self.opened.editor is not None
                                                          and self.opened.editor.name == "rapidraw")]:
            if p.kind != "settings_change":
                cards.append({"serious": p.serious, "message": p.message, "fix": p.fix, "note": False})
        if changes:
            first, last = changes[0].message, changes[-1].message
            def name(m: str) -> str:
                return m.split(" at ", 1)[-1].split(":", 1)[0] if " at " in m else ""
            span = f" ({name(first)} to {name(last)})" if len(changes) > 1 else f" at {name(first)}"
            cards.append({"serious": False, "note": True, "fix": "",
                          "message": f"{len(changes)} exposure change{'s' if len(changes) > 1 else ''} while shooting"
                                     f"{span}. Each is smoothed into a gentle ramp.",
                          "details": [p.message for p in changes]})
        return cards

    @property
    def next_step(self) -> tuple[str, str]:
        if self.opened is not None and self.opened.editor is not None:
            return "prepare", "Next: preview corrections"
        if self.raw_without_edits:
            return "prepare", "Choose your editor first"                 # disabled until an editor is chosen
        return "finish", "Next: finish an export"

    @property
    def raw_without_edits(self) -> bool:
        """RAW files with no editor edit files beside them: Prepare has nowhere to write, and Finish needs exports."""
        from ...sequencer import RAW_EXTENSIONS
        return (self.opened is not None and self.opened.editor is None
                and any(f.path.suffix.lower() in RAW_EXTENSIONS for f in self.opened.seq))

    @property
    def resume(self) -> tuple[str, str] | None:
        """Where an earlier session got to, and the screen to carry on from."""
        if self.project is None or self.project.stage == "opened":
            return None
        target = {"prepared": "grade", "exported": "finish", "finished": "finish"}[self.project.stage]
        words = {"prepared": "corrections applied", "exported": "exported", "finished": "finished"}
        return target, f"Last time this folder got as far as: {words[self.project.stage]}."

    # --- prepare --------------------------------------------------------------------------------

    @property
    def window(self) -> int:
        return int(self.project.settings["window"]) if self.project else 25

    @property
    def white_balance_available(self) -> bool:
        """Lightroom's white balance isn't supported yet, so its switch isn't shown."""
        return self.opened is not None and self.opened.editor is not None and self.opened.editor.name != "lightroom"

    @property
    def white_balance(self) -> bool:
        return bool(self.project.settings["white_balance"]) if self.project else True

    def _ensure_measured(self) -> None:
        if self.measured is None and self.opened is not None:
            self._run("Measuring", lambda t: self.s.measure(self.opened.seq, t), self._measured)

    def _measured(self, measured) -> None:
        self.measured = measured
        self.request_preview()
        self.load_previews()

    # --- time slices from the RAW previews (Prepare and Grade) ----------------------------------------

    def load_previews(self) -> None:
        """Load the frames' small previews in the background, first making any that are missing (folders
        measured before previews existed: one more read of the RAW files, at measuring speed)."""
        if self.opened is None or self.opened.editor is None or self.preview_images is not None:
            return
        seq = self.opened.seq
        from ... import previews

        def work(t):
            from ...timeslice import MODES, Slice
            self.s.ensure_previews(seq, t)
            slices, means = None, []
            loading = t.stage("Loading previews")
            for i, f in enumerate(seq):           # each preview hands its own pixels to every mode, then goes
                loading(i + 1, len(seq))
                image = previews.load(f.path)
                if image is None:
                    return None
                if slices is None:
                    h, w = image.shape[:2]
                    slices = {mode: Slice(h, w, len(seq), mode, dtype=np.float16) for mode in MODES}   # half precision
                if image.shape[:2] != (slices["left-right"].h, slices["left-right"].w):
                    return None                   # previews of different sizes: can't make one picture
                for s in slices.values():
                    s.take(i, image)
                means.append(float(image.mean()))
            return slices, means

        def event(e):
            if isinstance(e, Progress):
                self.preview_progress = (e.done, e.total)
                self.preview_status = (f"Making previews: {e.done} of {e.total} (once for this folder)"
                                       if e.stage == "Making previews" else
                                       f"Loading previews: {e.done} of {e.total}")
            elif isinstance(e, Done):
                if e.result is None:
                    self.preview_status = "Previews couldn't be made for every frame."
                else:
                    self.preview_images, means = e.result
                    typical = float(np.median([m for m in means if m > 0] or [1.0]))
                    self._slice_scale = 0.18 / typical
                    self.preview_status = ""
                    self.preview_progress = None
            elif isinstance(e, Failed):
                self.preview_status = f"Previews couldn't be made: {e.message}"
            self.changed()
        self.preview_status = "Loading previews..."
        self.preview_progress = (0, 0)
        self.runner.start("Previews", work, lambda e: self.post(lambda: event(e)))

    def time_slice(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Before and after the prepared corrections, as 8-bit pictures, from the RAW previews. Instant: only
        one band per frame is recomposed when the corrections change."""
        if self.preview_images is None or self.preview is None or self.preview.deflicker is None:
            return None
        key = (id(self.preview), self.slice_mode)
        if self._slice is None or self._slice[0] != key:
            from ...timeslice import to_display
            s = self.preview_images[self.slice_mode]
            stops = np.nan_to_num(np.asarray(self.preview.deflicker.exposure, dtype=np.float64))
            self._slice = (key, to_display(s.picture(), self._slice_scale), to_display(s.picture(stops), self._slice_scale))
        return self._slice[1], self._slice[2]

    @property
    def slice_mode(self) -> str:
        return (self.project.settings.get("slice_mode") if self.project else None) or "left-right"

    def set_slice_mode(self, mode: str) -> None:
        from ...timeslice import MODES
        if mode in MODES and self.project is not None:
            self.project.settings["slice_mode"] = mode
            self.project.save()
            self.changed()

    def set_window(self, window: int) -> None:
        self.project.settings["window"] = int(window)
        self.request_preview()

    def set_white_balance(self, on: bool) -> None:
        self.project.settings["white_balance"] = bool(on)
        self.request_preview()

    def request_preview(self) -> None:
        """Re-plan with the current settings. While one preview is running, only the latest request waits."""
        if self.measured is None or self.opened is None or self.opened.editor is None:
            return
        self._preview_wanted = (self.window, self.white_balance)
        if not self.previewing:
            self._start_preview()

    def _start_preview(self) -> None:
        window, white_balance = self._preview_wanted
        self._preview_wanted = None
        self.previewing = True
        editor, seq, measured = self.opened.editor, self.opened.seq, self.measured
        self.runner.start("Preview", lambda t: self.s.preview(editor, seq, measured, window=window,
                                                                white_balance=white_balance),
                          lambda e: None if isinstance(e, Progress) else self.post(lambda: self._previewed(e)))

    def _previewed(self, event) -> None:
        self.previewing = False
        if isinstance(event, Done):
            self.preview = event.result
            self._view_gains()                 # the viewer's After follows the smoothness slider straight away
        elif isinstance(event, Failed):
            self._notice(event.message, "error")
        if self._preview_wanted is not None:
            self._start_preview()
        self.changed()

    @property
    def prepare_series(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(measured, smoothing curve) for the timeline, once a preview exists."""
        if self.preview is None or self.preview.deflicker is None:
            return None
        return self.preview.deflicker.brightness, self.preview.deflicker.target

    @property
    def exposure_steps(self) -> tuple[int, ...]:
        if self.preview is None or self.preview.deflicker is None:
            return ()
        return tuple(int(c.frame) for c in self.preview.deflicker.changes)

    @property
    def prepare_summary(self) -> str:
        series = self.prepare_series
        if series is None:
            return ""
        measured, target = series
        biggest = float(np.nanmax(np.abs(self.preview.deflicker.exposure)))
        return (f"Flicker: {_wobble(measured):.3f} stops now, {_wobble(target):.3f} after.\n"
                f"Largest correction: {biggest:.2f} stops.")

    @property
    def needs_metadata_saved(self) -> bool:
        return self.opened is not None and self.opened.editor is not None and self.opened.editor.name == "lightroom"

    def confirm_metadata_saved(self, yes: bool) -> None:
        self.metadata_saved = bool(yes)
        self.changed()

    @property
    def can_apply(self) -> bool:
        return (self.job is None and self.preview is not None and not self.previewing
                and (self.metadata_saved or not self.needs_metadata_saved))

    def apply(self) -> None:
        if not self.can_apply:
            return
        editor, seq, measured = self.opened.editor, self.opened.seq, self.measured
        window, wb = self.window, self.white_balance
        self._run("Applying corrections",
                  lambda t: self.s.apply(editor, seq, measured, window=window, white_balance=wb, tracker=t),
                  self._applied)

    def _applied(self, report) -> None:
        self.project.reach("prepared")
        self.project.save()
        self.export_since = self.clock()
        created = next((n for n in report.notes if n.startswith("Created edit files")), "")
        self._notice(f"Corrections applied to {report.changed} frames." + (f" {created}" if created else ""))
        self.screen = "grade"

    def undo(self) -> None:
        if self.opened is None or self.opened.editor is None:
            return
        editor, seq = self.opened.editor, self.opened.seq
        self._run("Undoing", lambda t: self.s.undo(editor, seq), self._undone)

    def _undone(self, count) -> None:
        self.project.start_over()
        self.project.save()
        self._notice(f"Took the corrections out of {count} frames: the edit files are back as they were.")
        self.screen = "folder"

    # --- grade: waiting for the export ------------------------------------------------------------

    @property
    def handoff_steps(self) -> list[str]:
        if self.opened is not None and self.opened.editor is not None:
            return self.opened.editor.handoff_steps()
        return ["Grade the sequence in your editor.", "Export every frame as JPEG into a new, empty folder."]

    def poll_export(self) -> None:
        """Called every second or so by the screen while it's waiting."""
        if self.opened is None or self.export_since is None:
            return
        status = self.s.export_status(self.opened.seq.folder, len(self.opened.seq), self.export_since)
        if status != self.export:
            self.export = status
            self.changed()

    def use_export(self, folder: str | Path | None = None) -> None:
        folder = Path(folder) if folder else (self.export.folder if self.export else None)
        if folder is None or not Path(folder).is_dir():
            return
        self.project.export = str(folder)
        self.project.reach("exported")
        self.project.save()
        self.planned = self.finished = None
        self.export_view, self.export_cursor = None, 0
        self.screen = "finish"
        self._default_video_path()
        self.changed()

    # --- finish: the export window -------------------------------------------------------------------

    def _default_video_path(self) -> None:
        if self.opened is None:
            return
        saved = (self.project.settings.get("export") if self.project else None) or {}
        if saved:
            self.out = ExportSettings.from_dict(saved)
        videos = Path.home() / "Videos"
        base = videos if videos.is_dir() else Path.home()
        name = self.opened.seq.folder.name
        if not self.out.video_path:
            self.out.video_path = str(base / (name + self.out.suffix))
        if not self.out.frames_folder and self.project and self.project.export:
            source = Path(self.project.export)
            self.out.frames_folder = str(source.parent / (source.name + "-finished"))

    def _changed_export(self, custom: bool = True) -> None:
        self.out_version += 1
        self._check_cache = None
        self.out.preset = matching_preset(self.out)          # worked out, never just flagged
        if self.out.video_path:
            self.out.video_path = str(Path(self.out.video_path).with_suffix(self.out.suffix))
        if self.project is not None:
            self.project.settings["export"] = self.out.to_dict()
        self.changed()

    def set_export(self, **values) -> None:
        """Change options (codec, quality, resolution, fps, output, trim, ...). The preset shown is whichever one
        the encoding settings match; output, frame rate, trim and locations never change it."""
        for k, v in values.items():
            if not hasattr(self.out, k):
                raise AttributeError(k)
            setattr(self.out, k, v)
        self._changed_export()

    def choose_preset(self, preset: str) -> None:
        try:
            self.out = apply_preset(self.out, preset)
        except ValueError as e:
            self._notice(str(e), "error")
        self._changed_export(custom=False)

    def save_preset(self, name: str) -> None:
        try:
            save_user_preset(name, self.out)
            self._notice(f"Saved the preset {name.strip()}.")
        except ValueError as e:
            self._notice(str(e), "error")
        self._changed_export(custom=False)

    def delete_preset(self, name: str) -> None:
        delete_user_preset(name)
        self._changed_export(custom=False)

    @property
    def presets(self) -> list[tuple[str, str]]:
        """(key, label) for every preset: built-in first, then your own."""
        return [(k, v["label"]) for k, v in BUILT_IN.items()] + [(k, k) for k in sorted(user_presets())]

    @property
    def preset(self) -> str:
        return matching_preset(self.out)

    # --- the visual trim: thumbnails made quietly in the background -------------------------------------

    def request_trim_thumbs(self) -> None:
        """A strip of thumbnails across the export, and the chosen first and last frames, made in the background."""
        if self.project is None or not self.project.export or not Path(self.project.export).is_dir():
            return
        from ...app.thumbs import thumbnail
        from ...finish import finished_frames
        export = Path(self.project.export)
        first, last = self.out.first, self.out.last

        def work(t):
            frames = finished_frames(export)
            n = len(frames)
            picks = sorted({round(i * (n - 1) / (STRIP - 1)) for i in range(STRIP)})
            strip = [(i, thumbnail(frames[i])) for i in picks]
            a, b = (first or 1) - 1, (last or n) - 1
            ends = (thumbnail(frames[min(max(a, 0), n - 1)]), thumbnail(frames[min(max(b, 0), n - 1)]))
            return n, strip, ends

        def done(event):
            if isinstance(event, Done):
                self.trim_frames, self.trim_strip, self.trim_ends = event.result
                self.trim_version += 1
                self.changed()
        self.runner.start("Thumbnails", work, lambda e: None if isinstance(e, Progress) else self.post(lambda: done(e)))

    def set_trim(self, first: int | None, last: int | None) -> None:
        n = self.trim_frames
        self.set_export(first=None if not first or first <= 1 else int(first),
                        last=None if not last or (n and last >= n) else int(last))
        self.request_trim_thumbs()

    def frame_time(self, frame: int) -> str:
        """Where a frame lands in the video, as minutes:seconds."""
        seconds = (frame - 1) / max(self.out.fps, 1)
        return f"{int(seconds // 60)}:{seconds % 60:04.1f}"

    @property
    def video_path(self) -> Path | None:
        return Path(self.out.video_path) if self.out.video_path else None

    @property
    def fps(self) -> float:
        return self.out.fps

    @property
    def also_frames(self) -> bool:
        return self.out.output == "both"

    def set_video_path(self, path: str | Path) -> None:
        self.out.video_path = str(path)
        self._changed_export(custom=False)

    def set_frames_folder(self, path: str | Path) -> None:
        self.out.frames_folder = str(path)
        self._changed_export(custom=False)

    def set_fps(self, fps: float) -> None:
        self.set_export(fps=float(fps))

    def set_also_frames(self, on: bool) -> None:
        self.set_export(output="both" if on else "video")

    def _source(self) -> tuple[int, int, int, float]:
        """Width, height, number of frames and average file size of the export, for estimates."""
        from PIL import Image
        from ...finish import finished_frames
        frames = finished_frames(Path(self.project.export))
        with Image.open(frames[0]) as im:
            w, h = im.size
        sample = frames[:: max(1, len(frames) // 12)]
        return w, h, len(frames), sum(f.stat().st_size for f in sample) / len(sample)

    @property
    def export_check(self) -> dict:
        """Everything the export window shows before anything is written: problems, notes, estimates.
        Worked out once per change of settings (it reads a few files, which is slow on some drives)."""
        key = (self.out_version, self.project.export if self.project else None)
        if self._check_cache is None or self._check_cache[0] != key:
            self._check_cache = (key, self._work_out_check())
        return self._check_cache[1]

    def _work_out_check(self) -> dict:
        if self.project is None or not self.project.export or not Path(self.project.export).is_dir():
            return {"problems": ["Choose the folder of exported frames."], "notes": [], "estimate": ""}
        try:
            w, h, n, average = self._source()
        except (OSError, ValueError, IndexError) as e:
            return {"problems": [f"Couldn't read the exported frames ({e})."], "notes": [], "estimate": ""}
        s = self.out
        first, last = s.first or 1, s.last or n
        count = max(0, last - first + 1)
        out_w, out_h = output_size(s, w, h)
        size = estimate_bytes(s, w, h, count, average)
        seconds = estimate_seconds(s, count)
        amount = describe(size)
        bits = ["Under 1 MB" if amount.startswith("under") else f"About {amount}"]
        bits.append(f"Takes about {int(seconds // 60)} min {int(seconds % 60)} s" if seconds else
                    "Time shown after the first export of this kind")
        where = s.video_path if s.video else s.frames_folder
        free = free_bytes(where)
        problems_ = problems(s, n, Path(self.project.export))
        if free is not None:
            bits.append(f"{describe(free)} free")
            if size > free:
                problems_.append(f"There isn't enough free space: about {describe(size)} needed, "
                                 f"{describe(free)} free.")
        if s.video:
            bits.insert(0, f"{count} frames at {out_w} × {out_h}")
        return {"problems": problems_, "notes": notes(s, out_w if s.video else None, out_h if s.video else None),
                "estimate": ". ".join(bits) + "."}

    @staticmethod
    def _preset_width(export: Path, limit) -> int | None:
        """The video width that fits the frames in the chosen box (or under a plain width): never enlarged."""
        if limit is None:
            return None
        box = limit if isinstance(limit, tuple) else (limit, None)
        from PIL import Image
        from ...finish import finished_frames
        from .export import fit_width
        try:
            with Image.open(finished_frames(export)[0]) as im:
                return fit_width(im.size[0], im.size[1], box)
        except (OSError, ValueError, IndexError):
            return box[0]

    @property
    def can_finish(self) -> bool:
        return self.job is None and not self.export_check["problems"]

    def _finish_work(self, frames_from):
        """The finishing job with the current export settings: ``frames_from(tracker)`` gives the folder of
        graded frames (the found export, or a render made in the same job), then it's measured, corrected and
        written. Shared by Finish and Export and finish, so both write films identically."""
        s = self.out
        video = Path(s.video_path) if s.video else None
        frames_out = Path(s.frames_folder) if s.frames else None
        options, window, mode = s.options(), self.window, self.slice_mode
        fmt = s.frames_format if s.frames else None

        def work(t):
            export = Path(frames_from(t))
            width = self._preset_width(export, s.width_limit()) if s.video else None
            planned = self.s.plan_export(export, window=window, tracker=t)
            done = self.s.write_export(planned, out=frames_out, video=video, fps=s.fps, width=width, slice_mode=mode,
                                       codec=s.codec, dither=s.dither, fmt=fmt, options=options,
                                       jpeg_quality=s.jpeg_quality, first=s.first, last=s.last,
                                       settings={"window": window, "preset": s.preset}, tracker=t)
            return export, planned, done
        return work

    def start_finish(self) -> None:
        if not self.can_finish:
            return
        export = Path(self.project.export)
        if self._is_render(export) and self.rapidraw_program is not None:
            # Frames this app rendered: render again first, in the format these settings need, only frames whose
            # grade changed (none, if nothing did), then finish.
            seq, program, deep = self.opened.seq, self.rapidraw_program, self.renders_deep(self.out)

            def done(result):
                rendered, planned, finished = result
                self.project.export = str(rendered)
                self.project.save()
                self._finished((planned, finished))
            self._run("Exporting and finishing",
                      self._finish_work(lambda t: self.s.render_with_rapidraw(seq, deep, tracker=t, program=program)),
                      done)
            return
        self._run("Finishing", self._finish_work(lambda t: export), lambda r: self._finished(r[1:]))

    def _is_render(self, folder: Path) -> bool:
        """One of this app's render folders for the open sequence (either format)."""
        if self.opened is None:
            return False
        try:
            return Path(folder).resolve() in {self.s.render_folder_for(self.opened.seq.folder, d).resolve()
                                              for d in (False, True)}
        except OSError:
            return False

    def close_folder(self) -> None:
        """Back to the start screen, to open another folder."""
        if self.job is not None:
            return
        self._reset_project_state()
        self.opened, self.project = None, None
        self.screen = "start"
        self.changed()

    # --- Export and finish: render through RapidRAW, then finish, in one run -----------------------------------

    @staticmethod
    def renders_deep(s: ExportSettings) -> bool:
        """16-bit TIFF renders for films and frames that keep more than 8 bits; JPEG for everything else."""
        return (s.video and (s.codec == "prores" or (s.options().depth or 8) > 8)) or (
            s.frames and s.frames_format in ("tif", "tiff"))

    def render_plan(self, preset: str) -> dict:
        """What Export and finish with ``preset`` would render: format words and the space still needed."""
        if not self.keyframes_supported:
            return {}
        try:
            deep = self.renders_deep(apply_preset(ExportSettings(), preset))
        except ValueError:
            return {}
        need = self.s.render_needs(self.opened.seq, deep)
        return {"deep": deep, "need": need,
                "words": ("16-bit frames" if deep else "JPEG frames")
                         + (f", about {need / 1e9:.1f} GB to render" if need else ", already rendered")}

    def export_and_finish(self, preset: str = "share", finish: bool = True) -> None:
        """Render every frame through RapidRAW (only frames whose edit changed since the last render), then
        measure, correct and write the film with ``preset``, in one run. ``finish=False`` renders, then opens
        Finish with every option instead."""
        import shutil

        from .export import problems as export_problems
        if self.rapidraw_program is None or self.opened is None:
            return
        self.load_keyframes()
        stale = self.keyframes_out_of_date
        if stale and not getattr(self, "_export_warned", False):
            self._export_warned = True
            self._notice(stale + " Blend again, or press again to export them as they are.", "error")
            self.changed()
            return
        seq, program = self.opened.seq, self.rapidraw_program
        self.choose_preset(preset)
        deep = self.renders_deep(self.out)
        folder = self.s.render_folder_for(seq.folder, deep)
        videos = Path.home() / "Videos"
        if not self.out.video_path:
            self.out.video_path = str((videos if videos.is_dir() else Path.home()) / (seq.folder.name + self.out.suffix))
        if not self.out.frames_folder:
            self.out.frames_folder = str(folder.parent / (folder.name + "-finished"))
        need = self.s.render_needs(seq, deep)
        try:
            free = shutil.disk_usage(folder.parent).free
        except OSError:
            free = need
        if need > free * 0.95:
            self._notice(f"Rendering needs about {need / 1e9:.1f} GB, and the drive has {free / 1e9:.1f} GB free. "
                         "Free some space, or choose Share (JPEG frames, much smaller).", "error")
            self.changed()
            return
        found = export_problems(self.out, len(seq), None) if finish else []
        if found:
            self._notice(found[0], "error")
            self.changed()
            return

        def render(t):
            return self.s.render_with_rapidraw(seq, deep, tracker=t, program=program)

        def kept(export) -> None:
            self.project.export = str(export)
            self.project.reach("exported")
            self.project.save()

        if not finish:
            if self.s.render_needs(seq, deep) == 0:          # already rendered: straight to the options
                kept(folder)
                self.use_export(folder)
                return

            def rendered(export):
                kept(export)
                self.use_export(export)
            self._notice("Rendering the frames first: the options work from them. Only frames you've changed are "
                         "ever rendered again.")
            self._run("Rendering with RapidRAW", render, rendered)
            return

        def done(result):
            export, planned, finished = result
            kept(export)
            self._finished((planned, finished))
        self._run("Exporting and finishing", self._finish_work(render), done)

    def _finished(self, result) -> None:
        self.planned, self.finished = result
        self.export_view, self.export_cursor = None, 0          # a new run: the Review viewer loads it afresh
        t = self.finished.timings
        if t.get("elapsed") and t.get("frames"):
            remember_speed(self.out, t["elapsed"] / t["frames"])     # for the next estimate of this kind
        self._check_cache = None
        self.project.video = str(self.finished.video) if self.finished.video else None
        self.project.reach("finished")
        self.project.save()
        config.remember_project(self.opened.seq.folder, len(self.opened.seq), self.project.stage)
        self.screen = "done"

    # --- done: what came out -------------------------------------------------------------------------

    @property
    def result_lines(self) -> list[str]:
        if self.finished is None:
            return []
        p = self.planned.plan
        lines = [f"Brightness wobble {p.brightness_before:.3f} stops before, {self.finished.brightness_after:.3f} after.",
                 f"Colour wobble {p.colour_before:.3f} before, {self.finished.colour_after:.3f} after."]
        if p.tone is not None:
            lines.append(p.tone.describe())
        if self.planned.moving_camera:
            lines.append(self.planned.moving_camera)
        if self.finished.where_the_time_went:
            lines.append(self.finished.where_the_time_went)
        return lines

    @property
    def worth_a_look(self) -> list[tuple[str, str]]:
        """(frame name, what's odd about it), one line per frame, without ones marked fine."""
        if self.finished is None:
            return []
        out: dict[str, list[str]] = {}
        for f in self.finished.findings:
            if f.name not in self.project.dismissed:
                out.setdefault(f.name, []).append(f.note)
        return [(name, "; ".join(notes)) for name, notes in out.items()]

    def looks_fine(self, name: str) -> None:
        if self.project is not None and name not in self.project.dismissed:
            self.project.dismissed.append(name)
            self.project.save()
            self.changed()

    # --- editor choice, for RAW folders with no edit files yet ---------------------------------------------

    def choose_editor(self, name: str) -> None:
        """Remember the editor for this folder and open it again with it."""
        if self.opened is None:
            return
        folder = self.opened.seq.folder
        if name == "lightroom":
            self._notice("Import the folder into Lightroom, select every frame, choose Metadata > Save Metadata to "
                         "Files, then open the folder here again.")
            self.changed()
            return
        self.s.choose_editor(folder, name)
        self.open_folder(folder)

    # --- keyframes (RapidRAW): suggest, adjust, label, blend -------------------------------------------------

    @property
    def keyframes_supported(self) -> bool:
        return self.opened is not None and self.opened.editor is not None and self.opened.editor.name == "rapidraw"

    def load_keyframes(self) -> None:
        """Pick up the keyframes labelled in RapidRAW, and whether the frames between them still match the blend
        (quick: reads the edit files). A list you're still choosing is left alone."""
        if not self.keyframes_supported or self.keyframes_state == "chosen":
            return
        found = self.s.keyframes_now(self.opened.editor, self.opened.seq)
        if not found:
            return
        reasons = {k["frame"]: k["reason"] for k in (self.keyframes or [])}
        self.keyframes = [{"frame": i, "reason": reasons.get(i, "labelled in RapidRAW")} for i in found]
        self._check_blend()

    def _check_blend(self) -> None:
        """Blended only if every frame has an edit file and matches the blend. Frames without one simply haven't
        been blended yet (not a change since the last blend, so no warning)."""
        from ...editors.rapidraw.sidecar import read_sidecar
        seq = self.opened.seq
        off = self.s.frames_off_the_blend(self.opened.editor, seq) if len(self.keyframes or []) >= 2 else []
        keys = {k["frame"] for k in self.keyframes or []}
        unblended = any(read_sidecar(f.path) is None for i, f in enumerate(seq) if i not in keys)
        self.keyframes_off = [] if unblended else off
        self.keyframes_state = "labelled" if (off or unblended or len(keys) < 2) else "blended"

    @property
    def keyframes_out_of_date(self) -> str:
        """Plain words for frames that no longer match the blend (keyframes edited since), or ""."""
        off = getattr(self, "keyframes_off", [])
        if not off or self.keyframes_state != "labelled":
            return ""
        settings = sorted({name for _, names in off for name in names if name != self.s.NOT_BLENDED})
        if not settings:
            return ""                                    # simply not blended yet: not out of date
        off = [o for o in off if o[1] != [self.s.NOT_BLENDED]]
        words = ", ".join(re.sub(r"(?<!^)([A-Z])", r" \1", s).lower() for s in settings[:4])
        more = f" and {len(settings) - 4} more" if len(settings) > 4 else ""
        return f"{len(off)} frames don't match your keyframes ({words}{more})."

    def suggest_keyframes(self) -> None:
        if not self.keyframes_supported:
            return
        seq, measured = self.opened.seq, self.measured

        def work(t):
            m = measured if measured is not None else self.s.measure(seq, t)
            return m, self.s.suggest_keyframes(seq, m, t)

        def done(result):
            m, picks = result
            self.measured = m
            self.keyframes = [{"frame": p.frame, "reason": p.reason} for p in picks]
            self.keyframes_state = "chosen"
            self._notice(f"Suggested {len(picks)} keyframes. Add or remove any, then label them.")
        self._run("Suggesting keyframes", work, done)

    def add_keyframe(self, frame: int) -> None:
        if self.opened is None or not 0 <= frame < len(self.opened.seq):
            return
        keys = self.keyframes or []
        if all(k["frame"] != frame for k in keys):
            self.keyframes = sorted(keys + [{"frame": frame, "reason": "added by you"}], key=lambda k: k["frame"])
            self.keyframes_state = "chosen"
        self.changed()

    def remove_keyframe(self, frame: int) -> None:
        self.keyframes = [k for k in (self.keyframes or []) if k["frame"] != frame]
        self.keyframes_state = "chosen"
        self.changed()

    def label_keyframes(self) -> None:
        keys = [k["frame"] for k in (self.keyframes or [])]
        if len(keys) < 2:
            self._notice("Choose at least two keyframes (the first and last frames, say).", "error")
            self.changed()
            return
        editor, seq = self.opened.editor, self.opened.seq

        def done(result):
            self.keyframes_state = "labelled"
            self._notice(f"Labelled {len(keys)} keyframes purple. Grade them in RapidRAW, close it, then blend.")
        self._run("Labelling keyframes", lambda t: self.s.set_keyframes(editor, seq, keys), done)

    def blend_keyframes(self) -> None:
        editor, seq = self.opened.editor, self.opened.seq

        def done(result):
            self.keyframes_state = "blended"
            self.keyframes_off, self._export_warned = [], False
            held = ""
            if result.held:
                names = sorted(result.held)
                held = (" These change at each keyframe instead: " + ", ".join(names[:6])
                        + (f" and {len(names) - 6} more" if len(names) > 6 else "") + ".")
            self._notice(f"Blended {len(result.keyframes)} keyframes across {len(seq)} frames.{held}")
        self._run("Blending", lambda t: self.s.blend_keyframes(editor, seq), done)

    def keyframe_thumb(self, frame: int):
        """A small picture of the frame, from its preview (None if there isn't one yet)."""
        from ... import previews
        from ...app.thumbs import thumbnail
        try:
            source = previews._file(self.opened.seq[frame].path)
            return thumbnail(source, 160) if source is not None and source.exists() else None
        except (OSError, IndexError, ValueError):
            return None

    # --- exporting through RapidRAW's command line ------------------------------------------------------------

    @property
    def rapidraw_program(self):
        """RapidRAW's program, if this is a RapidRAW folder and it's installed (looked up once)."""
        if not self.keyframes_supported:
            return None
        if not hasattr(self, "_rapidraw_program"):
            self._rapidraw_program = self.s.find_rapidraw()
        return self._rapidraw_program

    def export_with_rapidraw(self, replace: bool = False) -> None:
        """Render every frame with its grade through RapidRAW, then go straight on to Finish. ``replace``: into
        the export already found, its old frames removed first (after blending again, say)."""
        if self.rapidraw_program is None:
            return
        self.load_keyframes()
        stale = self.keyframes_out_of_date
        if stale and not getattr(self, "_export_warned", False):
            self._export_warned = True
            self._notice(stale + " Blend again, or press Export again to export them as they are.", "error")
            self.changed()
            return
        seq, program = self.opened.seq, self.rapidraw_program
        target = self.export.folder if replace and self.export is not None and self.export.folder else None

        def done(folder):
            self._notice(f"RapidRAW exported {len(seq)} frames into {folder}.")
            self.use_export(folder)
        self._run("Exporting with RapidRAW",
                  lambda t: self.s.export_with_rapidraw(seq, out=target, tracker=t, program=program,
                                                        replace=target is not None), done)

    # --- the sequence viewer: frames above the timeline -------------------------------------------------------

    def load_view(self) -> None:
        """The viewer's frames, from the previews, in the background: a tiny pass for one brightness scale, then
        each frame at the viewer's size."""
        if self.opened is None or self.view is not None or getattr(self, "_view_loading", False):
            return
        seq = self.opened.seq
        from ... import previews
        from .viewer import WIDTH, SequenceFrames, reduce

        def work(t):
            self.s.ensure_previews(seq, t)
            means = []
            hook = t.stage("Loading the viewer")
            for i, f in enumerate(seq):
                tiny = previews.load_small(f.path, 200)
                if tiny is None:
                    return None
                means.append(float(tiny.mean()))
                hook(i + 1, 2 * len(seq))
            scale = 0.18 / float(np.median([m for m in means if m > 0] or [1.0]))
            frames = []
            for i, f in enumerate(seq):
                frames.append(reduce(previews.load_small(f.path, WIDTH * 5 // 4), scale))
                hook(len(seq) + i + 1, 2 * len(seq))
            out = SequenceFrames(frames)
            out.scale = scale                     # kept for sharp single frames in full screen
            return out

        def event(e):
            if isinstance(e, Progress):
                self.view_status = f"Loading the viewer: {e.done * 100 // max(e.total, 1)}%"
            elif isinstance(e, Done):
                self._view_loading = False
                if e.result is None:
                    self.view_status = "The viewer needs the previews, which aren't all there yet."
                else:
                    self.view, self.view_status = e.result, ""
                    self._view_gains()
            elif isinstance(e, Failed):
                self._view_loading = False
                self.view_status = f"The viewer couldn't load: {e.message}"
            self.changed()
        self._view_loading = True
        self.view_status = "Loading the viewer..."
        self.runner.start("Viewer", work, lambda e: self.post(lambda: event(e)))

    def _view_gains(self) -> None:
        """Each frame's correction for the viewer's After: Prepare's (the preview of the corrections)."""
        if self.view is None:
            return
        series = self.prepare_series
        self.view.set_gains(None if series is None else np.asarray(series[1]) - np.asarray(series[0]))

    def set_cursor(self, frame: int) -> None:
        if self.opened is not None:
            self.cursor = int(min(max(frame, 0), len(self.opened.seq) - 1))

    def view_image(self, frame: int | None = None):
        """The picture the viewer shows for ``frame`` (default: the playhead), or None while loading."""
        if self.view is None:
            return None
        if self.view.gains.shape[0] and not self.view.gains.any() and self.prepare_series is not None:
            self._view_gains()
        return self.view.image(self.cursor if frame is None else frame, self.view_mode, self.wipe)

    def toggle_keyframe_here(self) -> None:
        keys = [k["frame"] for k in (self.keyframes or [])]
        (self.remove_keyframe if self.cursor in keys else self.add_keyframe)(self.cursor)

    # --- the Review screen's viewer: your exported frames, before and after Finish ----------------------------

    @property
    def finish_series(self):
        """(measured light, smoothing curve) of the export, in stops, for the Review timeline."""
        if self.planned is None:
            return None
        from ...finish import LUMA
        plan = self.planned.plan
        return plan.measured @ LUMA, plan.target @ LUMA

    def load_export_view(self) -> None:
        """The exported frames, small, in the background (graded already, so shown as they are; After adds
        Finish's per-channel, tone-aware correction)."""
        if self.planned is None or self.export_view is not None or getattr(self, "_export_view_loading", False):
            return
        frames, plan = list(self.planned.frames), self.planned.plan
        from .viewer import WIDTH, SequenceFrames

        def small(path):
            from PIL import Image
            with Image.open(path) as im:
                im.draft("RGB", (WIDTH * 5 // 4, WIDTH * 5 // 4))
                im = im.convert("RGB")
                if im.size[0] > WIDTH:
                    im = im.resize((WIDTH, max(1, round(im.size[1] * WIDTH / im.size[0]))), Image.BILINEAR)
                return np.ascontiguousarray(np.asarray(im))

        def work(t):
            hook = t.stage("Loading the viewer")
            out = []
            for i, f in enumerate(frames):
                out.append(small(f))
                hook(i + 1, len(frames))
            return SequenceFrames(out, gains=plan.gains, tone=plan.tone)

        def event(e):
            if isinstance(e, Progress):
                self.export_view_status = f"Loading the viewer: {e.done * 100 // max(e.total, 1)}%"
            elif isinstance(e, Done):
                self._export_view_loading = False
                self.export_view, self.export_view_status = e.result, ""
            elif isinstance(e, Failed):
                self._export_view_loading = False
                self.export_view_status = f"The viewer couldn't load: {e.message}"
            self.changed()
        self._export_view_loading = True
        self.export_view_status = "Loading the viewer..."
        self.runner.start("Export viewer", work, lambda e: self.post(lambda: event(e)))

    def set_export_cursor(self, frame: int) -> None:
        if self.planned is not None:
            self.export_cursor = int(min(max(frame, 0), len(self.planned.frames) - 1))

    def export_view_image(self, frame: int | None = None):
        if self.export_view is None:
            return None
        return self.export_view.image(self.export_cursor if frame is None else frame, self.view_mode, self.wipe)

    # --- sharp single frames, for full screen when not playing ----------------------------------------------

    SHARP = 1600

    def view_sharp(self, frame: int | None = None):
        """The frame at full preview size (1,600 px), as the viewer would show it; None when it can't be."""
        from ... import previews
        from .viewer import SequenceFrames, reduce
        if self.view is None or self.view_mode == "flicker":
            return None
        i = self.cursor if frame is None else frame
        linear = previews.load_small(self.opened.seq[i].path, self.SHARP)
        if linear is None:
            return None
        one = SequenceFrames([reduce(linear, getattr(self.view, "scale", 1.0), width=self.SHARP)],
                            gains=[self.view.gains[i]])
        return one.image(0, self.view_mode, self.wipe)

    def export_view_sharp(self, frame: int | None = None):
        if self.export_view is None or self.view_mode == "flicker" or self.planned is None:
            return None
        from PIL import Image

        from .viewer import SequenceFrames
        i = self.export_cursor if frame is None else frame
        try:
            with Image.open(self.planned.frames[i]) as im:
                im.draft("RGB", (self.SHARP, self.SHARP))
                im = im.convert("RGB")
                if im.size[0] > self.SHARP:
                    im = im.resize((self.SHARP, max(1, round(im.size[1] * self.SHARP / im.size[0]))), Image.BILINEAR)
                picture = np.ascontiguousarray(np.asarray(im))
        except OSError:
            return None
        one = SequenceFrames([picture], gains=[self.export_view.gains[i]], tone=self.export_view.tone)
        return one.image(0, self.view_mode, self.wipe)
