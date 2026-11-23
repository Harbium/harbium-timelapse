"""The screens: a shell (header, stage bar, messages, job progress) and one class per step.

Each screen is built once when it's shown and then refreshed in place, so a
slider being dragged is never rebuilt under the pointer. Everything shown comes
from ``AppModel``; every click calls one of its commands.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import flet as ft

from ...app.session import ExportStatus
from ..models import AppModel, Series, render_timeline
from ..models.export import (BUILT_IN, CHROMA_CHOICES, CODEC_CHOICES, DEPTH_CHOICES, FRAME_RATES, PRORES_CHOICES,
                             RESOLUTIONS, explain)
from ..models.timeline import Theme
from ..theme import BODY, DARK, DISPLAY, FONTS, LIGHT, MONO

FPS_CHOICES = ("23.976", "24", "25", "29.97", "30", "50", "60")


def open_path(path: Path, select: bool = False) -> None:
    """Open a file with its usual program, or show it in its folder."""
    path = Path(path)
    if sys.platform.startswith("win"):
        if select:
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))                      # noqa: S606 - opening the user's own file
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R" if select else "", str(path)] if select else ["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent if select else path)])


def _marked(picture, boxes, tokens) -> bytes:
    """The middle frame with the found areas outlined."""
    import io
    from PIL import Image, ImageDraw
    im = Image.fromarray(picture).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for top, left, bottom, right in boxes:
        d.rectangle([left * w, top * h, right * w - 1, bottom * h - 1], outline=(255, 255, 255), width=2)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _image_bytes(path) -> bytes:
    """A small image's data, so it shows the same in the desktop app and in a browser (which can't read local paths)."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return b""


class UI:
    """Small helpers so every screen uses the same brand styles."""

    def __init__(self, tokens: dict):
        self.t = tokens
        self.scale = 1.12                 # text size: a little larger than the design sizes, growing with the window

    def px(self, size: float) -> int:
        return int(round(size * self.scale))

    def title(self, text: str, size: int = 32) -> ft.Text:
        return ft.Text(text, font_family=DISPLAY, size=self.px(size), color=self.t["text"], weight=ft.FontWeight.W_500)

    def text(self, text: str, dim: bool = False, size: int = 15, mono: bool = False, selectable=False) -> ft.Text:
        return ft.Text(text, font_family=MONO if mono else BODY, size=self.px(size - (1 if mono else 0)),
                       color=self.t["dim" if dim else "text"], selectable=selectable)

    def primary(self, label: str, on_click, disabled: bool = False) -> ft.FilledButton:
        t = self.t
        return ft.FilledButton(label, on_click=on_click, disabled=disabled, style=ft.ButtonStyle(
            bgcolor={ft.ControlState.DEFAULT: t["primary_bg"], ft.ControlState.DISABLED: t["line"]},
            color={ft.ControlState.DEFAULT: t["primary_text"], ft.ControlState.DISABLED: t["dim"]},
            shape=ft.RoundedRectangleBorder(radius=2), padding=ft.Padding(18, 14, 18, 14),
            text_style=ft.TextStyle(size=self.px(14), font_family=BODY)))

    def secondary(self, label: str, on_click, disabled: bool = False) -> ft.OutlinedButton:
        t = self.t
        return ft.OutlinedButton(label, on_click=on_click, disabled=disabled, style=ft.ButtonStyle(
            color={ft.ControlState.DEFAULT: t["text"], ft.ControlState.DISABLED: t["dim"]},
            shape=ft.RoundedRectangleBorder(radius=2),
            side=ft.BorderSide(1, t["line"]), padding=ft.Padding(16, 14, 16, 14),
            text_style=ft.TextStyle(size=self.px(14), font_family=BODY)))

    def panel(self, content, padding: int = 16) -> ft.Container:
        return ft.Container(content, bgcolor=self.t["panel"], padding=padding, border_radius=2,
                            border=ft.Border.all(1, self.t["line"]))

    def mark(self, size: int = 28) -> ft.Container:
        """The H mark from harbium.studio's favicon."""
        return ft.Container(ft.Text("H", font_family=DISPLAY, size=size * 0.78, color=self.t["bg"]),
                            width=size, height=size, bgcolor=self.t["text"], alignment=ft.Alignment.CENTER)


# --- the screens ---------------------------------------------------------------------------------

class Screen:
    def columns(self, side: int = 360) -> tuple[int, int, bool]:
        """(side width, main width, side by side?): two columns when there's room for the main one to be at
        least 640 wide, otherwise one column using the full width."""
        width, side = self.view.content_width, self.ui.px(side)
        if width - side - 32 >= 640:
            return side, width - side - 32, True
        return width, width, False

    def __init__(self, view: "View"):
        self.view, self.m, self.ui = view, view.model, view.ui

    def build(self) -> ft.Control:
        raise NotImplementedError

    def refresh(self) -> None:
        """Update in place after the model changes. Default: nothing to update."""

    def leave(self) -> None:
        """Called when another screen takes over."""


class SlicePanel:
    """Before and after the prepared corrections, side by side, as time slices from the RAW previews."""
    WIDTH = 784                     # the same width as the timeline panel, so the page never wraps
    EACH = 372

    def __init__(self, screen: Screen, words: str, width: int | None = None):
        if width:
            self.WIDTH, self.EACH = int(width), max(120, (int(width) - 40) // 2)
        self.screen, self.m, self.ui = screen, screen.m, screen.ui
        self.before = ft.Image(src=b"", width=self.EACH, fit=ft.BoxFit.CONTAIN, visible=False)
        self.after = ft.Image(src=b"", width=self.EACH, fit=ft.BoxFit.CONTAIN, visible=False)
        self.status = self.ui.text("", dim=True, size=13)
        self.words = words
        self._shown = None
        ui = self.ui
        self.progress_text = ft.Text("", size=16, color=ui.t["text"], font_family=BODY)
        self.progress_bar = ft.ProgressBar(value=None, width=420, color=ui.t["text"], bgcolor=ui.t["line"])
        self.progress = ft.Container(
            ft.Column([self.progress_text, self.progress_bar,
                       ui.text("The time slices appear here when they're ready.", dim=True, size=13)],
                      spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            height=220, alignment=ft.Alignment.CENTER, visible=False)

    def control(self) -> ft.Control:
        ui = self.ui
        def side(label, image):
            return ft.Column([ui.text(label, size=13),
                              ft.Container(image, on_click=lambda e: SliceViewer(self.screen.view).open())], spacing=4)
        from ...timeslice import MODES
        chooser = ft.Dropdown(options=[ft.DropdownOption(key=k, text=t) for k, t in MODES.items()],
                              value=self.m.slice_mode, width=230, dense=True,
                              on_select=lambda e: self.m.set_slice_mode(e.control.value))
        return ft.Container(ui.panel(ft.Column(
            [ft.Row([ui.text("Time slice", dim=True, size=13), chooser], spacing=8),
             ui.text(self.words + " Click a picture to enlarge it.", dim=True, size=13), self.status,
             self.progress,
             ft.Container(ft.Row([side("Before", self.before), side("After", self.after)], spacing=8),
                          bgcolor=ui.t["surround"], padding=6, border_radius=2)], spacing=6), padding=12),
            width=self.WIDTH)

    def refresh(self) -> None:
        busy = self.m.preview_progress
        self.progress.visible = busy is not None
        self.status.visible = busy is None                    # the progress block says it while busy
        if busy is not None:
            done, total = busy
            self.progress_text.value = self.m.preview_status or "Loading previews..."
            self.progress_bar.value = (done / total) if total else None
        self.status.value = self.m.preview_status
        pictures = self.m.time_slice()
        if pictures is not None and pictures[0] is not self._shown:
            self._shown = pictures[0]
            self.screen.view.page.run_task(self.draw, pictures)

    async def draw(self, pictures):
        import io
        from PIL import Image

        def make(picture) -> bytes:
            img = Image.fromarray(picture)
            scale = (self.EACH * 2) / img.width                       # twice the shown size: sharp on high-DPI screens
            img = img.resize((self.EACH * 2, max(1, round(img.height * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
        self.before.src, self.after.src = await asyncio.gather(asyncio.to_thread(make, pictures[0]),
                                                               asyncio.to_thread(make, pictures[1]))
        self.before.visible = self.after.visible = True
        self.screen.view.page.update()


class SliceViewer:
    """The time slice filling the window: Before and After side by side (or one alone), zoomable, with the mode
    chooser and Save buttons. Esc or Close goes back."""
    current = None

    def __init__(self, view: "View"):
        self.view, self.m, self.ui = view, view.model, view.ui
        self.show = "both"
        self.saved: Path | None = None
        # The viewer covers the window's message banner, so it says what it saved in its own bar.
        self.note = ft.Text("", size=13, color=self.ui.t["text"])
        self.reveal = self.ui.secondary("Show in folder", lambda e: self.saved and open_path(self.saved, select=True))
        self.reveal.visible = False
        self.saved_row = ft.Container(ft.Row([self.note, self.reveal], spacing=10), bgcolor=self.ui.t["panel"],
                                      padding=ft.Padding(10, 4, 10, 4), border_radius=2, visible=False)

    def open(self) -> None:
        SliceViewer.current = self
        self.layer = ft.Container(expand=True, bgcolor=self.ui.t["surround"], padding=12)
        self.view.page.overlay.append(self.layer)
        self.view.page.run_task(self.draw)

    def close(self, e=None) -> None:
        if self.layer in self.view.page.overlay:
            self.view.page.overlay.remove(self.layer)
        SliceViewer.current = None
        self.view.page.update()

    def choose(self, show: str) -> None:
        self.show = show
        self.view.page.run_task(self.draw)

    def set_mode(self, mode: str) -> None:
        self.m.set_slice_mode(mode)
        self.view.page.run_task(self.draw)

    def save(self, which: str) -> None:
        from PIL import Image
        pictures = self.m.time_slice()
        folder = Path(self.m.opened.seq.folder)
        if pictures is None:
            return
        target = folder.parent / f"{folder.name} time slice ({self.m.slice_mode}) {which}.png"
        try:
            Image.fromarray(pictures[0 if which == "before" else 1]).save(target)
            self.m._notice(f"Saved {target}")
            self.saved, self.note.value, self.reveal.visible = target, f"Saved {target}", True
            self.saved_row.visible = True
        except OSError as e:
            self.m._notice(f"Couldn't save the picture: {e}", "error")
            self.note.value, self.reveal.visible = f"Couldn't save the picture: {e}", False
            self.saved_row.visible = True
        self.m.changed()
        self.view.page.update()

    async def draw(self):
        import io
        from PIL import Image
        from ...timeslice import MODES
        ui, pictures = self.ui, self.m.time_slice()
        if pictures is None:
            self.close()
            return

        def encode(picture) -> bytes:
            buf = io.BytesIO()
            Image.fromarray(picture).save(buf, format="JPEG", quality=92)
            return buf.getvalue()
        before, after = await asyncio.gather(asyncio.to_thread(encode, pictures[0]),
                                             asyncio.to_thread(encode, pictures[1]))

        page = self.view.page
        width = max(640, int(page.width or 1280)) - 48                  # the window, less the margins
        height = max(400, int(page.height or 800)) - 150                # less the bar and the hint
        each = width if self.show != "both" else (width - 12) // 2

        def big(label, data):
            return ft.Column([ft.Text(label, color="#ffffff", size=14),
                              ft.Container(ft.InteractiveViewer(content=ft.Image(src=data, width=each, height=height,
                                                                                 fit=ft.BoxFit.CONTAIN),
                                                                min_scale=1, max_scale=8,
                                                                boundary_margin=ft.Margin.all(60)),
                                           width=each, height=height, clip_behavior=ft.ClipBehavior.HARD_EDGE)],
                             spacing=6)
        shown = {"both": [big("Before", before), big("After", after)], "before": [big("Before", before)],
                 "after": [big("After", after)]}[self.show]
        pick = lambda label, key: (ui.primary if self.show == key else ui.secondary)(label, lambda e: self.choose(key))  # noqa: E731
        bar = ft.Container(ft.Row([
            ft.Row([ft.Dropdown(options=[ft.DropdownOption(key=k, text=t) for k, t in MODES.items()],
                                value=self.m.slice_mode, width=230, dense=True,
                                on_select=lambda e: self.set_mode(e.control.value)),
                    pick("Side by side", "both"), pick("Before only", "before"), pick("After only", "after")], spacing=8),
            ft.Row([ui.secondary("Save Before", lambda e: self.save("before")),
                    ui.secondary("Save After", lambda e: self.save("after")),
                    ui.primary("Close", self.close)], spacing=8)],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN), bgcolor=ui.t["panel"], padding=10, border_radius=2)
        self.saved_row.visible = self.saved is not None or bool(self.note.value)
        bar = ft.Column([bar, self.saved_row], spacing=6)
        self.layer.content = ft.Column([bar, ft.Row(shown, spacing=12),
                                        ft.Text("Scroll to zoom, drag to look around. Esc closes.", color="#e8e4db",
                                                size=12)], spacing=10)
        self.view.page.update()


class StartScreen(Screen):
    def build(self):
        ui = self.ui
        recent = [ft.Container(
            ft.Row([ft.Container(ui.text(f"{Path(r['folder']).parent.name} / {Path(r['folder']).name}"), expand=True),
                    ft.Container(ui.text(f"{r['frames']} frames", dim=True), width=120,          # fixed columns,
                                 alignment=ft.Alignment.CENTER_RIGHT),                         # so they line up
                    ft.Container(ui.text(r["stage"].capitalize(), dim=True), width=110,
                                 alignment=ft.Alignment.CENTER_RIGHT)]),
            padding=ft.Padding(4, 12, 4, 12), border=ft.Border(bottom=ft.BorderSide(1, ui.t["line"])),
            on_click=lambda e, f=r["folder"]: self.m.open_folder(f), ink=True) for r in self.m.recent]
        return ft.Column([
            ui.title("Open a timelapse"),
            ui.text("Choose a folder of RAW frames, or of exported frames to finish.", dim=True),
            ft.Container(height=8),
            ft.Row([ui.primary("Choose folder", self.choose)]),
            ft.Container(height=24),
            *([ui.text("Recent", size=14), *recent] if recent else []),
        ], spacing=10, width=760)

    async def choose(self, e):
        folder = await self.view.ask("get_directory_path", dialog_title="Choose a folder of timelapse frames")
        if folder:
            self.m.open_folder(folder)


class FolderScreen(Screen):
    def build(self):
        ui, m = self.ui, self.m
        facts = ft.Row([ft.Column([ui.text(k, dim=True, size=13), ui.text(v)], spacing=2) for k, v in m.facts],
                       spacing=36, wrap=True)
        cards = []
        for p in m.problems or [{"serious": False, "message": "Nothing to report: the folder looks ready.", "fix": ""}]:
            icon = ft.Container(ft.Text("!" if (p["serious"] or p["fix"]) and not p.get("note") else "✓", size=12,
                                        color=ui.t["bg"]),
                                width=20, height=20, border_radius=10, bgcolor=ui.t["text"],
                                alignment=ft.Alignment.CENTER)
            body = [ui.text(p["message"])] + ([ui.text(p["fix"], dim=True, size=14)] if p["fix"] else [])
            if p.get("choose_editor"):
                body.append(ft.Container(ft.Row([
                    ui.primary("Grade in RapidRAW", lambda e: m.choose_editor("rapidraw")),
                    ui.secondary("Grade in Lightroom", lambda e: m.choose_editor("lightroom"))], spacing=10),
                    padding=ft.Padding(0, 8, 0, 0)))
            if p.get("details"):
                listing = ft.Column([ui.text(d, dim=True, size=13) for d in p["details"]], spacing=2, visible=False)
                def toggle(e, box=listing):
                    box.visible = not box.visible
                    e.control.text = "Hide them" if box.visible else "Show them"
                    self.view.page.update()
                body += [ft.TextButton("Show them", on_click=toggle, style=ft.ButtonStyle(color=ui.t["dim"])), listing]
            cards.append(ft.Container(ft.Row([icon, ft.Column(body, spacing=2, expand=True)],
                                             vertical_alignment=ft.CrossAxisAlignment.START),
                                      padding=ft.Padding(0, 10, 0, 10),
                                      border=ft.Border(bottom=ft.BorderSide(1, ui.t["line"]))))
        target, label = m.next_step
        actions = [ui.primary(label, lambda e: m.go(target), disabled=not m.can_go(target))]
        resume = []
        if m.resume:
            where, words = m.resume
            resume = [ui.panel(ft.Row([ui.text(words), ui.secondary("Continue", lambda e: m.go(where))],
                                      alignment=ft.MainAxisAlignment.SPACE_BETWEEN))]
        return ft.Column([ui.title(Path(m.opened.seq.folder).name, 30), facts, *resume,
                          ui.text("Checks", size=14), *cards, ft.Row(actions, alignment=ft.MainAxisAlignment.END)],
                         spacing=14, width=900)


class PrepareScreen(Screen):
    WIDTH, HEIGHT = 760, 260

    def build(self):
        ui, m = self.ui, self.m
        side_w, main_w, two = self.columns(380)
        self.seq = SequenceView(self, keyframes=False, width=main_w - 26,
                                picture_height=self.view.content_height - self.ui.px(400))
        self.summary = ui.text("", dim=True, size=14)
        self.notes = ft.Column(spacing=4)
        self.apply_btn = ui.primary("Apply corrections", lambda e: m.apply())
        self.state = ui.text("", dim=True, size=14)
        self.slider = ft.Slider(min=7, max=61, divisions=27, value=m.window, label="{value} frames",
                                on_change=lambda e: m.set_window(int(e.control.value)), expand=True)
        side = [ui.title("Prepare"),
                ui.text("Steady the light in your RAW files before you grade.", dim=True, size=14)]
        if m.white_balance_available:
            side.append(ft.Switch(label="Steady the camera's auto white balance", value=m.white_balance,
                                  on_change=lambda e: m.set_white_balance(e.control.value)))
        side += [ui.text("Smoothness", size=14),
                 ft.Row([ui.text("Steady", dim=True, size=13), self.slider, ui.text("Follows\nthe light", dim=True,
                                                                                       size=13)], spacing=4)]
        if m.needs_metadata_saved:
            side.append(ft.Row([ft.Checkbox(value=m.metadata_saved,
                                            on_change=lambda e: m.confirm_metadata_saved(e.control.value)),
                                ft.Container(ui.text("In Lightroom, I selected every frame and chose Metadata > "
                                                     "Save Metadata to Files", size=14), expand=True)],
                               vertical_alignment=ft.CrossAxisAlignment.START))
        side += [self.summary, self.notes, self.state,
                 ft.Row([ui.secondary("Undo", lambda e: m.undo()), self.apply_btn], spacing=10)]
        self._drawn = None
        self.refresh()
        self.slice = SlicePanel(self, "Each frame gives one strip of the picture. After should change smoothly.",
                                width=main_w)
        self.slice.refresh()
        main = ft.Column([ui.panel(self.seq.control(), padding=12), self.slice.control()], spacing=16, width=main_w)
        side_col = ft.Column(side, spacing=12, width=side_w)
        if two:
            return ft.Row([side_col, main], spacing=32, vertical_alignment=ft.CrossAxisAlignment.START)
        return ft.Column([side_col, main], spacing=24)

    def refresh(self):
        m, ui = self.m, self.ui
        self.summary.value = m.prepare_summary
        self.notes.controls = [ui.text(n, dim=True, size=13) for n in (m.preview.notes if m.preview else [])]
        self.apply_btn.disabled = not m.can_apply
        self.state.value = ("Measuring your RAW files..." if m.measured is None
                            else "Updating..." if m.previewing else "")
        if getattr(self, "seq", None):
            self.seq.refresh()
        if hasattr(self, "slice"):
            self.slice.refresh()

    def on_key(self, e):
        self.seq.on_key(e)

    def leave(self):
        self.seq.playing = False


class RawSource:
    """What a viewer on Prepare or Grade shows: the RAW previews with Prepare's corrections."""

    def __init__(self, m):
        self.m = m

    frames = property(lambda self: self.m.view)
    cursor = property(lambda self: self.m.cursor)
    status = property(lambda self: self.m.view_status)
    series = property(lambda self: self.m.prepare_series if self.m.prepare_series is not None
                      else ((self.m.measured, None) if self.m.measured is not None else None))
    steps = property(lambda self: self.m.exposure_steps)
    names = property(lambda self: [f.name for f in self.m.opened.seq])

    def load(self):
        self.m.load_view()

    def set(self, frame):
        self.m.set_cursor(frame)

    def image(self):
        return self.m.view_image()

    def sharp(self):
        return self.m.view_sharp()


class ExportSource:
    """What the viewer on Review shows: your exported (graded) frames, before and after Finish."""

    def __init__(self, m):
        self.m = m

    frames = property(lambda self: self.m.export_view)
    cursor = property(lambda self: self.m.export_cursor)
    status = property(lambda self: self.m.export_view_status)
    series = property(lambda self: self.m.finish_series)
    steps = property(lambda self: ())
    names = property(lambda self: [f.name for f in self.m.planned.frames] if self.m.planned else [])

    def load(self):
        self.m.load_export_view()

    def set(self, frame):
        self.m.set_export_cursor(frame)

    def image(self):
        return self.m.export_view_image()

    def sharp(self):
        return self.m.export_view_sharp()


class FullView:
    """The viewer filling the window: the same sequence, playhead and mode, as large as the screen allows.
    Esc or Close goes back."""
    current = None

    def __init__(self, view: "View", keyframes: bool, source):
        self.view, self.keyframes, self.source = view, keyframes, source

    def open(self) -> None:
        FullView.current = self
        page, ui = self.view.page, self.view.ui
        width = max(480, int(page.width or 1400) - 48)
        tall = max(240, int(page.height or 900) - ui.px(250))           # room for the timeline and controls
        holder = type("Holder", (), {})()
        holder.m, holder.ui, holder.view = self.view.model, ui, self.view
        self.seq = SequenceView(holder, keyframes=self.keyframes, source=self.source, width=width,
                                picture_height=tall, full=True)
        self.layer = ft.Container(ft.Column([self.seq.control()], scroll=ft.ScrollMode.AUTO),
                                  expand=True, bgcolor=ui.t["bg"], padding=24)
        page.overlay.append(self.layer)
        self.seq.refresh()
        page.update()

    def close(self, e=None) -> None:
        self.seq.playing = False
        if self.layer in self.view.page.overlay:
            self.view.page.overlay.remove(self.layer)
        FullView.current = None
        if self.view.screen is not None:
            self.view.screen.refresh()                   # the page's own viewer catches up with the playhead
        self.view.page.update()


class SequenceView:
    """The viewer and timeline: the frame at the playhead above, the light over time below with keyframes as
    diamonds (solid: labelled; outlined: suggested), the playhead as a dotted line, and a scrub bar to drag
    through. Click the timeline to jump (a diamond selects that keyframe); space plays, the arrows step (Shift:
    ten), K adds or removes a keyframe at the playhead."""

    LINE_H = 110
    FPS = 24
    LOOP = 12                                           # frames either side of the playhead when looping
    MODES = (("after", "After"), ("before", "Before"), ("wipe", "Wipe"), ("flicker", "Flicker ×5"))

    def __init__(self, screen, keyframes: bool = True, source=None, width: int = 720,
                 picture_height: int | None = None, full: bool = False):
        self.screen, self.m, self.ui = screen, screen.m, screen.ui
        self.src = source or RawSource(self.m)
        self.with_keyframes = keyframes                 # Grade: keyframes on; Prepare and Review: not
        self.full = full                                # filling the window (FullView)
        self.WIDTH = int(width)
        w, h = self.src.frames.size if self.src.frames is not None else (3, 2)
        pw, ph = self.WIDTH, round(self.WIDTH * h / w)
        if picture_height and ph > picture_height:      # never taller than the room there is
            pw, ph = round(picture_height * w / h), int(picture_height)
        self.HEIGHT = ph
        self.picture = ft.RawImage(width=pw, height=ph, fit=ft.BoxFit.CONTAIN, filter_quality=ft.FilterQuality.HIGH)
        self.sharp_when_stopped = full or pw > 700        # stopped: redraw from the full-size preview
        self.line = ft.RawImage(width=self.WIDTH, height=self.LINE_H)
        self.line_box = ft.Container(self.line, width=self.WIDTH, height=self.LINE_H, on_tap_down=self._tapped)
        self.scrub = ft.Slider(min=0, max=1, value=0, width=self.WIDTH, on_change=self._scrubbing,
                               on_change_end=self._scrubbed, active_color=self.ui.t["text"])
        self.status = self.ui.text("", dim=True, size=13)
        self.label = self.ui.text("", size=14)
        self.controls = ft.Row(spacing=8, wrap=True)
        self.help = self.ui.text("", dim=True, size=13)
        self.playing = False
        self.loop = False
        self._shown = self._line_key = None

    def control(self):
        ui = self.ui
        return ft.Container(ft.Column([
            ft.Container(self.picture, bgcolor=ui.t["surround"], border_radius=2, width=self.WIDTH,
                         alignment=ft.Alignment.CENTER),
            self.status, self.line_box, self.scrub, self.label, self.controls, self.help], spacing=6),
            width=self.WIDTH)

    # --- drawing -----------------------------------------------------------------------------------------

    def _keys(self):
        if not self.with_keyframes:
            return (), ()
        keys = tuple(k["frame"] for k in (self.m.keyframes or []))
        return ((), keys) if self.m.keyframes_state == "chosen" else (keys, ())

    def line_image(self):
        from ..models.timeline import Theme
        t, series = self.ui.t, self.src.series
        if series is None:
            return None
        measured, curve = series
        lines = [Series(measured, t["measured"], 1)] + ([Series(curve, t["curve"], 3)] if curve is not None else [])
        solid, ghosts = self._keys()
        theme = self.screen.view.chart_theme if self.screen.view is not None else Theme()
        return render_timeline(lines, self.WIDTH, self.LINE_H, theme, steps=self.src.steps,
                               flags=solid, ghosts=ghosts, current=self.src.cursor)

    def _label(self) -> str:
        names, i = self.src.names, self.src.cursor
        if not names:
            return ""
        keys = [k["frame"] for k in (self.m.keyframes or [])] if self.with_keyframes else []
        return f"{names[i]}  ·  frame {i + 1} of {len(names)}" + ("  ·  keyframe" if i in keys else "")

    def refresh(self) -> None:
        m, ui, src = self.m, self.ui, self.src
        src.load()
        self.status.value = src.status
        self.status.visible = bool(src.status)
        n = len(src.names)
        self.scrub.max, self.scrub.divisions, self.scrub.value = max(1, n - 1), max(1, n - 1), src.cursor
        self.label.value = self._label()
        keys = [k["frame"] for k in (m.keyframes or [])] if self.with_keyframes else []

        def mode_button(mode, words):
            return (ui.primary if m.view_mode == mode else ui.secondary)(words, lambda e, md=mode: self._mode(md))
        self.controls.controls = [ui.primary("Pause" if self.playing else "Play", lambda e: self.play()),
                                  ui.secondary("Loop: on" if self.loop else "Loop: off", lambda e: self._loop()),
                                  *[mode_button(mode, words) for mode, words in self.MODES]]
        if self.with_keyframes:
            here = src.cursor in keys
            self.controls.controls.append(ui.secondary("Remove keyframe (K)" if here else "Add keyframe here (K)",
                                                       lambda e: self._toggle_key()))
        if m.view_mode == "wipe":
            self.controls.controls.append(ft.Slider(min=0, max=1, value=m.wipe, width=200,
                                                    on_change=self._wiped, active_color=ui.t["text"]))
        if self.full:
            self.controls.controls.append(ui.secondary("Close full screen (Esc)", lambda e: FullView.current.close()))
        else:
            self.controls.controls.append(ui.secondary("Full screen", lambda e: self._full_screen()))
        help_text = {"flicker": "Flicker ×5 magnifies any flicker left. Moving things smear.",
                     "wipe": "Before on the left, After on the right."}.get(m.view_mode, "")
        keys_text = "Space: play · ← →: step (Shift: 10)" + (" · K: keyframe" if self.with_keyframes else "")
        source_note = ("Your export, before and after Finish." if isinstance(src, ExportSource) else
                       "RAW preview with Prepare's corrections, without your grade.")
        self.help.value = "   ".join(x for x in (help_text, source_note, keys_text) if x)
        self._draw()

    def _draw(self, picture_only: bool = False) -> None:
        if self.screen.view is None:
            return
        m, src = self.m, self.src
        key = (src.cursor, m.view_mode, round(m.wipe, 3), id(src.frames), id(src.series))
        line_key = (src.cursor, self._keys(), id(src.series))

        async def draw():
            if key != self._shown:
                image = await asyncio.to_thread(src.image)
                if image is not None:
                    self._shown = key
                    await self.picture.render(image)
                if self.sharp_when_stopped and not self.playing and m.view_mode != "flicker":
                    sharp = await asyncio.to_thread(src.sharp)       # stopped: the frame at full preview size
                    if sharp is not None and key == self._shown and not self.playing:
                        await self.picture.render(sharp)
            if not picture_only and line_key != self._line_key:
                image = await asyncio.to_thread(self.line_image)
                if image is not None:
                    self._line_key = line_key
                    await self.line.render(image)
        self.screen.view.page.run_task(draw)

    # --- acting ------------------------------------------------------------------------------------------

    def go(self, frame: int) -> None:
        self.src.set(frame)
        self.refresh()
        if self.screen.view is not None:
            self.screen.view.page.update()

    def _scrubbing(self, e) -> None:                    # dragging: the picture and label follow, the curve after
        self.playing = False
        self.src.set(int(round(float(e.control.value))))
        self.label.value = self._label()
        self._draw(picture_only=True)
        if self.screen.view is not None:
            self.screen.view.page.update()

    def _scrubbed(self, e) -> None:
        self.go(int(round(float(e.control.value))))

    def _tapped(self, e) -> None:
        from ..models.timeline import PAD, TOP, frame_at
        n = len(self.src.names)
        frame = frame_at(e.local_position.x, self.WIDTH, n)
        if self.with_keyframes and e.local_position.y <= TOP + 4:   # on the diamonds' row: the nearest keyframe
            keys = [k["frame"] for k in (self.m.keyframes or [])]
            step = (self.WIDTH - 2 * PAD) / max(n - 1, 1)
            near = [k for k in keys if abs(k - frame) * step <= 8]
            if near:
                frame = min(near, key=lambda k: abs(k - frame))
        self.go(frame)

    def _full_screen(self) -> None:
        if self.screen.view is None:
            return
        self.playing = False
        FullView(self.screen.view, self.with_keyframes, self.src).open()

    def _mode(self, mode: str) -> None:
        self.m.view_mode = mode
        self.go(self.src.cursor)

    def _wiped(self, e) -> None:
        self.m.wipe = float(e.control.value)
        self._draw(picture_only=True)

    def _loop(self) -> None:
        self.loop = not self.loop
        self.go(self.src.cursor)

    def _toggle_key(self) -> None:
        self.m.toggle_keyframe_here()
        self.go(self.src.cursor)

    def on_key(self, e) -> None:
        key = (e.key or "").lower()
        if key in (" ", "space"):
            self.play()
        elif key in ("arrow left", "arrow right"):
            self.playing = False
            self.go(self.src.cursor + (1 if key == "arrow right" else -1) * (10 if e.shift else 1))
        elif key == "k" and self.with_keyframes:
            self._toggle_key()

    def play(self) -> None:
        if self.playing:
            self.playing = False
            self.refresh()
            return
        if self.src.frames is None or self.screen.view is None:
            return
        self.playing = True
        start, n = self.src.cursor, len(self.src.names)
        first, last = (max(0, start - self.LOOP), min(n - 1, start + self.LOOP)) if self.loop else (0, n - 1)
        if not self.loop and start >= n - 1:
            self.src.set(0)
        self.refresh()

        async def run():
            import time as _time
            tick = 1.0 / self.FPS
            while self.playing:
                began = _time.perf_counter()
                nxt = self.src.cursor + 1
                if nxt > last:
                    if not self.loop:
                        self.playing = False
                        break
                    nxt = first
                self.src.set(nxt)
                image = await asyncio.to_thread(self.src.image)
                if image is not None:
                    await self.picture.render(image)
                self.label.value = self._label()
                self.scrub.value = nxt
                self.screen.view.page.update()
                await asyncio.sleep(max(0.0, tick - (_time.perf_counter() - began)))
            self.refresh()
            self.screen.view.page.update()
        self.screen.view.page.run_task(run)


class KeyframesPanel:
    """Keyframes on the Grade screen: the list (click one to jump to it) and the steps: suggest, label purple
    in RapidRAW, then blend. Picking frames happens in the viewer above."""

    def __init__(self, screen):
        self.screen, self.m, self.ui = screen, screen.m, screen.ui
        self.body = ft.Column(spacing=8)
        self.m.load_keyframes()                         # however you got here: your labelled keyframes, and
                                                        # whether the frames between them still match the blend

    def control(self):
        return ft.Container(self.ui.panel(self.body), padding=ft.Padding(0, 8, 0, 8))

    def _thumb(self, frame: int, width: int, height: int):
        path = self.m.keyframe_thumb(frame)
        if path is None:
            return ft.Container(width=width, height=height, bgcolor=self.ui.t["line"], border_radius=2)
        return ft.Image(src=_image_bytes(path), width=width, height=height, fit=ft.BoxFit.COVER, border_radius=2)

    def _go(self, frame: int) -> None:
        viewer = getattr(self.screen, "seq", None)
        if viewer is not None:
            viewer.go(frame)

    def refresh(self):
        ui, m = self.ui, self.m
        seq = m.opened.seq
        keys, state = m.keyframes or [], m.keyframes_state
        rows = [ui.text("Keyframes", size=17),
                ui.text("Grade a few frames. The app blends every setting between them.", dim=True, size=14)]
        if not keys:
            rows.append(ft.Row([ui.primary("Suggest keyframes", lambda e: m.suggest_keyframes())]))
            rows.append(ui.text("Or press K in the viewer to add one.", dim=True, size=13))
            self.body.controls = rows
            return
        for k in keys:
            rows.append(ft.Container(ft.Row([
                self._thumb(k["frame"], 72, 48),
                ft.Column([ui.text(f"{seq[k['frame']].name}  (frame {k['frame'] + 1})", size=14),
                           ui.text(k["reason"], dim=True, size=13)], spacing=0, expand=True),
                ft.TextButton("Remove", on_click=lambda e, f=k["frame"]: m.remove_keyframe(f),
                              style=ft.ButtonStyle(color=ui.t["dim"]))], spacing=10),
                on_click=lambda e, f=k["frame"]: self._go(f), ink=True, border_radius=2))
        if state == "chosen":
            rows.append(ft.Row([ui.primary("Label them purple in RapidRAW", lambda e: m.label_keyframes()),
                                ui.secondary("Suggest again", lambda e: m.suggest_keyframes())], spacing=10,
                               wrap=True))
        elif state == "labelled":
            stale = m.keyframes_out_of_date
            rows.append(ui.text(stale + " Blend again before exporting." if stale else
                                "Grade the purple frames in RapidRAW, close it, then blend.", size=14))
            rows.append(ft.Row([ui.primary("Blend between keyframes", lambda e: m.blend_keyframes())]))
        elif state == "blended":
            rows.append(ui.text("All frames are blended. If you change a keyframe, blend again.", size=14))
            rows.append(ft.Row([ui.secondary("Blend again", lambda e: m.blend_keyframes())]))
        self.body.controls = rows


class GradeScreen(Screen):
    def build(self):
        ui, m = self.ui, self.m
        editor = m.opened.editor.label if m.opened.editor else "your editor"
        self.watch = ft.Column(spacing=10)
        self._polling = True
        self.view.page.run_task(self.poll)
        self.refresh()
        steps = [ft.Container(ui.text(s, size=14), padding=ft.Padding(0, 10, 0, 10),
                              border=ft.Border(bottom=ft.BorderSide(1, ui.t["line"]))) for s in m.handoff_steps]
        side_w, main_w, two = self.columns(440)
        self.slice = SlicePanel(self, "Each frame gives one strip of the picture. After should change smoothly.",
                                width=side_w if two else main_w)
        self.slice.refresh()
        self.keys = KeyframesPanel(self) if m.keyframes_supported else None
        if self.keys:
            self.keys.refresh()
        self.seq = SequenceView(self, width=main_w, picture_height=self.view.content_height - self.ui.px(380))
        self.seq.refresh()
        main = ft.Column([ui.title("Grade"),
                          ui.text(f"Grade in {editor}, then export every frame.", dim=True, size=14),
                          self.seq.control(), *([self.keys.control()] if self.keys else []),
                          *([] if self.keys else steps)], spacing=8, width=main_w)
        side_col = ft.Column([ui.panel(self.watch), self.slice.control()], spacing=16,
                             width=side_w if two else main_w)
        if two:
            return ft.Row([main, side_col], spacing=32, vertical_alignment=ft.CrossAxisAlignment.START)
        return ft.Column([main, side_col], spacing=24)

    def refresh(self):
        ui, m = self.ui, self.m
        if hasattr(self, "slice"):
            self.slice.refresh()
        if getattr(self, "keys", None):
            self.keys.refresh()
        if getattr(self, "seq", None):
            self.seq.refresh()
        s = m.export or ExportStatus("waiting")
        choose = ui.secondary("Choose a folder", self.choose)
        if s.state == "waiting":
            self.watch.controls = [ui.text("Export", size=17),
                                   ui.text("Export every frame into a new folder next to the RAW files. The app "
                                           "picks it up when it's done.", dim=True, size=14), choose]
            if m.rapidraw_program is not None:
                self.watch.controls = [
                    ui.text("Export", size=17),
                    ui.text("RapidRAW renders every frame into a new folder, then Finish opens.", dim=True, size=14),
                    ft.Row([ui.primary("Export with RapidRAW", lambda e: m.export_with_rapidraw()), choose],
                           spacing=10, wrap=True)]
        elif s.state == "arriving":
            total = len(m.opened.seq)
            self.watch.controls = [ui.text(f"Export arriving: {s.frames} of {total} frames"),
                                   ui.text(str(s.folder), dim=True, mono=True, size=13),
                                   ft.ProgressBar(value=min(1.0, s.frames / total), color=ui.t["text"],
                                                  bgcolor=ui.t["line"]), choose]
        else:
            self.watch.controls = [ui.text(f"Found {s.frames} exported frames", size=17),
                                   ui.text(str(s.folder), dim=True, mono=True, size=13),
                                   ft.Row([ui.primary("Finish", lambda e: m.use_export()),
                                           *([ui.secondary("Export again with RapidRAW",
                                                           lambda e: m.export_with_rapidraw(replace=True))]
                                             if m.rapidraw_program is not None else []),
                                           choose], spacing=10, wrap=True),
                                   *([ui.text("Export again replaces these frames.", dim=True, size=13)]
                                     if m.rapidraw_program is not None else [])]

    async def poll(self):
        while self._polling:
            await asyncio.to_thread(self.m.poll_export)
            await asyncio.sleep(1.0)

    def leave(self):
        self._polling = False
        if getattr(self, "seq", None):
            self.seq.playing = False

    def on_key(self, e):
        if getattr(self, "seq", None):
            self.seq.on_key(e)

    async def choose(self, e):
        folder = await self.view.ask("get_directory_path", dialog_title="Choose the folder of exported frames")
        if folder:
            self.m.use_export(folder)


class FinishScreen(Screen):
    """The export window: presets up front, every option one click away, checks before anything is written."""

    def build(self):
        self.box = ft.Column(spacing=14, width=min(1100, self.view.content_width))
        self.advanced_open = getattr(self, "advanced_open", False)
        self._version = None
        self.go_btn = None
        self.refresh()
        return self.box

    def refresh(self):
        m = self.m
        version = (m.out_version, m.trim_version)
        if self._version != version:                         # settings or thumbnails changed: lay out again
            self._version = version
            self.box.controls = self._controls()
        if self.go_btn is not None:
            self.go_btn.disabled = not m.can_finish

    # -- layout ----------------------------------------------------------------------------------
    def _controls(self) -> list:
        ui, m, s = self.ui, self.m, self.m.out
        check = m.export_check

        def row(label, *controls):
            return ft.Row([ft.Container(ui.text(label, dim=True, size=14), width=150), *controls],
                          vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)

        def dropdown(value, options, on_pick, width=200):
            return ft.Dropdown(options=[ft.DropdownOption(key=str(k), text=t) for k, t in options], value=str(value),
                               width=width, dense=True, on_select=lambda e: on_pick(e.control.value))

        def path_field(value, on_done, width=520):
            return ft.TextField(value=value or "", text_size=13, text_style=ft.TextStyle(font_family=MONO),
                                dense=True, width=width, border_color=ui.t["line"],
                                on_blur=lambda e: on_done(e.control.value), on_submit=lambda e: on_done(e.control.value))

        current = m.preset
        presets = []
        for key, label in m.presets:
            presets.append((ui.primary if key == current else ui.secondary)(label, lambda e, k=key: m.choose_preset(k)))
        presets.append(ui.primary("Custom", None) if current == "custom" else
                       ui.secondary("Custom", None, disabled=True))        # always there, so nothing jumps
        detail = BUILT_IN[current]["detail"] if current in BUILT_IN else (
            "Your own preset." if current != "custom" else "Your own settings: save them as a preset below.")

        out = [ui.title("Export"),
               ui.text("Smooth the flicker left in your export, then write the video, the frames, or both.",
                       dim=True, size=14),
               row("Exported frames", ui.text(m.project.export or "Not chosen yet", mono=True, size=14),
                   ui.secondary("Change", self.choose_export)),
               row("Output", ft.SegmentedButton(
                   segments=[ft.Segment(value=v, label=ft.Text(t)) for v, t in
                             (("video", "Video"), ("frames", "Frames"), ("both", "Both"))],
                   selected=[s.output], allow_multiple_selection=False,
                   on_change=lambda e: m.set_export(output=list(e.control.selected)[0])))]
        if s.video:
            out += [row("Video preset", ft.Row(presets, spacing=8, wrap=True)),
                    row("", ui.text(detail, dim=True, size=14))]
        if s.video:
            out.append(row("Video", path_field(s.video_path, self.typed_video), ui.secondary("Change", self.choose_video)))
        if s.frames:
            out.append(row("Frames", path_field(s.frames_folder, lambda v: v.strip() and m.set_frames_folder(v.strip().strip('"'))),
                           ui.secondary("Change", self.choose_frames)))
            fmt = [dropdown(s.frames_format, [("jpg", "JPEG"), ("png", "PNG"), ("tif", "16-bit TIFF")],
                            lambda v: m.set_export(frames_format=v), 180)]
            if s.frames_format == "jpg":
                fmt += [ui.text("Quality", dim=True, size=13),
                        ft.Slider(min=70, max=100, divisions=30, value=s.jpeg_quality, label="{value}", width=220,
                                  on_change_end=lambda e: m.set_export(jpeg_quality=int(e.control.value)))]
            out.append(row("Frame format", *fmt))

        out.append(ft.TextButton(("▾ " if self.advanced_open else "▸ ") + "Advanced", on_click=self.toggle_advanced,
                                 style=ft.ButtonStyle(color=ui.t["text"])))
        if self.advanced_open:
            out += self._advanced()

        out.append(ui.panel(ft.Column(
            [ui.text(check["estimate"], size=14)]
            + [ui.text(n, dim=True, size=13) for n in check["notes"]]
            + [ui.text("Before exporting: " + p, size=14) for p in check["problems"]], spacing=6)))
        name = ft.TextField(hint_text="Preset name", dense=True, width=220, text_size=13, border_color=ui.t["line"])
        saving = [name, ui.secondary("Save as preset", lambda e: m.save_preset(name.value or ""))]
        if s.preset not in BUILT_IN and s.preset != "custom":
            saving.append(ui.secondary("Delete this preset", lambda e: m.delete_preset(s.preset)))
        self.go_btn = ui.primary("Export", lambda e: m.start_finish(), disabled=not m.can_finish)
        out.append(ft.Row([ft.Row(saving, spacing=8), self.go_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                          wrap=True))
        return out

    def _advanced(self) -> list:
        ui, m, s = self.ui, self.m, self.m.out

        def row(label, *controls):
            return ft.Row([ft.Container(ui.text(label, dim=True, size=14), width=150), *controls],
                          vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)

        def dropdown(value, options, on_pick, width=200):
            return ft.Dropdown(options=[ft.DropdownOption(key=str(k), text=t) for k, t in options], value=str(value),
                               width=width, dense=True, on_select=lambda e: on_pick(e.control.value))
        rows = []
        if s.video:
            res = [dropdown(s.resolution, [(k, "Original size" if k == "native" else f"Up to {k}") for k in RESOLUTIONS]
                            + [("custom", "Custom width")], lambda v: m.set_export(resolution=v), 200)]
            if s.resolution == "custom":
                res.append(ft.TextField(value=str(s.custom_width or ""), hint_text="width in pixels", width=150, dense=True,
                                        text_size=13, border_color=ui.t["line"],
                                        on_blur=lambda e: e.control.value.strip().isdigit() and
                                        m.set_export(custom_width=int(e.control.value))))
            rows.append(row("Resolution", *res))
            rows.append(row("Frame rate", dropdown(s.fps, [(f, f"{f:g}") for f in FRAME_RATES],
                                                   lambda v: m.set_export(fps=float(v)), 140)))
            fmt = [dropdown(s.codec, CODEC_CHOICES, lambda v: m.set_export(codec=v, chroma=None, bit_depth=None), 300)]
            if s.codec == "h265":
                fmt.append(dropdown(s.bit_depth or 10, DEPTH_CHOICES, lambda v: m.set_export(bit_depth=int(v)), 270))
            if s.codec == "prores":
                fmt.append(dropdown(s.prores_profile, PRORES_CHOICES, lambda v: m.set_export(prores_profile=v), 280))
            else:
                fmt.append(dropdown(s.chroma or "420", CHROMA_CHOICES, lambda v: m.set_export(chroma=v), 360))
            rows.append(row("Format", *fmt))
            rows.append(row("", ft.Container(ui.text(explain(s), dim=True, size=13), width=720)))
            if s.codec != "prores":
                crf = s.quality if s.quality is not None else (16 if s.codec == "h264" else 18)
                rows.append(row("Quality", ui.text("Higher", dim=True, size=13),
                                ft.Slider(min=8, max=32, divisions=24, value=crf, width=260,
                                          label="{value}" + (" (the default)" if s.quality is None else ""),
                                          on_change_end=lambda e: m.set_export(quality=int(e.control.value))),
                                ui.text("Lower", dim=True, size=13),
                                ui.text("Higher quality means bigger files.", dim=True, size=13)))
            rows.append(row("Dither", ft.Switch(value=s.dither, on_change=lambda e: m.set_export(dither=e.control.value)),
                            ui.text("hides banding in smooth skies (costs a little file size)", dim=True, size=13)))
        rows += self._trim()
        return rows

    def _trim(self) -> list:
        """Trim by eye: thumbnails across the sequence, a two-handled slider, and the chosen first and last frames."""
        ui, m, s = self.ui, self.m, self.m.out
        n = m.trim_frames
        if n < 2:
            return [ft.Row([ft.Container(ui.text("Trim", dim=True, size=14), width=150),
                            ui.text("Making thumbnails...", dim=True, size=13)])]
        first, last = s.first or 1, s.last or n
        strip = ft.Row([ft.Image(src=_image_bytes(path), width=62, height=42, fit=ft.BoxFit.COVER, border_radius=2,
                                 opacity=1.0 if first - 1 <= i <= last - 1 else 0.3) for i, path in m.trim_strip],
                       spacing=2)
        slider = ft.RangeSlider(min=1, max=n, start_value=first, end_value=last, divisions=n - 1, label="{value}",
                                width=62 * len(m.trim_strip) + 2 * (len(m.trim_strip) - 1),
                                on_change_end=lambda e: m.set_trim(int(e.control.start_value), int(e.control.end_value)))
        ends = []
        for label, frame, path in (("First", first, m.trim_ends[0] if m.trim_ends else None),
                                   ("Last", last, m.trim_ends[1] if m.trim_ends else None)):
            ends.append(ft.Column([ft.Image(src=_image_bytes(path), width=180, height=120, fit=ft.BoxFit.COVER, border_radius=2)
                                   if path else ft.Container(width=180, height=120),
                                   ui.text(f"{label}: frame {frame} ({m.frame_time(frame)})", dim=True, size=13)],
                                  spacing=4))
        summary = (f"All {n} frames" if (first, last) == (1, n) else
                   f"Frames {first} to {last}: {last - first + 1} of {n}, "
                   f"{(last - first + 1) / max(s.fps, 1):.1f} s of video")
        return [ft.Row([ft.Container(ui.text("Trim", dim=True, size=14), width=150),
                        ft.Column([strip, slider, ui.text(summary, dim=True, size=13), ft.Row(ends, spacing=16)],
                                  spacing=6)], vertical_alignment=ft.CrossAxisAlignment.START)]

    # -- actions ----------------------------------------------------------------------------------
    def toggle_advanced(self, e):
        self.advanced_open = not self.advanced_open
        self.box.controls = self._controls()
        self.view.page.update()

    def typed_video(self, value):
        text = (value or "").strip().strip('"')
        if text:
            self.m.set_video_path(Path(text).with_suffix(self.m.out.suffix))

    async def choose_export(self, e):
        folder = await self.view.ask("get_directory_path", dialog_title="Choose the folder of exported frames")
        if folder:
            self.m.use_export(folder)

    async def choose_frames(self, e):
        folder = await self.view.ask("get_directory_path", dialog_title="Choose a folder for the corrected frames")
        if folder:
            self.m.set_frames_folder(folder)

    async def choose_video(self, e):
        current = self.m.video_path
        path = await self.view.ask(
            "save_file", dialog_title="Save the video as", file_name=current.name if current else None,
            initial_directory=str(current.parent) if current else None,
            file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=[self.m.out.suffix.lstrip(".")])
        if path:
            self.m.set_video_path(Path(path).with_suffix(self.m.out.suffix))


class DoneScreen(Screen):
    def build(self):
        ui, m = self.ui, self.m
        self.flags = ft.Column(spacing=0)
        self.seq = SequenceView(self, keyframes=False, source=ExportSource(m),
                                width=min(1100, self.view.content_width) - 34,
                                picture_height=self.view.content_height - self.ui.px(380))
        self.refresh()
        video = m.finished.video
        return ft.Column([
            ui.title("Review"),
            *[ui.text(line, dim=True, size=14) for line in m.result_lines],
            ui.panel(ft.Row([ui.text(str(video), mono=True, size=14, selectable=True),
                             ft.Row([ui.secondary("Show in folder", lambda e: open_path(video, select=True)),
                                     ui.primary("Play video", lambda e: open_path(video))], spacing=10)],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True)) if video else ft.Container(),
            ui.text("Your film", size=17),
            ui.panel(self.seq.control()),
            *self._checks(),
            *self._strip(),
            ui.text("Frames worth a look", size=14), self.flags,
            ft.Row([ui.secondary("Open another folder", lambda e: m.go("start"))]),
        ], spacing=14, width=min(1100, self.view.content_width))

    def _checks(self) -> list:
        """What the checks found: what each means, and what to do about it."""
        ui, m = self.ui, self.m
        findings = m.finished.diagnosis
        if not findings:
            return []
        marks = {"ok": "✓", "note": "i", "look": "!"}
        rows = []
        for f in findings:
            badge = ft.Container(ft.Text(marks.get(f.kind, "i"), size=12, color=ui.t["bg"]), width=20, height=20,
                                 border_radius=10, bgcolor=ui.t["text"] if f.kind != "ok" else ui.t["dim"],
                                 alignment=ft.Alignment.CENTER)
            body = [ui.text(f.title, size=14)]
            if f.detail:
                body.append(ui.text(f.detail, dim=True, size=13))
            if f.fix and f.kind != "ok":
                body.append(ui.text("What to do: " + f.fix, size=13))
            if f.boxes and m.finished.thumbnail is not None:
                body.append(ft.Image(src=_marked(m.finished.thumbnail, f.boxes, ui.t), width=420,
                                     fit=ft.BoxFit.CONTAIN, border_radius=2))
            rows.append(ft.Container(ft.Row([badge, ft.Column(body, spacing=4, expand=True)],
                                            vertical_alignment=ft.CrossAxisAlignment.START),
                                     padding=ft.Padding(0, 10, 0, 10),
                                     border=ft.Border(bottom=ft.BorderSide(1, ui.t["line"]))))
        return [ui.text("Checks", size=14), ft.Column(rows, spacing=0)]

    def _strip(self) -> list:
        """The time strip: a quick-to-load view to zoom and pan, and the full-detail file a click away."""
        ui, m = self.ui, self.m
        path = m.finished.strip
        if not path or "slice_before" not in m.finished.strip_columns:
            return []
        self.strip_view = ft.Image(src=b"", width=880, fit=ft.BoxFit.CONTAIN)
        self.view.page.run_task(self._load_strip)
        return [ui.text("Time slice", size=14),
                ui.text("Each frame gives one strip of the picture. Jumps show as stripes.", dim=True, size=13),
                ft.Container(ft.InteractiveViewer(content=self.strip_view, min_scale=1, max_scale=12,
                                                  boundary_margin=ft.Margin.all(40)),
                             bgcolor=ui.t["surround"], padding=8, border_radius=2, width=896, height=420,
                             clip_behavior=ft.ClipBehavior.HARD_EDGE),
                ft.Row([ui.secondary("Open full size", lambda e: open_path(path)),
                        ui.secondary("Show in folder", lambda e: open_path(path, select=True)),
                        ui.text(path.name, dim=True, mono=True, size=13)], spacing=10)]

    async def _load_strip(self):
        import io
        from PIL import Image
        cols = self.m.finished.strip_columns

        def make() -> bytes:
            from ...timeslice import stack, to_8bit
            img = Image.fromarray(stack(to_8bit(cols["slice_before"].picture()), to_8bit(cols["slice_after"].picture())))
            img.thumbnail((1760, 2400))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
        self.strip_view.src = await asyncio.to_thread(make)
        self.view.page.update()

    def on_key(self, e):
        self.seq.on_key(e)

    def leave(self):
        self.seq.playing = False

    def refresh(self):
        ui, m = self.ui, self.m
        if getattr(self, "seq", None):
            self.seq.refresh()
        items = m.worth_a_look
        if not items:
            self.flags.controls = [ui.text("No frames stand out.",
                                           dim=True, size=14)]
            return
        self.flags.controls = [ft.Container(
            ft.Row([ui.text(name, mono=True, size=14), ft.Container(ui.text(note, dim=True, size=14), expand=True),
                    ui.secondary("Looks fine", lambda e, n=name: m.looks_fine(n))], spacing=16),
            padding=ft.Padding(0, 8, 0, 8), border=ft.Border(bottom=ft.BorderSide(1, ui.t["line"])))
            for name, note in items]


SCREENS = {"start": StartScreen, "folder": FolderScreen, "prepare": PrepareScreen, "grade": GradeScreen,
           "finish": FinishScreen, "done": DoneScreen}


# --- the shell --------------------------------------------------------------------------------------

class View:
    def __init__(self, page: ft.Page, model: AppModel):
        self.page, self.model = page, model
        self.dark = page.theme_mode == ft.ThemeMode.DARK
        self._apply_theme()
        self.picker = ft.FilePicker()
        page.services.append(self.picker)
        self.screen: Screen | None = None
        self.screen_name = None
        self._fit()
        self._build_shell()
        model.subscribe(self.refresh)
        page.on_keyboard_event = self._key
        page.on_resize = lambda e: page.run_task(self._resized)
        self.refresh()

    # --- light and dark ---------------------------------------------------------------------------------

    def _apply_theme(self) -> None:
        from ..theme import CHART
        self.tokens = DARK if self.dark else LIGHT
        scale = getattr(getattr(self, "ui", None), "scale", 1.12)
        self.ui = UI(self.tokens)
        self.ui.scale = scale
        self.chart_theme = Theme(**CHART["dark" if self.dark else "light"])

    def toggle_theme(self, e=None) -> None:
        """Light or dark: the whole window at once, remembered for next time."""
        from ...app import config
        self.dark = not self.dark
        prefs = config.preferences()
        prefs["theme"] = "dark" if self.dark else "light"
        config.save_preferences(prefs)
        self.page.theme_mode = ft.ThemeMode.DARK if self.dark else ft.ThemeMode.LIGHT
        self.page.bgcolor = (DARK if self.dark else LIGHT)["bg"]
        self._apply_theme()
        if self.screen is not None:
            self.screen.leave()
        self.screen, self.screen_name = None, None
        self.page.controls.clear()
        self._build_shell()
        self.refresh()

    # --- the window's size ------------------------------------------------------------------------------

    MARGIN = 64                                         # the page's margins, left and right together

    def _fit(self) -> None:
        """Text grows a little with the window: from 1.12x the design sizes up to 1.3x on a large screen."""
        self.fitted = (int(self.page.width or 1400), int(self.page.height or 900))
        self.ui.scale = min(1.3, max(1.12, 1.12 * self.fitted[0] / 1600))

    @property
    def content_width(self) -> int:
        return max(360, int(self.page.width or 1400) - self.MARGIN)

    @property
    def content_height(self) -> int:
        return max(400, int(self.page.height or 900))

    async def _resized(self) -> None:
        """Lay the screen out again for the new size, once the window has stopped changing."""
        size = (int(self.page.width or 0), int(self.page.height or 0))
        await asyncio.sleep(0.3)
        if size != (int(self.page.width or 0), int(self.page.height or 0)) or size == self.fitted:
            return
        self._fit()
        self.screen_name = None                          # rebuild the current screen at the new size
        self.refresh()

    def _key(self, e) -> None:
        if e.key == "Escape" and SliceViewer.current is not None:
            SliceViewer.current.close()
            return
        if FullView.current is not None:
            if e.key == "Escape":
                FullView.current.close()
            else:
                FullView.current.seq.on_key(e)
            return
        handler = getattr(self.screen, "on_key", None)
        if handler is not None and SliceViewer.current is None:
            handler(e)

    async def ask(self, method: str, **kwargs):
        """Open a file dialog. If the window doesn't answer (Flet 1.0 sometimes loses track of a
        long-lived picker: "Timeout waiting for invoke method listener"), try once more with a fresh
        picker, as Flet's own examples do; if that fails too, say so plainly instead of crashing."""
        last = None
        for picker in (self.picker, None):
            try:
                return await getattr(picker or ft.FilePicker(), method)(**kwargs)
            except Exception as e:                       # noqa: BLE001 - any failure here is the dialog's
                last = e
        self.model._notice("The file dialog didn't open, so nothing was changed. "
                           + ("You can type the path into the box instead. " if method == "save_file" else "")
                           + f"(Flet said: {str(last)[:160]})", "error")
        self.model.changed()
        return None

    def _build_shell(self):
        ui, t = self.ui, self.tokens
        self.project_name = ui.text("", dim=True, size=14)
        self.stage_bar = ft.Row(spacing=4)
        self.banner_text = ui.text("", size=14)
        self.banner = ft.Container(ft.Row([ft.Container(self.banner_text, expand=True),
                                           ft.TextButton("Dismiss", on_click=lambda e: self.model.dismiss_message(),
                                                         style=ft.ButtonStyle(color=t["text"]))]),
                                   padding=ft.Padding(16, 10, 16, 10), bgcolor=t["panel"],
                                   border=ft.Border(bottom=ft.BorderSide(1, t["line"])), visible=False)
        self.job_text = ui.text("", size=14)
        self.job_bar = ft.ProgressBar(value=None, color=t["text"], bgcolor=t["line"])
        self.cancel_btn = ui.secondary("Cancel", lambda e: self.model.cancel())
        self.job_panel = ft.Container(ft.Column([ft.Row([ft.Container(self.job_text, expand=True), self.cancel_btn]),
                                                 self.job_bar], spacing=8),
                                      padding=ft.Padding(24, 12, 24, 12), bgcolor=t["panel"],
                                      border=ft.Border(bottom=ft.BorderSide(1, t["line"])), visible=False)
        self.body = ft.Container(padding=ft.Padding(32, 28, 32, 28), expand=True)
        header = ft.Container(ft.Row([ui.mark(), ft.Text("Harbium", font_family=DISPLAY, size=26, color=t["text"]),
                                      ui.text("Timelapse", dim=True, size=14), ft.Container(width=16),
                                      ft.Container(self.project_name, expand=True),
                                      ft.TextButton("Light mode" if self.dark else "Dark mode",
                                                    on_click=self.toggle_theme,
                                                    style=ft.ButtonStyle(color=t["dim"]))], spacing=10,
                                     vertical_alignment=ft.CrossAxisAlignment.CENTER),
                              padding=ft.Padding(24, 14, 24, 14), border=ft.Border(bottom=ft.BorderSide(1, t["line"])))
        self.stage_box = ft.Container(self.stage_bar, padding=ft.Padding(20, 6, 20, 6),
                                      border=ft.Border(bottom=ft.BorderSide(1, t["line"])))
        self.page.add(ft.Column([header, self.stage_box, self.banner, self.job_panel,
                                 ft.Column([self.body], scroll=ft.ScrollMode.AUTO, expand=True)],
                                spacing=0, expand=True))

    def _stage_button(self, number: int, screen: str, label: str, state: str) -> ft.Control:
        t = self.tokens
        filled = state in ("done", "current")
        dot = ft.Container(ft.Text("✓" if state == "done" else str(number), size=12,
                                   color=t["bg"] if filled else t["dim"]),
                           width=22, height=22, border_radius=11, alignment=ft.Alignment.CENTER,
                           bgcolor=t["text"] if filled else None, border=ft.Border.all(1, t["line"] if not filled else t["text"]))
        text = ft.Text(label, font_family=BODY, size=14, weight=ft.FontWeight.W_600 if state == "current" else None,
                       color=t["text"] if state in ("current", "done", "open") else t["dim"])
        return ft.Container(ft.Row([dot, text], spacing=8), padding=ft.Padding(10, 8, 10, 8),
                            on_click=None if state == "locked" else (lambda e, s=screen: self.model.go(s)),
                            opacity=0.5 if state == "locked" else 1.0)

    def refresh(self):
        m = self.model
        self.project_name.value = Path(m.opened.seq.folder).name if m.opened else ""
        self.stage_box.visible = m.screen != "start"
        self.stage_bar.controls = [self._stage_button(i + 1, s, label, state)
                                   for i, (s, label, state) in enumerate(m.stage_items())]
        self.banner.visible = bool(m.message)
        self.banner_text.value = m.message or ""
        self.banner_text.color = self.tokens["text"]
        self.job_panel.visible = m.job is not None
        if m.job is not None:
            self.job_text.value = f"{m.job.name}: {m.job.text}"
            self.job_bar.value = m.job.fraction
            self.cancel_btn.disabled = m.job.cancelling
        # Rebuild on a new screen, or on the same screen with a newly opened folder (choosing an editor reopens
        # the folder on the Folder screen: a refresh alone left the old cards showing).
        if self.screen_name is None and self.screen is not None:
            self.screen.leave()
            self.screen = None
        if m.screen != self.screen_name or m.opened is not getattr(self, "opened_shown", None):
            if self.screen is not None:
                self.screen.leave()
            self.screen_name = m.screen
            self.opened_shown = m.opened
            self.screen = SCREENS[m.screen](self)
            self.body.content = self.screen.build()
        else:
            self.screen.refresh()
        self.page.update()


def start(page: ft.Page) -> View:
    """Set up the window and show the app. Fonts are read from the app's assets folder."""
    page.title = "Harbium Timelapse"
    page.padding = 0
    page.fonts = FONTS
    from ...app import config
    dark = config.preferences().get("theme") == "dark"            # light unless you chose dark

    def brand(t: dict) -> ft.Theme:
        # Switches, sliders, checkboxes, the presets and the dropdown all take these: bone on ink, or ink on bone.
        return ft.Theme(font_family=BODY, color_scheme=ft.ColorScheme(
            primary=t["text"], on_primary=t["bg"], secondary=t["text"], on_secondary=t["bg"],
            surface=t["bg"], on_surface=t["text"], outline=t["line"], secondary_container=t["line"],
            on_secondary_container=t["text"]))
    page.theme, page.dark_theme = brand(LIGHT), brand(DARK)
    page.theme_mode = ft.ThemeMode.DARK if dark else ft.ThemeMode.LIGHT
    page.bgcolor = (DARK if dark else LIGHT)["bg"]

    async def call(fn):
        fn()
    model = AppModel(post=lambda fn: page.run_task(call, fn))
    return View(page, model)
