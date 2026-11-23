"""Harbium's brand, as the app uses it (GUI_PLAN.md, section 9).

A neutral light theme (the default) and a dark one, switched in the header and
remembered; Cormorant Garamond for titles, Inter for text, DM Mono for frame
names and paths. Behind photographs the surround is a dark neutral grey in both:
tinted or bright surroundings change how colour is judged.
"""

from __future__ import annotations

# Light (the default): neutral, near-white, crisp dark text. Dark: neutral charcoal, light text. Behind photographs
# a dark neutral grey in both, as photo editors use: a tinted or bright surround changes how colour is judged.
LIGHT = {
    "bg": "#f6f6f4", "panel": "#ffffff", "line": "#dededa", "text": "#161616", "dim": "#5f5f5b",
    "surround": "#2a2a2a", "primary_bg": "#161616", "primary_text": "#ffffff",
    "measured": (150, 150, 145), "curve": (22, 22, 22),
}
DARK = {
    "bg": "#141414", "panel": "#1d1d1d", "line": "#303030", "text": "#ececea", "dim": "#a2a29d",
    "surround": "#0e0e0e", "primary_bg": "#4a4a4a", "primary_text": "#f6f6f4",      # raised, not glaring
    "measured": (125, 125, 120), "curve": (236, 236, 234),
}
# The timeline's colours (models.timeline.Theme) for each mode
CHART = {
    "light": {"background": (255, 255, 255), "grid": (232, 232, 228), "text": (95, 95, 91), "mark": (22, 22, 22)},
    "dark": {"background": (29, 29, 29), "grid": (48, 48, 48), "text": (162, 162, 157), "mark": (236, 236, 234)},
}
FONTS = {"Inter": "fonts/Inter.ttf", "Cormorant": "fonts/CormorantGaramond.ttf", "DM Mono": "fonts/DMMono-Regular.ttf"}
DISPLAY, BODY, MONO = "Cormorant", "Inter", "DM Mono"
