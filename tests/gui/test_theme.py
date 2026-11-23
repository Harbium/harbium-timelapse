"""Both themes stay easy to read: body text at least 7:1 against the background and panels, dim text at
least 4.5:1 (WCAG AAA and AA for normal text)."""

import pytest

from harbium_timelapse.gui.theme import DARK, LIGHT


def luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_text_is_easy_to_read(theme):
    for surface in ("bg", "panel"):
        assert contrast(theme["text"], theme[surface]) >= 7
        assert contrast(theme["dim"], theme[surface]) >= 4.5
    assert contrast(theme["primary_text"], theme["primary_bg"]) >= 7
