"""ASCII art for the neofetch header.

Lines are padded to a common width at render time, so editing the art below
never breaks column alignment.
"""

from __future__ import annotations

from rich.text import Text

LOGO = r"""
       ▄▄▄▄▄▄▄▄▄
    ▄██▀▀     ▀▀██▄
   ██▀   ▄▄▄▄▄   ▀██
  ██   ▄██▀ ▀██▄   ██
  ██   ██     ██   ██
  ██   ▀██▄ ▄██▀   ██
   ██▄   ▀▀▀▀▀   ▄██
    ▀██▄▄     ▄▄██▀
       ▀▀▀▀▀▀▀▀▀
   ▄▀█ █ ▀█▀ █▀█ █▀█
   █▀█ █  █  █▄█ █▀▀
""".strip("\n").splitlines()

# Cyan -> magenta ramp, one entry per logo row.
GRADIENT = (
    "bright_cyan",
    "bright_cyan",
    "cyan",
    "cyan",
    "bright_blue",
    "blue",
    "magenta",
    "magenta",
    "bright_magenta",
    "bold bright_magenta",
    "bold bright_magenta",
)


def logo_width() -> int:
    return max((len(line) for line in LOGO), default=0)


def render_logo(color: bool = True) -> Text:
    """The logo as a single padded, gradient-coloured `Text` block."""
    width = logo_width()
    text = Text()
    for index, line in enumerate(LOGO):
        style = GRADIENT[index % len(GRADIENT)] if color else ""
        text.append(line.ljust(width), style=style)
        if index < len(LOGO) - 1:
            text.append("\n")
    return text
