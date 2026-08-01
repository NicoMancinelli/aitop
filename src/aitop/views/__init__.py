"""Renderers. Every view is a pure function of a `SystemSnapshot`."""

from aitop.views.neofetch import print_neofetch, render_neofetch

__all__ = ["print_neofetch", "render_neofetch"]
