"""Single source of truth for the package version.

Kept in its own module so engine adapters can read it without importing the
package root (which would circular-import the collector → registry → engines).
"""

__version__ = "0.2.2"
