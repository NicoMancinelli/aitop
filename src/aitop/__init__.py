"""aitop — AI-centric neofetch meets btop.

Public surface: build a `SnapshotCollector`, await `collect()`, render the
resulting `SystemSnapshot` however you like.
"""

from aitop.bus import EventBus, Topic
from aitop.collector import SnapshotCollector
from aitop.config import Config
from aitop.models import EngineSnapshot, HardwareSnapshot, SystemSnapshot
from aitop.version import __version__

__all__ = [
    "Config",
    "EngineSnapshot",
    "EventBus",
    "HardwareSnapshot",
    "SnapshotCollector",
    "SystemSnapshot",
    "Topic",
    "__version__",
]
