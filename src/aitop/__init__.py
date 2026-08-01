"""aitop — AI-centric neofetch meets btop.

Public surface: build a `SnapshotCollector`, await `collect()`, render the
resulting `SystemSnapshot` however you like.
"""

__version__ = "0.1.1"

from aitop.bus import EventBus, Topic
from aitop.collector import SnapshotCollector
from aitop.config import Config
from aitop.models import EngineSnapshot, HardwareSnapshot, SystemSnapshot

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
