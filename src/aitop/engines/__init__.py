"""Engine adapters — one module per AI runtime."""

from aitop.engines.base import BaseEngine, EngineCapability, classify_scope
from aitop.engines.lmstudio import LMStudioEngine
from aitop.engines.ollama import OllamaEngine
from aitop.engines.openai_compat import LlamaServerEngine, MLXEngine, VLLMEngine
from aitop.engines.registry import ADAPTERS, EngineRegistry

__all__ = [
    "ADAPTERS",
    "BaseEngine",
    "EngineCapability",
    "EngineRegistry",
    "LMStudioEngine",
    "LlamaServerEngine",
    "MLXEngine",
    "OllamaEngine",
    "VLLMEngine",
    "classify_scope",
]
