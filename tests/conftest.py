from __future__ import annotations

import pytest

from aitop import selfupdate
from aitop.config import EndpointConfig
from aitop.models import EngineKind


@pytest.fixture(autouse=True)
def no_network_update_checks(tmp_path, monkeypatch):
    """No test may reach github.com, and none may read the developer's cache.

    Tests that exercise the update path opt back in by patching
    `check_for_update` or `cache_path` themselves.
    """
    monkeypatch.setattr(selfupdate, "cache_path", lambda: tmp_path / "update-check.json")
    monkeypatch.setenv("AITOP_NO_UPDATE_CHECK", "1")


@pytest.fixture
def ollama_endpoint() -> EndpointConfig:
    return EndpointConfig(kind=EngineKind.OLLAMA, host="127.0.0.1", port=11434)


@pytest.fixture
def lmstudio_endpoint() -> EndpointConfig:
    return EndpointConfig(kind=EngineKind.LMSTUDIO, host="127.0.0.1", port=1234)


@pytest.fixture
def ollama_tags() -> dict:
    return {
        "models": [
            {
                "name": "llama3.2:3b",
                "model": "llama3.2:3b",
                "modified_at": "2025-01-14T10:22:31.833753871-08:00",
                "size": 2019393189,
                "digest": "a80c4f17acd5",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {
                "name": "qwen2.5-coder:7b",
                "model": "qwen2.5-coder:7b",
                "modified_at": "2025-02-01T09:00:00Z",
                "size": 4683087519,
                "details": {
                    "format": "gguf",
                    "family": "qwen2",
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                },
            },
        ]
    }


@pytest.fixture
def ollama_ps() -> dict:
    return {
        "models": [
            {
                "name": "llama3.2:3b",
                "model": "llama3.2:3b",
                "size": 4108928000,
                "size_vram": 4108928000,
                "context_length": 8192,
                "expires_at": "2099-06-04T14:38:31.83753-07:00",
                "details": {
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }


@pytest.fixture
def lmstudio_native() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-7b-instruct",
                "object": "model",
                "type": "llm",
                "publisher": "lmstudio-community",
                "arch": "qwen2",
                "compatibility_type": "gguf",
                "quantization": "Q4_K_M",
                "state": "loaded",
                "max_context_length": 32768,
                "loaded_context_length": 4096,
            },
            {
                "id": "mlx-community/Llama-3.2-1B",
                "object": "model",
                "type": "llm",
                "arch": "llama",
                "compatibility_type": "mlx",
                "quantization": "4bit",
                "state": "not-loaded",
                "max_context_length": 131072,
            },
        ],
    }
