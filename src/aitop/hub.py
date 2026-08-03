"""Hugging Face hub search — lightweight, no `huggingface_hub` dependency.

Used by `aitop models search` to surface GGUF / MLX candidates the user can
then pull through Ollama (or download manually). We only hit the public
REST API; authentication is optional via `HF_TOKEN`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from aitop.models import HubModel
from aitop.utils.parse import first, parse_timestamp, to_int

log = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/models"
_USER_AGENT = "aitop/0.2"


async def search_hub(
    query: str,
    *,
    limit: int = 20,
    filter_tag: str | None = "gguf",
    token: str | None = None,
    timeout: float = 10.0,
) -> list[HubModel]:
    """Search the HF model hub. Returns an empty list on any failure."""
    headers = {"User-Agent": _USER_AGENT}
    auth = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    params: dict[str, Any] = {
        "search": query,
        "limit": max(1, min(limit, 100)),
        "sort": "downloads",
        "direction": "-1",
    }
    if filter_tag:
        params["filter"] = filter_tag

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=True) as client:
            response = await client.get(HF_API, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.debug("HF search failed: %s", exc)
        return []

    if not isinstance(payload, list):
        return []

    out: list[HubModel] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        model_id = str(first(entry, "id", "modelId", default="") or "")
        if not model_id:
            continue
        tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
        out.append(
            HubModel(
                id=model_id,
                author=model_id.split("/", 1)[0] if "/" in model_id else None,
                downloads=to_int(entry.get("downloads")),
                likes=to_int(entry.get("likes")),
                tags=[str(t) for t in tags if t is not None],
                pipeline_tag=first(entry, "pipeline_tag"),
                last_modified=parse_timestamp(first(entry, "lastModified", "last_modified")),
                url=f"https://huggingface.co/{model_id}",
            )
        )
    return out
