"""RouterBase provider profile: OpenAI-compatible gateway with slash-form model ids.

``GET /v1/models`` is public and lists every surface the gateway serves, tagged with
``modality`` (``chat`` / ``image`` / ``video`` / ``audio`` / ``embedding``). Only chat rows
belong in the chat picker, and the generic ``chat_catalog_ids`` filter does not read that
field, so ``fetch_models`` drops non-chat rows here. The same rows carry ``context_length``;
it is remembered per base URL so ``get_model_context_length`` answers from memory.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

ROUTERBASE_DEFAULT_BASE_URL = "https://routerbase.com/v1"

# Rows with no modality fail open (kept); anything else is not a chat completion.
_CHAT_MODALITIES = frozenset({"chat", "text"})
# A failed catalog probe is not retried on every context lookup.
_RETRY_AFTER_FAILURE_S = 300.0

_lock = threading.Lock()
_context_by_base: dict[str, dict[str, int]] = {}
_failed_at: dict[str, float] = {}


def _is_chat_row(item: dict[str, Any]) -> bool:
    modality = item.get("modality")
    if not isinstance(modality, str) or not modality.strip():
        return True
    return modality.strip().lower() in _CHAT_MODALITIES


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


class RouterBaseProfile(ProviderProfile):
    """RouterBase: chat-only picker from a mixed-modality public catalog."""

    def _models_url(self, base_url: str | None) -> tuple[str, str]:
        base = ((base_url or "").strip() or self.base_url).rstrip("/")
        return base, base + "/models"

    def _fetch_rows(self, *, api_key: str | None, base_url: str | None, timeout: float) -> list[dict[str, Any]] | None:
        base, url = self._models_url(base_url)
        from hermes_cli.urllib_security import open_credentialed_url

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("routerbase: catalog fetch from %s failed: %s", url, exc)
            with _lock:
                _failed_at[base] = time.monotonic()
            return None
        items = data if isinstance(data, list) else (data.get("data") if isinstance(data, dict) else None)
        if not isinstance(items, list):
            return None
        rows = [item for item in items if isinstance(item, dict) and item.get("id")]
        contexts = {
            str(row["id"]): ctx
            for row in rows
            if _is_chat_row(row) and (ctx := _positive_int(row.get("context_length"))) is not None
        }
        with _lock:
            _context_by_base[base] = contexts
            _failed_at.pop(base, None)
        return rows

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        rows = self._fetch_rows(api_key=api_key, base_url=base_url, timeout=timeout)
        if rows is None:
            return None
        from hermes_cli.chat_catalog import chat_catalog_ids, note_generation_model

        chat_rows = []
        for row in rows:
            if _is_chat_row(row):
                chat_rows.append(row)
            else:
                note_generation_model(row["id"])
        return chat_catalog_ids(chat_rows)

    def get_model_context_length(self, model: str) -> int | None:
        """``context_length`` from the RouterBase catalog; None when absent so core defaults apply."""
        base, _ = self._models_url(None)
        with _lock:
            known = _context_by_base.get(base)
            failed_at = _failed_at.get(base)
        if known is None:
            if failed_at is not None and time.monotonic() - failed_at < _RETRY_AFTER_FAILURE_S:
                return None
            self._fetch_rows(api_key=None, base_url=None, timeout=5.0)
            with _lock:
                known = _context_by_base.get(base)
        return (known or {}).get(model)


routerbase = RouterBaseProfile(
    name="routerbase",
    aliases=("router-base", "routerbase-ai"),
    display_name="RouterBase",
    description="RouterBase — OpenAI-compatible gateway to OpenAI, Anthropic, Google, DeepSeek, Qwen, Kimi, GLM and more",
    signup_url="https://routerbase.com/api-keys",
    env_vars=("ROUTERBASE_API_KEY", "ROUTERBASE_BASE_URL"),
    base_url=ROUTERBASE_DEFAULT_BASE_URL,
    auth_type="api_key",
    default_aux_model="deepseek/deepseek-v4.1-flash",
    fallback_models=(
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "zai/glm-5.1",
        "qwen/qwen3-coder-480b-a35b-instruct",
    ),
)

register_provider(routerbase)
