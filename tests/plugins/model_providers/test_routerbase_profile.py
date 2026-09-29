"""RouterBase profile: registry wiring, chat-only catalog, and catalog-sourced context length.

RouterBase's public ``/v1/models`` mixes chat, image, video, audio and embedding rows under a
``modality`` field. The picker must only offer rows that can serve a chat completion, and the
``context_length`` those rows publish is the window the context compressor should use.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest


@pytest.fixture
def routerbase(monkeypatch):
    import model_tools  # noqa: F401  (plugin discovery registers the profile)
    import providers

    profile = providers.get_provider_profile("routerbase")
    assert profile is not None, "routerbase provider profile must be registered"
    module = __import__(type(profile).__module__, fromlist=["_context_by_base"])
    monkeypatch.setattr(module, "_context_by_base", {})
    monkeypatch.setattr(module, "_failed_at", {})
    return profile


_ROWS: list[dict[str, Any]] = [
        {"id": "deepseek/deepseek-v4.1-flash", "modality": "chat", "context_length": 1048576},
        {"id": "openai/gpt-5.4", "modality": "chat", "context_length": 1050000},
        {"id": "qwen/qwen-2.5-72b-instruct", "modality": "chat"},  # limits not published yet
        {"id": "legacy/untagged-model"},  # no modality: fail open
        {"id": "bytedance/seedream-4.5", "modality": "image"},
        {"id": "vidu/q3-pro", "modality": "video"},
        {"id": "openai/text-embedding-3-small", "modality": "embedding"},
        {"id": "openai/whisper-1", "modality": "audio"},
]
_CATALOG = {"object": "list", "data": _ROWS}


def _serve(monkeypatch, payload, seen=None):
    def fake_open(req, *, timeout, **kwargs):
        if seen is not None:
            seen.append(req)
        if isinstance(payload, Exception):
            raise payload
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)


def test_aliases_and_env_vars_resolve_to_one_identity(routerbase):
    import providers

    for alias in ("router-base", "routerbase-ai"):
        assert providers.get_provider_profile(alias) is routerbase
    key_vars = [v for v in routerbase.env_vars if not v.endswith("_BASE_URL")]
    url_vars = [v for v in routerbase.env_vars if v.endswith("_BASE_URL")]
    assert key_vars and url_vars


def test_hostname_maps_back_to_provider(routerbase):
    from agent.model_metadata import _URL_TO_PROVIDER

    assert _URL_TO_PROVIDER.get(routerbase.get_hostname()) == routerbase.name


def test_fallback_and_aux_models_are_slash_form(routerbase):
    for model in (routerbase.default_aux_model, *routerbase.fallback_models):
        assert "/" in model


def test_fetch_models_keeps_only_chat_rows(routerbase, monkeypatch):
    seen = []
    _serve(monkeypatch, _CATALOG, seen)

    ids = routerbase.fetch_models(api_key="sk-rb-test")

    chat_or_untagged = [
        row["id"] for row in _ROWS if row.get("modality") in (None, "chat")
    ]
    assert ids == chat_or_untagged
    assert seen[0].full_url == routerbase.base_url.rstrip("/") + "/models"
    assert seen[0].get_header("Authorization") == "Bearer sk-rb-test"


def test_non_chat_rows_are_remembered_as_non_chat(routerbase, monkeypatch):
    from hermes_cli.chat_catalog import is_known_non_chat_model

    _serve(monkeypatch, _CATALOG)
    routerbase.fetch_models()

    for row in _ROWS:
        expected = row.get("modality") not in (None, "chat")
        assert is_known_non_chat_model(row["id"]) is expected


def test_fetch_models_honours_custom_base_url(routerbase, monkeypatch):
    seen = []
    _serve(monkeypatch, _CATALOG, seen)

    routerbase.fetch_models(base_url="https://proxy.example.com/v1/")

    assert seen[0].full_url == "https://proxy.example.com/v1/models"


def test_fetch_failure_returns_none(routerbase, monkeypatch):
    _serve(monkeypatch, OSError("offline"))

    assert routerbase.fetch_models() is None


def test_context_length_comes_from_catalog_and_fails_open(routerbase, monkeypatch):
    _serve(monkeypatch, _CATALOG)

    by_id = {row["id"]: row for row in _ROWS}
    for model, row in by_id.items():
        expected = row.get("context_length") if row.get("modality") in (None, "chat") else None
        assert routerbase.get_model_context_length(model) == expected


def test_context_length_probe_is_not_repeated_after_failure(routerbase, monkeypatch):
    seen = []
    _serve(monkeypatch, OSError("offline"), seen)

    assert routerbase.get_model_context_length("openai/gpt-5.4") is None
    assert routerbase.get_model_context_length("openai/gpt-5.4") is None
    assert len(seen) == 1


def test_context_length_reaches_model_metadata(routerbase, monkeypatch):
    from agent.model_metadata import get_model_context_length

    _serve(monkeypatch, _CATALOG)

    ctx = get_model_context_length(
        "deepseek/deepseek-v4.1-flash", base_url=routerbase.base_url, provider="routerbase",
    )
    assert ctx == next(row["context_length"] for row in _ROWS if row["id"] == "deepseek/deepseek-v4.1-flash")


def test_hermes_cli_resolves_routerbase_credentials(routerbase, monkeypatch):
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials, resolve_provider
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.models import CANONICAL_PROVIDERS

    key_var, url_var = routerbase.env_vars
    monkeypatch.setenv(key_var, "sk-rb-secret")
    monkeypatch.setenv(url_var, "https://proxy.example.com/v1")

    for alias in (routerbase.name, *routerbase.aliases):
        assert resolve_provider(alias) == routerbase.name
    assert PROVIDER_REGISTRY[routerbase.name].base_url_env_var == url_var
    creds = resolve_api_key_provider_credentials(routerbase.name)
    assert creds["api_key"] == "sk-rb-secret"
    assert creds["base_url"] == "https://proxy.example.com/v1"
    assert routerbase.name in {p.slug for p in CANONICAL_PROVIDERS}
    assert OPTIONAL_ENV_VARS[key_var]["password"] is True
    assert url_var in OPTIONAL_ENV_VARS


def test_picker_lists_chat_models_only(routerbase, monkeypatch):
    from hermes_cli.models import provider_model_ids

    monkeypatch.setenv(routerbase.env_vars[0], "sk-rb-secret")
    monkeypatch.delenv(routerbase.env_vars[1], raising=False)
    _serve(monkeypatch, _CATALOG)

    ids = provider_model_ids(routerbase.name)

    non_chat = {row["id"] for row in _ROWS if row.get("modality") not in (None, "chat")}
    assert ids and not (set(ids) & non_chat)
