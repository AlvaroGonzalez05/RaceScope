"""Tests for OpenF1Client — rate limiting, caching, auth headers, sync wrapper."""
import asyncio
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

# Make sure we can import from app
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.openf1_client import OpenF1Client
from app.config import OPENF1_RATE_PER_SECOND, OPENF1_RATE_PER_MINUTE


# ── helpers ──────────────────────────────────────────────────────────────────

def make_client(tmp_path, auth_enabled=False):
    client = OpenF1Client(
        base_url="https://api.openf1.org/v1",
        cache_dir=tmp_path,
        timeout=5,
        min_interval=0.0,   # disable burst floor for unit tests
        max_retries=3,
        backoff_base=0.01,
    )
    client.auth_enabled = auth_enabled
    return client


# ── token bucket ─────────────────────────────────────────────────────────────

class TestTokenBucket:
    """Token-bucket logic — does not hit network."""

    @pytest.mark.asyncio
    async def test_minute_tokens_decrease_on_acquire(self, tmp_path):
        client = make_client(tmp_path)
        initial = client._minute_tokens
        await client._acquire_rate_token()
        assert client._minute_tokens == pytest.approx(initial - 1.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self, tmp_path):
        client = make_client(tmp_path)
        client._minute_tokens = 0.5   # below 1 → should sleep
        client._minute_refill_time = time.monotonic() - 0.5  # 0.5 s elapsed
        # At refill_rate = 60/60 = 1 token/s, 0.5 s elapsed = 0.5 tokens refilled
        # total = 1.0 tokens → no wait needed
        start = time.monotonic()
        await client._acquire_rate_token()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, "Should not sleep long when tokens cover the request"

    @pytest.mark.asyncio
    async def test_tokens_depleted_causes_sleep(self, tmp_path):
        client = make_client(tmp_path)
        client._minute_tokens = 0.0
        client._minute_refill_time = time.monotonic()
        # With 0 tokens, need to wait 1/refill_rate ≈ 1s but refill_rate=1 so wait=1s
        # In test, we just confirm it awaits (not that it waits exactly 1s — use mock)
        slept = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            slept.append(duration)
            # Actually sleep a tiny amount so loop doesn't hang
            await original_sleep(0.001)

        with patch("app.openf1_client.asyncio.sleep", side_effect=mock_sleep):
            await client._acquire_rate_token()

        assert len(slept) >= 1, "Should have slept when tokens are exhausted"
        assert slept[0] > 0, "Sleep duration should be positive"

    def test_rate_constants_loaded(self, tmp_path):
        client = make_client(tmp_path)
        assert client._rate_per_second == OPENF1_RATE_PER_SECOND
        assert client._rate_per_minute == OPENF1_RATE_PER_MINUTE


# ── cache key uses sha256 ─────────────────────────────────────────────────────

class TestCacheKey:
    def test_cache_path_uses_sha256(self, tmp_path):
        client = make_client(tmp_path)
        path = client._cache_path("sessions", {"year": 2023})
        # sha256 hex is 64 chars; we truncate to 32
        assert len(path.stem) == 32
        # Confirm it's a valid hex string
        int(path.stem, 16)

    def test_cache_path_deterministic(self, tmp_path):
        client = make_client(tmp_path)
        p1 = client._cache_path("sessions", {"year": 2023})
        p2 = client._cache_path("sessions", {"year": 2023})
        assert p1 == p2

    def test_cache_path_differs_for_different_params(self, tmp_path):
        client = make_client(tmp_path)
        p1 = client._cache_path("sessions", {"year": 2023})
        p2 = client._cache_path("sessions", {"year": 2024})
        assert p1 != p2


# ── get() — mock httpx ────────────────────────────────────────────────────────

class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_json_data(self, tmp_path):
        client = make_client(tmp_path)
        fake_data = [{"session_key": 1, "year": 2023}]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_data
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with patch("app.openf1_client.httpx.AsyncClient", return_value=mock_async_client):
            result = await client.get("sessions", {"year": 2023}, use_cache=False)

        assert result == fake_data

    @pytest.mark.asyncio
    async def test_get_uses_cache(self, tmp_path):
        client = make_client(tmp_path)
        fake_data = [{"session_key": 99}]

        # Pre-populate cache
        cache_path = client._cache_path("sessions", {"year": 2023})
        cache_path.write_text(json.dumps(fake_data))

        # Should return cached data without hitting network
        result = await client.get("sessions", {"year": 2023}, use_cache=True)
        assert result == fake_data

    @pytest.mark.asyncio
    async def test_get_retries_on_429(self, tmp_path):
        client = make_client(tmp_path)
        client.max_retries = 3

        call_count = 0

        # First two calls return 429, third returns 200
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {}
        rate_limited.raise_for_status = MagicMock()

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = [{"ok": True}]
        success.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return rate_limited if call_count < 3 else success

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_instance.get = mock_get

        with patch("app.openf1_client.httpx.AsyncClient", return_value=mock_client_instance):
            with patch("app.openf1_client.asyncio.sleep", AsyncMock()):
                result = await client.get("sessions", {}, use_cache=False)

        assert result == [{"ok": True}]
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_get_raises_after_all_retries_fail(self, tmp_path):
        client = make_client(tmp_path)
        client.max_retries = 2

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_instance.get = AsyncMock(side_effect=httpx.RequestError("timeout", request=MagicMock()))

        with patch("app.openf1_client.httpx.AsyncClient", return_value=mock_client_instance):
            with patch("app.openf1_client.asyncio.sleep", AsyncMock()):
                with pytest.raises(RuntimeError, match="OpenF1 request failed"):
                    await client.get("sessions", {}, use_cache=False)


# ── get_sync wrapper ──────────────────────────────────────────────────────────

class TestGetSync:
    def test_get_sync_calls_async_get(self, tmp_path):
        client = make_client(tmp_path)
        fake_data = [{"session_key": 1}]

        async def fake_get(endpoint, params=None, use_cache=True):
            return fake_data

        client.get = fake_get
        result = client.get_sync("sessions", {"year": 2023}, use_cache=False)
        assert result == fake_data
