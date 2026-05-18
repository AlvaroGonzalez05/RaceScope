from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .config import (
    OPENF1_BASE_URL,
    CACHE_DIR,
    CACHE_TTL_SECONDS,
    OPENF1_MIN_INTERVAL,
    OPENF1_MAX_RETRIES,
    OPENF1_BACKOFF_BASE,
    OPENF1_AUTH_ENABLED,
    OPENF1_USERNAME,
    OPENF1_PASSWORD,
    OPENF1_RATE_PER_SECOND,
    OPENF1_RATE_PER_MINUTE,
)

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry (transient server / rate-limit errors).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Hard cap on any single backoff sleep (seconds).
_BACKOFF_CAP = 60.0


def _jittered_backoff(attempt: int, base: float, cap: float = _BACKOFF_CAP) -> float:
    """Full-jitter exponential backoff.

    Returns a random value in [0, min(cap, base ** attempt)].
    Full jitter avoids synchronised thundering-herd retries when multiple
    processes (or repeated script runs) hit the same 429 window.

    Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """
    ceiling = min(cap, base ** attempt)
    return random.uniform(0, ceiling)


def _parse_retry_after(header_value: str | None) -> float | None:
    """Parse Retry-After header.  Returns seconds to wait, or None if unparseable."""
    if header_value is None:
        return None
    try:
        return max(0.0, float(header_value))
    except ValueError:
        # Could be an HTTP-date; fall back to None and let caller use backoff.
        return None


class OpenF1Client:
    def __init__(
        self,
        base_url: str = OPENF1_BASE_URL,
        cache_dir: Path = CACHE_DIR,
        timeout: int = 30,
        min_interval: float = OPENF1_MIN_INTERVAL,
        max_retries: int = OPENF1_MAX_RETRIES,
        backoff_base: float = OPENF1_BACKOFF_BASE,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # Use monotonic clock throughout to avoid wall-clock jumps (NTP, DST).
        self._last_request: float = 0.0           # monotonic
        self._minute_tokens: float = float(OPENF1_RATE_PER_MINUTE)
        self._minute_refill_time: float = time.monotonic()

        self.auth_enabled = OPENF1_AUTH_ENABLED
        self.username = OPENF1_USERNAME
        self.password = OPENF1_PASSWORD
        self._access_token = ""
        self._token_exp_mono: float = 0.0         # monotonic expiry
        self._auth_url = self.base_url.rsplit("/v1", 1)[0] + "/token"
        self._rate_per_second: int = OPENF1_RATE_PER_SECOND
        self._rate_per_minute: int = OPENF1_RATE_PER_MINUTE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Path:
        payload = json.dumps({"endpoint": endpoint, "params": params or {}}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _token_expired(self) -> bool:
        if not self._access_token:
            return True
        return time.monotonic() >= self._token_exp_mono

    async def _fetch_token(self) -> None:
        if not self.auth_enabled:
            return
        if not self.username or not self.password:
            raise RuntimeError(
                "OpenF1 auth enabled but OPENF1_USERNAME/OPENF1_PASSWORD not configured."
            )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._auth_url,
                data={"username": self.username, "password": self.password},
            )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))
        if not token:
            raise RuntimeError("OpenF1 token response missing access_token.")
        self._access_token = token
        # Subtract 30 s as safety margin; monotonic from here.
        self._token_exp_mono = time.monotonic() + max(60, expires_in - 30)
        logger.debug("OpenF1 auth token refreshed, valid for ~%ds", expires_in)

    async def _auth_headers(self) -> dict[str, str]:
        if not self.auth_enabled:
            return {}
        if self._token_expired():
            await self._fetch_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _acquire_rate_token(self) -> None:
        """Block until both rate limits allow the next request.

        Two independent controls:
        - Sustained: token bucket at `rate_per_minute` tokens/min.
        - Burst floor: hard minimum gap of `min_interval` seconds.
        """
        now = time.monotonic()

        # --- sustained rate: token bucket (60 req/min) ---
        elapsed = now - self._minute_refill_time
        refill_rate = self._rate_per_minute / 60.0          # tokens per second
        self._minute_tokens = min(
            float(self._rate_per_minute),
            self._minute_tokens + elapsed * refill_rate,
        )
        self._minute_refill_time = now

        if self._minute_tokens < 1.0:
            wait = (1.0 - self._minute_tokens) / refill_rate
            if wait > 1.0:
                logger.info(
                    "Rate bucket exhausted — waiting %.1fs before next request", wait
                )
            await asyncio.sleep(wait)
            self._minute_tokens = 0.0
        else:
            self._minute_tokens -= 1.0

        # --- burst floor: enforce min gap between consecutive calls ---
        burst_wait = self.min_interval - (time.monotonic() - self._last_request)
        if burst_wait > 0:
            await asyncio.sleep(burst_wait)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, endpoint: str, params=None, use_cache: bool = True) -> list[dict]:
        endpoint = endpoint.lstrip("/")
        cache_path = self._cache_path(endpoint, params)
        if use_cache and cache_path.exists():
            if (time.monotonic() - cache_path.stat().st_mtime) < CACHE_TTL_SECONDS:
                return json.loads(cache_path.read_text())

        url = f"{self.base_url}/{endpoint}"
        data: list[dict] | None = None
        last_error: str | None = None
        refreshed_once = False

        for attempt in range(self.max_retries):
            await self._acquire_rate_token()
            try:
                headers = await self._auth_headers()
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, params=params, headers=headers)
                self._last_request = time.monotonic()

                status = response.status_code

                # --- 401: try one token refresh ---
                if status == 401 and self.auth_enabled and not refreshed_once:
                    logger.warning("OpenF1 401 — refreshing token (attempt %d)", attempt)
                    await self._fetch_token()
                    refreshed_once = True
                    last_error = "unauthorized_refreshed"
                    continue

                # --- retryable: 429 and transient 5xx ---
                if status in _RETRYABLE_STATUS:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    backoff = _jittered_backoff(attempt, self.backoff_base)
                    wait = max(retry_after or 0.0, backoff)
                    logger.warning(
                        "OpenF1 %d for %s (attempt %d/%d) — retrying in %.1fs",
                        status, endpoint, attempt + 1, self.max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    last_error = f"http_{status}"
                    continue

                # Non-retryable HTTP error (e.g. 404, 400, 403).
                # Convert to RuntimeError so callers don't need to import httpx.
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as http_exc:
                    raise RuntimeError(
                        f"OpenF1 HTTP {response.status_code} for {endpoint} "
                        f"params={params}: {http_exc}"
                    ) from http_exc
                data = response.json()
                break

            except httpx.RequestError as exc:
                # Network-level error (timeout, DNS, connection reset, …)
                wait = _jittered_backoff(attempt, self.backoff_base)
                logger.warning(
                    "OpenF1 network error for %s (attempt %d/%d) — retrying in %.1fs: %s",
                    endpoint, attempt + 1, self.max_retries, wait, exc,
                )
                last_error = str(exc)
                await asyncio.sleep(wait)
                continue

        if data is None:
            raise RuntimeError(
                f"OpenF1 request failed: endpoint={endpoint} params={params} "
                f"after {self.max_retries} attempts — last_error={last_error or 'unknown'}"
            )

        if use_cache:
            cache_path.write_text(json.dumps(data))

        return data

    def get_sync(self, endpoint: str, params=None, use_cache: bool = True) -> list[dict]:
        """Synchronous wrapper — CLI scripts only.  Do NOT call from async handlers."""
        return asyncio.run(self.get(endpoint, params, use_cache))
