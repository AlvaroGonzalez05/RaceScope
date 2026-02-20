from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config import (
    OPENF1_BASE_URL,
    CACHE_DIR,
    CACHE_TTL_SECONDS,
    OPENF1_MIN_INTERVAL,
    OPENF1_MAX_RETRIES,
    OPENF1_BACKOFF_BASE,
)


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
        self._last_request = 0.0

    def _cache_path(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Path:
        payload = json.dumps({"endpoint": endpoint, "params": params or {}}, sort_keys=True)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> list[dict]:
        endpoint = endpoint.lstrip("/")
        cache_path = self._cache_path(endpoint, params)
        if use_cache and cache_path.exists():
            if (time.time() - cache_path.stat().st_mtime) < CACHE_TTL_SECONDS:
                return json.loads(cache_path.read_text())

        url = f"{self.base_url}/{endpoint}"
        data = None
        for attempt in range(self.max_retries):
            now = time.time()
            wait = self.min_interval - (now - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                self._last_request = time.time()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        time.sleep(float(retry_after))
                    else:
                        time.sleep(self.backoff_base ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException:
                time.sleep(self.backoff_base ** attempt)
                continue

        if data is None:
            raise RuntimeError(f"OpenF1 request failed for {endpoint} with params {params}")

        if use_cache:
            cache_path.write_text(json.dumps(data))

        return data
