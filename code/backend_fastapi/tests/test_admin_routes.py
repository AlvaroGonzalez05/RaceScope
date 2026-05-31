"""Tests for admin API routes: POST /api/admin/ingest and GET /api/admin/ingest/status.

Security tests verify that:
- Admin routes require a valid X-Admin-Key header.
- Missing or wrong key returns 403.
- Correct key grants access.
- Rate-limited public routes reject excessive callers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_TEST_KEY = "test-admin-key-abc123"
_VALID_HEADERS = {"X-Admin-Key": _TEST_KEY}


# ---------------------------------------------------------------------------
# Admin key security
# ---------------------------------------------------------------------------

class TestAdminKeyRequired:
    def test_ingest_without_key_returns_403(self):
        """No key → 403, even with a valid body."""
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY):
            response = client.post("/api/admin/ingest", json={"year": 2023})
        assert response.status_code == 403

    def test_ingest_wrong_key_returns_403(self):
        """Wrong key → 403."""
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY):
            response = client.post(
                "/api/admin/ingest",
                json={"year": 2023},
                headers={"X-Admin-Key": "wrong-key"},
            )
        assert response.status_code == 403

    def test_ingest_correct_key_returns_200(self):
        """Correct key + valid body → 200."""
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY), patch("app.main._run_ingest"):
            response = client.post(
                "/api/admin/ingest",
                json={"year": 2023},
                headers=_VALID_HEADERS,
            )
        assert response.status_code == 200
        assert response.json()["status"] == "ingestion_started"

    def test_status_without_key_returns_403(self):
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY):
            response = client.get("/api/admin/ingest/status")
        assert response.status_code == 403

    def test_status_correct_key_returns_200(self):
        fake_state = {"stale_mode": False, "last_updated": "2023-01-01", "last_error": None}
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY), \
             patch("app.main.load_snapshot_state", return_value=fake_state):
            response = client.get("/api/admin/ingest/status", headers=_VALID_HEADERS)
        assert response.status_code == 200
        assert "stale_mode" in response.json()

    def test_admin_disabled_when_key_not_configured(self):
        """ADMIN_API_KEY='' → 403 with 'disabled' message."""
        with patch("app.main.ADMIN_API_KEY", ""):
            response = client.post(
                "/api/admin/ingest",
                json={"year": 2023},
                headers={"X-Admin-Key": "anything"},
            )
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Ingest payload validation (requires valid key)
# ---------------------------------------------------------------------------

class TestAdminIngestValidation:
    def test_invalid_year_too_low_returns_422(self):
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY):
            response = client.post(
                "/api/admin/ingest", json={"year": 2000}, headers=_VALID_HEADERS
            )
        assert response.status_code == 422

    def test_invalid_year_too_high_returns_422(self):
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY):
            response = client.post(
                "/api/admin/ingest", json={"year": 2050}, headers=_VALID_HEADERS
            )
        assert response.status_code == 422

    def test_missing_year_returns_422(self):
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY):
            response = client.post(
                "/api/admin/ingest", json={}, headers=_VALID_HEADERS
            )
        assert response.status_code == 422

    def test_year_string_coerced_or_422(self):
        # Pydantic v2 coerces "2023" → 2023, so 200 is acceptable here.
        with patch("app.main.ADMIN_API_KEY", _TEST_KEY), patch("app.main._run_ingest"):
            response = client.post(
                "/api/admin/ingest", json={"year": "2023"}, headers=_VALID_HEADERS
            )
        assert response.status_code in (200, 422)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_strategy_rate_limit_triggers_after_threshold(self):
        """Hammering /api/strategy from same IP should eventually get a 429."""
        from app.main import _rate_limit_store
        _rate_limit_store.clear()

        with patch("app.main.load_features", return_value=__import__("pandas").DataFrame()):
            responses = [
                client.post("/api/strategy", json={
                    "year": 2023, "circuit_id": "Sakhir", "driver_id": 14,
                })
                for _ in range(35)
            ]

        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, "Expected a 429 after rate limit threshold"


# ---------------------------------------------------------------------------
# Smoke tests (public routes unaffected)
# ---------------------------------------------------------------------------

class TestPublicRoutes:
    def test_strategy_route_returns_400_without_features(self):
        """Strategy route is public (no key needed) but returns 400 with no data."""
        from app.main import _rate_limit_store
        _rate_limit_store.clear()  # reset after rate limit test
        with patch("app.main.load_features", return_value=__import__("pandas").DataFrame()):
            response = client.post("/api/strategy", json={
                "year": 2023, "circuit_id": "Sakhir", "driver_code": "VER",
            })
        assert response.status_code == 400


class TestDataStoreInvalidation:
    def test_invalidate_features_cache_callable(self):
        from app.data_store import invalidate_features_cache
        invalidate_features_cache()
