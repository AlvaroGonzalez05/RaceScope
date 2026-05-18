"""Tests for Phase 2 BE-API changes in the RaceScope FastAPI backend."""
from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from cachetools import TTLCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient with data-store functions mocked so tests don't
    need real parquet files on disk."""
    with (
        # Patch at app.main.* so the already-imported references are replaced.
        # Patching app.data_store.* alone doesn't work because main.py binds
        # these names at import time and the module may already be in sys.modules.
        patch("app.main.load_features") as mock_load_features,
        patch("app.main.seasons_available") as mock_seasons,
        patch("app.main.metadata_for_year") as mock_meta,
        patch("app.main.load_snapshot_state") as mock_snapshot,
    ):
        import pandas as pd

        # Seasons available: only 2023
        mock_seasons.return_value = [2023]

        # features: empty DataFrame (strategies won't run but metadata will)
        mock_load_features.return_value = pd.DataFrame()

        # metadata for 2023
        mock_meta.return_value = {
            "circuits": pd.DataFrame({"circuit_id": ["Sakhir", "Monaco"]}),
            "drivers": pd.DataFrame(
                {
                    "driver_id": [1, 14, 44],
                    "driver_code": ["VER", "ALO", "HAM"],
                    "team_name": ["Red Bull", "Aston Martin", "Mercedes"],
                }
            ),
            "teams": pd.DataFrame({"team_name": ["Red Bull", "Aston Martin", "Mercedes"]}),
        }

        mock_snapshot.return_value = {
            "last_successful_ingest_at": None,
            "years_available": [2023],
            "stale_mode": False,
            "last_error": None,
        }

        # Import app AFTER patches are in place
        from app.main import app

        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc


# ---------------------------------------------------------------------------
# A. Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_context_out_valid(self):
        from app.schemas import ContextOut

        ctx = ContextOut(
            total_laps=57,
            track_temp=35.0,
            air_temp=28.0,
            pit_loss=22.5,
            sc_probability=0.2,
        )
        assert ctx.total_laps == 57
        assert ctx.pit_loss == 22.5

    def test_context_out_extra_fields_allowed(self):
        from app.schemas import ContextOut

        ctx = ContextOut(
            total_laps=57,
            track_temp=35.0,
            air_temp=28.0,
            pit_loss=22.5,
            sc_probability=0.2,
            extra_key="extra_value",
        )
        assert ctx.extra_key == "extra_value"

    def test_strategy_out_valid(self):
        from app.schemas import StrategyOut, StintCurve, PitWindow

        strat = StrategyOut(
            strategy_id="s1",
            type="1-stop",
            compounds=["SOFT", "MEDIUM"],
            stints=[20, 37],
            stint_curves=[
                StintCurve(
                    compound="SOFT",
                    start_lap=1,
                    end_lap=20,
                    lap_time_data=[90.5, 91.0],
                    tyre_life_data=[1.0, 2.0],
                )
            ],
            pit_windows=[PitWindow(lap_min=18, lap_max=22)],
            stop_laps=[20],
            expected_time=3600.0,
            variance=5.0,
            risk_score=0.3,
        )
        assert strat.strategy_id == "s1"
        assert strat.type == "1-stop"

    def test_strategy_out_extra_fields_allowed(self):
        from app.schemas import StrategyOut, StintCurve, PitWindow

        strat = StrategyOut(
            strategy_id="s1",
            type="1-stop",
            compounds=["SOFT"],
            stints=[57],
            stint_curves=[],
            pit_windows=[],
            stop_laps=[],
            expected_time=3600.0,
            variance=5.0,
            risk_score=0.3,
            custom_field="hello",
        )
        assert strat.custom_field == "hello"

    def test_strategy_response_extra_allowed(self):
        from app.schemas import StrategyResponse, ContextOut

        ctx = ContextOut(
            total_laps=57,
            track_temp=35.0,
            air_temp=28.0,
            pit_loss=22.5,
            sc_probability=0.2,
        )
        resp = StrategyResponse(
            context=ctx,
            strategies=[],
            degradation={},
            extra_top_level="extra",
        )
        assert resp.extra_top_level == "extra"

    def test_compare_response_extra_allowed(self):
        from app.schemas import CompareResponse, StrategyResponse, ContextOut

        ctx = ContextOut(
            total_laps=57,
            track_temp=35.0,
            air_temp=28.0,
            pit_loss=22.5,
            sc_probability=0.2,
        )
        sr = StrategyResponse(context=ctx, strategies=[], degradation={})
        resp = CompareResponse(driver_a=sr, driver_b=sr, bonus_key=99)
        assert resp.bonus_key == 99

    def test_driver_out_valid(self):
        from app.schemas import DriverOut

        driver = DriverOut(driver_id=14, driver_code="ALO", team_name="Aston Martin")
        assert driver.driver_id == 14
        assert driver.driver_code == "ALO"

    def test_driver_out_optional_fields(self):
        from app.schemas import DriverOut

        driver = DriverOut(driver_id=14)
        assert driver.driver_code is None
        assert driver.team_name is None

    def test_driver_out_extra_allowed(self):
        from app.schemas import DriverOut

        driver = DriverOut(driver_id=14, driver_code="ALO", extra="x")
        assert driver.extra == "x"


# ---------------------------------------------------------------------------
# B. Season 404 validation
# ---------------------------------------------------------------------------

class TestSeason404:
    def test_circuits_unknown_season_returns_404(self, client):
        resp = client.get("/api/metadata/circuits?season=1900")
        assert resp.status_code == 404
        assert "1900" in resp.json()["detail"]

    def test_circuits_404_detail_message(self, client):
        resp = client.get("/api/metadata/circuits?season=1900")
        detail = resp.json()["detail"]
        assert "Season 1900 not found" in detail

    def test_drivers_unknown_season_returns_404(self, client):
        resp = client.get("/api/metadata/drivers?season=1900")
        assert resp.status_code == 404

    def test_teams_unknown_season_returns_404(self, client):
        resp = client.get("/api/metadata/teams?season=1900")
        assert resp.status_code == 404

    def test_circuits_valid_season_not_404(self, client):
        resp = client.get("/api/metadata/circuits?season=2023")
        # Should succeed (200) — data is mocked
        assert resp.status_code != 404

    def test_circuits_valid_season_returns_200(self, client):
        resp = client.get("/api/metadata/circuits?season=2023")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# C. X-Request-ID middleware
# ---------------------------------------------------------------------------

class TestRequestIDMiddleware:
    def test_seasons_has_request_id_header(self, client):
        resp = client.get("/api/metadata/seasons")
        assert "x-request-id" in resp.headers

    def test_circuits_has_request_id_header(self, client):
        resp = client.get("/api/metadata/circuits?season=2023")
        assert "x-request-id" in resp.headers

    def test_request_id_is_non_empty(self, client):
        resp = client.get("/api/metadata/seasons")
        assert len(resp.headers["x-request-id"]) > 0

    def test_request_id_is_string(self, client):
        resp = client.get("/api/metadata/seasons")
        assert isinstance(resp.headers["x-request-id"], str)

    def test_two_requests_get_different_ids(self, client):
        resp1 = client.get("/api/metadata/seasons")
        resp2 = client.get("/api/metadata/seasons")
        id1 = resp1.headers["x-request-id"]
        id2 = resp2.headers["x-request-id"]
        assert id1 != id2, "Two consecutive requests should produce different request IDs"

    def test_request_id_on_404(self, client):
        resp = client.get("/api/metadata/circuits?season=1900")
        assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# D. CORS header
# ---------------------------------------------------------------------------

class TestCORS:
    def test_cors_origin_reflected_for_allowed_origin(self, client):
        resp = client.get(
            "/api/metadata/seasons",
            headers={"Origin": "http://localhost:5173"},
        )
        assert resp.status_code == 200
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:5173", (
            f"Expected ACAO header to be 'http://localhost:5173', got '{acao}'"
        )

    def test_cors_preflight_returns_200_for_allowed_origin(self, client):
        resp = client.options(
            "/api/metadata/seasons",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 204)


# ---------------------------------------------------------------------------
# E. TTLCache
# ---------------------------------------------------------------------------

class TestTTLCache:
    def test_cache_is_ttlcache_instance(self):
        from app.main import _cache

        assert isinstance(_cache, TTLCache), (
            f"Expected _cache to be TTLCache, got {type(_cache)}"
        )

    def test_cache_maxsize_is_128(self):
        from app.main import _cache

        assert _cache.maxsize == 128

    def test_cache_ttl_matches_config(self):
        from app.main import _cache
        from app.config import CACHE_TTL_SECONDS

        assert _cache.ttl == CACHE_TTL_SECONDS

    def test_cache_set_and_get(self):
        from app.main import _cache

        _cache["test_key"] = {"value": 42}
        assert _cache.get("test_key") == {"value": 42}
        del _cache["test_key"]

    def test_cache_miss_returns_none(self):
        from app.main import _cache

        result = _cache.get("nonexistent_key_xyz")
        assert result is None


# ---------------------------------------------------------------------------
# F. Legacy route deprecation warnings
# ---------------------------------------------------------------------------

class TestLegacyRouteDeprecation:
    def test_legacy_metadata_seasons_logs_deprecation(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            resp = client.get("/metadata/seasons")
        assert resp.status_code == 200
        deprecation_logged = any(
            "Deprecated" in record.message or "deprecated" in record.message.lower()
            for record in caplog.records
        )
        assert deprecation_logged, (
            f"Expected a deprecation warning in logs. Records: {[r.message for r in caplog.records]}"
        )

    def test_legacy_metadata_circuits_logs_deprecation(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            resp = client.get("/metadata/circuits?season=2023")
        assert resp.status_code == 200
        assert any(
            "Deprecated" in record.message or "deprecated" in record.message.lower()
            for record in caplog.records
        )

    def test_legacy_metadata_drivers_logs_deprecation(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            resp = client.get("/metadata/drivers?season=2023")
        assert resp.status_code == 200
        assert any(
            "Deprecated" in record.message or "deprecated" in record.message.lower()
            for record in caplog.records
        )

    def test_legacy_metadata_teams_logs_deprecation(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            resp = client.get("/metadata/teams?season=2023")
        assert resp.status_code == 200
        assert any(
            "Deprecated" in record.message or "deprecated" in record.message.lower()
            for record in caplog.records
        )

    def test_legacy_post_strategy_exists(self, client):
        """Verify the legacy POST /strategy route exists (returns 400 due to no data, not 404/405)."""
        resp = client.post(
            "/strategy",
            json={
                "year": 2023,
                "circuit_id": "Sakhir",
                "driver_id": 14,
            },
        )
        # Route exists — will be 400 (no features) not 404 (not found)
        assert resp.status_code != 404, "Legacy /strategy route should exist"
        assert resp.status_code != 405, "Legacy /strategy route should accept POST"

    def test_legacy_post_strategy_logs_deprecation(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            client.post(
                "/strategy",
                json={
                    "year": 2023,
                    "circuit_id": "Sakhir",
                    "driver_id": 14,
                },
            )
        assert any(
            "Deprecated" in record.message or "deprecated" in record.message.lower()
            for record in caplog.records
        )
