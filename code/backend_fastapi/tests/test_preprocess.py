"""Tests for preprocess.py — circuit_id canonical mapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app.preprocess import CIRCUIT_ID_CANONICAL


class TestCircuitIdCanonical:
    @pytest.mark.parametrize("raw,expected", [
        ("bahrain", "Sakhir"),
        ("sakhir", "Sakhir"),
        ("jeddah", "Jeddah"),
        ("albert park", "Melbourne"),
        ("melbourne", "Melbourne"),
        ("suzuka", "Suzuka"),
        ("baku", "Baku"),
        ("miami", "Miami"),
        ("monte carlo", "Monaco"),
        ("monaco", "Monaco"),
        ("barcelona", "Barcelona"),
        ("montmelo", "Barcelona"),
        ("red bull ring", "Spielberg"),
        ("spielberg", "Spielberg"),
        ("silverstone", "Silverstone"),
        ("hungaroring", "Budapest"),
        ("budapest", "Budapest"),
        ("spa", "Spa"),
        ("spa-francorchamps", "Spa"),
        ("monza", "Monza"),
        ("marina bay", "Singapore"),
        ("singapore", "Singapore"),
        ("lusail", "Lusail"),
        ("qatar", "Lusail"),
        ("cota", "Austin"),
        ("austin", "Austin"),
        ("rodriguez", "Mexico City"),
        ("mexico city", "Mexico City"),
        ("interlagos", "São Paulo"),
        ("las vegas", "Las Vegas"),
        ("yas marina", "Abu Dhabi"),
        ("abu dhabi", "Abu Dhabi"),
        ("imola", "Imola"),
        ("zandvoort", "Zandvoort"),
    ])
    def test_known_circuits(self, raw, expected):
        assert CIRCUIT_ID_CANONICAL.get(raw) == expected, f"Expected {raw!r} -> {expected!r}"

    def test_all_keys_are_lowercase(self):
        for key in CIRCUIT_ID_CANONICAL:
            assert key == key.lower(), f"Key {key!r} is not lowercase"

    def test_all_values_are_title_case(self):
        for key, value in CIRCUIT_ID_CANONICAL.items():
            # Values should be non-empty strings (may contain spaces and special chars)
            assert isinstance(value, str) and len(value) > 0

    def test_unknown_circuit_returns_none(self):
        # Unknown circuits should not be in the map
        assert CIRCUIT_ID_CANONICAL.get("unknown_circuit_xyz") is None
