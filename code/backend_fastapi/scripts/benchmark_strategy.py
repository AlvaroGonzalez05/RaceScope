from __future__ import annotations

import json
import statistics
import time
from typing import Dict, List

import requests

BASE = "http://localhost:8000"
PAYLOAD = {"year": 2023, "circuit_id": "Sakhir", "driver_id": 14}


def call_strategy(n: int) -> Dict[str, List[float]]:
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/strategy", json=PAYLOAD, timeout=120)
        r.raise_for_status()
        _ = r.json()
        timings.append((time.perf_counter() - t0) * 1000)
    return timings


def summary(label: str, timings: List[float]) -> Dict[str, float]:
    return {
        "label": label,
        "count": len(timings),
        "mean_ms": statistics.mean(timings),
        "p50_ms": statistics.median(timings),
        "p90_ms": statistics.quantiles(timings, n=10)[8] if len(timings) >= 10 else max(timings),
        "p95_ms": statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main() -> None:
    # Cold run (clear caches by first call)
    cold = call_strategy(1)
    warm = call_strategy(10)
    hot = call_strategy(30)

    report = {
        "payload": PAYLOAD,
        "cold": summary("cold", cold),
        "warm": summary("warm", warm),
        "hot": summary("hot", hot),
        "raw": {
            "cold_ms": cold,
            "warm_ms": warm,
            "hot_ms": hot,
        },
    }

    path = "/Users/alvarogonzaleztabernero/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/ICAI/5/TFG BA/code/backend_fastapi/benchmark_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
