from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        info = client.get("/model-info").json()
        demo = client.get("/demo/agronomist").json()
        latencies = []
        generations = []

        if info.get("available"):
            for _ in range(args.requests):
                started = time.perf_counter()
                response = client.post("/generate", json={
                    "prompt": "Привет! Ответь одним коротким предложением.",
                    "max_new_tokens": 32
                })
                response.raise_for_status()
                latencies.append((time.perf_counter() - started) * 1000)
                generations.append(response.json())

    result = {
        "model_info": info,
        "verified_context_demo": demo,
        "requests": len(latencies),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "min": round(min(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "generations": generations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
