import json
from pathlib import Path

from talgin.agent import deterministic_agronomist_brief, verified_context


def test_verified_demo_detects_anomaly():
    payload = json.loads(Path("demo/sample_farm.json").read_text(encoding="utf-8"))
    context = verified_context(payload)
    brief = deterministic_agronomist_brief(context)
    reasons = {item["reason"] for item in brief["alerts"]}
    assert "vegetation_index_drop" in reasons
    assert "dry_window" in reasons
    assert brief["requires_human_confirmation"] is True
