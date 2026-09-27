from __future__ import annotations

import hashlib
import json
from typing import Any


ALLOWED_CONTEXT_KEYS = {
    "farm", "field", "crop", "weather", "satellite", "tasks", "observations"
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verified_context(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in ALLOWED_CONTEXT_KEYS if key in payload}


def deterministic_agronomist_brief(context: dict[str, Any]) -> dict[str, Any]:
    field = context.get("field") or {}
    weather = context.get("weather") or {}
    satellite = context.get("satellite") or {}
    tasks = context.get("tasks") or []

    alerts = []
    ndvi_now = satellite.get("ndvi")
    ndvi_prev = satellite.get("ndvi_previous")
    if isinstance(ndvi_now, (int, float)) and isinstance(ndvi_prev, (int, float)):
        drop = ndvi_prev - ndvi_now
        if drop >= 0.10:
            alerts.append({
                "severity": "high",
                "reason": "vegetation_index_drop",
                "message": f"NDVI decreased by {drop:.2f}; field inspection is recommended."
            })

    rain_7d = weather.get("rain_7d_mm")
    forecast_rain = weather.get("forecast_rain_5d_mm")
    if isinstance(rain_7d, (int, float)) and isinstance(forecast_rain, (int, float)):
        if rain_7d < 10 and forecast_rain < 3:
            alerts.append({
                "severity": "medium",
                "reason": "dry_window",
                "message": "Low recent and forecast rainfall; verify soil moisture before irrigation decisions."
            })

    overdue = [task for task in tasks if task.get("status") == "overdue"]
    if overdue:
        alerts.append({
            "severity": "medium",
            "reason": "overdue_tasks",
            "message": f"{len(overdue)} overdue field task(s)."
        })

    return {
        "field": field.get("name"),
        "alerts": alerts,
        "requires_human_confirmation": True,
        "note": "Satellite/weather signals indicate where to inspect; they do not diagnose disease by themselves.",
        "context_sha256": canonical_hash(context),
    }
