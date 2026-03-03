"""
WearableTool – ingest and parse wearable sensor data.

In production this would connect to a data lake or streaming API
(Catapult, Polar, Garmin, etc.). Here we validate and structure the input.
"""
from __future__ import annotations
from typing import Any

from .base_tool import BaseTool


class WearableTool(BaseTool):
    name = "ingest_wearable_data"
    description = (
        "Ingest and validate wearable sensor data for a training session. "
        "Returns a cleaned, structured summary of physiological and workload metrics."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "session_id": {"type": "string"},
            "avg_heart_rate": {"type": "number", "description": "bpm"},
            "max_heart_rate": {"type": "number", "description": "bpm"},
            "hrv": {"type": "number", "description": "RMSSD in ms"},
            "training_load": {"type": "number", "description": "sRPE × duration (AU)"},
            "distance_km": {"type": "number"},
            "sprint_count": {"type": "integer"},
            "acceleration_count": {"type": "integer"},
            "sleep_hours_prev_night": {"type": "number"},
            "sleep_quality_score": {"type": "number", "description": "0–1"},
        },
        "required": ["athlete_id", "session_id"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        # Validate ranges
        warnings: list[str] = []
        hr = tool_input.get("avg_heart_rate")
        if hr and not (30 <= hr <= 250):
            warnings.append(f"avg_heart_rate {hr} bpm seems out of range")

        hrv = tool_input.get("hrv")
        if hrv and hrv < 0:
            warnings.append("HRV cannot be negative")

        return {
            "status": "ok",
            "athlete_id": tool_input["athlete_id"],
            "session_id": tool_input["session_id"],
            "processed_metrics": {k: v for k, v in tool_input.items()
                                   if k not in ("athlete_id", "session_id")},
            "warnings": warnings,
        }
