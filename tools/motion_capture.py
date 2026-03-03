"""
MotionCaptureTool – ingest biomechanical / motion capture data.
"""
from __future__ import annotations
from typing import Any

from .base_tool import BaseTool


class MotionCaptureTool(BaseTool):
    name = "ingest_motion_capture"
    description = (
        "Ingest motion capture or video-AI biomechanical data for a session. "
        "Flags asymmetries and movement-pattern anomalies."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "session_id": {"type": "string"},
            "system": {
                "type": "string",
                "description": "Capture system used, e.g. 'Vicon', 'IMU', 'video-AI'",
            },
            "joint_angles": {
                "type": "object",
                "description": "Map of joint name → peak angle (degrees)",
            },
            "ground_reaction_forces": {
                "type": "object",
                "description": "Map of force component → value in Newtons",
            },
            "symmetry_index": {
                "type": "number",
                "description": "0–1; 1 = perfect bilateral symmetry",
            },
            "anomalies": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["athlete_id", "session_id", "system"],
    }

    _SYMMETRY_ALERT_THRESHOLD = 0.85

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        si = tool_input.get("symmetry_index", 1.0)
        alerts = list(tool_input.get("anomalies", []))
        if si < self._SYMMETRY_ALERT_THRESHOLD:
            alerts.append(
                f"Low symmetry index ({si:.2f}) – consider asymmetry screening"
            )
        return {
            "status": "ok",
            "athlete_id": tool_input["athlete_id"],
            "session_id": tool_input["session_id"],
            "system": tool_input["system"],
            "joint_angles": tool_input.get("joint_angles", {}),
            "ground_reaction_forces": tool_input.get("ground_reaction_forces", {}),
            "symmetry_index": si,
            "alerts": alerts,
        }
