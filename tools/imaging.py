"""
ImagingTool – parse and structure imaging reports (MRI, X-ray, Ultrasound).
"""
from __future__ import annotations
from typing import Any

from .base_tool import BaseTool


class ImagingTool(BaseTool):
    name = "ingest_imaging_report"
    description = (
        "Parse a radiology or imaging report and extract structured findings. "
        "Supports MRI, X-ray, and Ultrasound modalities."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "modality": {
                "type": "string",
                "enum": ["MRI", "X-ray", "Ultrasound"],
            },
            "body_part": {"type": "string"},
            "date": {"type": "string", "description": "ISO-8601 date"},
            "radiologist_summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of individual findings",
            },
            "severity_grade": {
                "type": "string",
                "enum": ["Normal", "Grade I", "Grade II", "Grade III"],
            },
        },
        "required": ["athlete_id", "modality", "body_part", "date", "radiologist_summary"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        severity = tool_input.get("severity_grade", "Normal")
        urgent = severity in ("Grade II", "Grade III")
        return {
            "status": "ok",
            "athlete_id": tool_input["athlete_id"],
            "modality": tool_input["modality"],
            "body_part": tool_input["body_part"],
            "date": tool_input["date"],
            "radiologist_summary": tool_input["radiologist_summary"],
            "findings": tool_input.get("findings", []),
            "severity_grade": severity,
            "requires_urgent_review": urgent,
        }
