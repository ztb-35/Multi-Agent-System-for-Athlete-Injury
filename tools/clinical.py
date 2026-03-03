"""
ClinicalTool – ingest physician / physiotherapist clinical notes (SOAP format).
"""
from __future__ import annotations
from typing import Any

from .base_tool import BaseTool


class ClinicalTool(BaseTool):
    name = "ingest_clinical_note"
    description = (
        "Ingest a structured SOAP clinical note from a physician, physiotherapist, "
        "or athletic trainer. Returns a normalised summary."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "author": {"type": "string"},
            "role": {
                "type": "string",
                "enum": ["physician", "physio", "athletic_trainer"],
            },
            "date": {"type": "string", "description": "ISO-8601 date"},
            "subjective": {"type": "string", "description": "Athlete-reported symptoms"},
            "objective": {"type": "string", "description": "Measurable examination findings"},
            "assessment": {"type": "string"},
            "plan": {"type": "string"},
        },
        "required": ["athlete_id", "author", "role", "date", "subjective", "objective",
                     "assessment", "plan"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "athlete_id": tool_input["athlete_id"],
            "note": {
                "author": tool_input["author"],
                "role": tool_input["role"],
                "date": tool_input["date"],
                "subjective": tool_input["subjective"],
                "objective": tool_input["objective"],
                "assessment": tool_input["assessment"],
                "plan": tool_input["plan"],
            },
        }
