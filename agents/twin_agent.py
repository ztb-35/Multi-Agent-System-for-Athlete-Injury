"""
TwinAgent – maintains versioned digital twins for each athlete.

Responsibilities:
  - Listen to NewSessionEvent from the EventBus.
  - Ingest multimodal session data via tools (wearable, imaging, motion capture, clinical).
  - Compute derived features (ACWR, load trends, biomechanical flags).
  - Update and persist the DigitalTwin via MemoryManager.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any

from agents.base_agent import BaseAgent
from events.event_types import NewSessionEvent
from llm.prompts import SystemPrompts
from models.twin import DigitalTwin, TwinSnapshot
from tools.wearable import WearableTool
from tools.imaging import ImagingTool
from tools.motion_capture import MotionCaptureTool
from tools.clinical import ClinicalTool

logger = logging.getLogger(__name__)


class TwinAgent(BaseAgent):
    agent_name = "TwinAgent"

    def __init__(self, llm, memory) -> None:
        super().__init__(llm, memory)
        self._wearable = WearableTool()
        self._imaging = ImagingTool()
        self._motion = MotionCaptureTool()
        self._clinical = ClinicalTool()

        self._tools = {
            self._wearable.name: self._wearable,
            self._imaging.name: self._imaging,
            self._motion.name: self._motion,
            self._clinical.name: self._clinical,
        }

    # ── EventBus handler ──────────────────────────────────────────────────

    def on_new_session(self, event: NewSessionEvent) -> None:
        logger.info("TwinAgent processing session %s for athlete %s",
                    event.session_id, event.athlete_id)
        self.process_session(event.athlete_id, event.session_data)

    # ── Public API ────────────────────────────────────────────────────────

    def process_session(self, athlete_id: str, session_dict: dict) -> DigitalTwin:
        """Ingest a session, update the digital twin, and persist it."""
        # 1. Persist raw session
        self.memory.store_session(session_dict)

        # 2. Load or create twin
        twin = self.memory.get_twin(athlete_id) or DigitalTwin(athlete_id=athlete_id)
        athlete_dict = self.memory.get_athlete_dict(athlete_id)

        # 3. Ask LLM to orchestrate ingestion and summarise the new twin state
        user_msg = SystemPrompts.twin_agent_user_prompt(session_dict, athlete_dict)
        response = self._run_loop(SystemPrompts.TWIN_AGENT, user_msg)
        logger.debug("TwinAgent LLM response: %s", response[:200])

        # 4. Build a new snapshot from session data (rule-based update to ensure
        #    the twin state is always deterministically consistent with raw data)
        new_snapshot = self._build_snapshot(twin, session_dict)
        twin.apply_snapshot(new_snapshot)
        self.memory.save_twin(twin)

        return twin

    # ── BaseAgent interface ───────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return tool.run(tool_input)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_snapshot(self, twin: DigitalTwin, session: dict) -> TwinSnapshot:
        """Deterministically compute the next twin snapshot from raw session data."""
        current = twin.current
        version = twin.version + 1

        wearable = session.get("wearable", {})
        motion = session.get("motion_capture", {})
        clinical = session.get("clinical_note", {})
        imaging = session.get("imaging", {})

        # Compute rolling training loads (simplified EWM)
        session_load = wearable.get("training_load", 0.0)
        prev_acute = current.acute_training_load if current else 0.0
        prev_chronic = current.chronic_training_load if current else 0.0
        # Exponential moving averages: 7-day (α≈2/8) and 28-day (α≈2/29)
        acute = prev_acute * (1 - 2 / 8) + session_load * (2 / 8)
        chronic = prev_chronic * (1 - 2 / 29) + session_load * (2 / 29)

        # Carry over injury / rehab state
        active_injury = current.active_injury if current else None
        in_rehab = current.in_rehabilitation if current else False
        rehab_day = (current.rehabilitation_day + 1) if in_rehab else 0

        # Update from clinical note if present
        if clinical:
            plan_text = clinical.get("plan", "").lower()
            if "return to play" in plan_text or "cleared" in plan_text:
                active_injury = None
                in_rehab = False
                rehab_day = 0

        # Biomechanical alerts from motion capture
        bio_alerts: list[str] = motion.get("anomalies", []) if motion else []
        si = motion.get("symmetry_index") if motion else None

        return TwinSnapshot(
            version=version,
            created_at=datetime.utcnow(),
            athlete_id=twin.athlete_id,
            current_heart_rate_resting=wearable.get("avg_heart_rate"),
            current_hrv=wearable.get("hrv"),
            current_body_weight_kg=None,  # requires explicit weigh-in
            acute_training_load=round(acute, 2),
            chronic_training_load=round(chronic, 2),
            active_injury=active_injury,
            in_rehabilitation=in_rehab,
            rehabilitation_day=rehab_day,
            symmetry_index=si,
            biomechanical_alerts=bio_alerts,
            last_session_id=session.get("session_id"),
        )
