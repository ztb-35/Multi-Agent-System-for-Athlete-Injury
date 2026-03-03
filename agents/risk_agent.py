"""
RiskAgent – individual-baseline injury risk and fatigue assessment.

Responsibilities:
  - Listen to NewSessionEvent (auto-assess after each session).
  - Compute ensemble risk scores via RiskToolkit.
  - Emit RiskAlertEvent when risk exceeds critical threshold.
  - Provide risk context to the DecisionAgent on demand.
"""
from __future__ import annotations
import logging
from typing import Any

from agents.base_agent import BaseAgent
from events.event_types import NewSessionEvent, RiskAlertEvent
from events.event_bus import EventBus
from llm.prompts import SystemPrompts
from tools.risk_tools import RiskToolkit

import config

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    agent_name = "RiskAgent"

    def __init__(self, llm, memory, event_bus: EventBus | None = None) -> None:
        super().__init__(llm, memory)
        self.event_bus = event_bus
        self._toolkit = RiskToolkit()
        self._last_assessments: dict[str, dict] = {}  # athlete_id → latest result

    # ── EventBus handler ──────────────────────────────────────────────────

    def on_new_session(self, event: NewSessionEvent) -> None:
        logger.info("RiskAgent assessing risk for athlete %s", event.athlete_id)
        result = self.assess(event.athlete_id)
        # Emit alert if critical
        if result.get("overall_risk_level") in ("high", "critical") and self.event_bus:
            alert = RiskAlertEvent(
                athlete_id=event.athlete_id,
                risk_score=result["overall_risk_score"],
                risk_level=result["overall_risk_level"],
                risk_factors=result.get("risk_factors", []),
            )
            self.event_bus.publish(alert)

    # ── Public API ────────────────────────────────────────────────────────

    def assess(self, athlete_id: str) -> dict:
        """Run a full risk and fatigue assessment for athlete_id."""
        context = self.memory.build_context_for_agent(athlete_id)
        user_msg = SystemPrompts.risk_agent_user_prompt(context)
        response = self._run_loop(SystemPrompts.RISK_AGENT, user_msg)
        logger.debug("RiskAgent LLM response: %s", response[:200])

        # The actual structured results come from tool calls captured internally;
        # return the last tool result cached by the loop for programmatic use.
        result = self._last_assessments.get(athlete_id, {})
        return result

    def get_latest_assessment(self, athlete_id: str) -> dict | None:
        return self._last_assessments.get(athlete_id)

    # ── BaseAgent interface ───────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        return self._toolkit.schemas()

    def handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool_map = {t.name: t for t in self._toolkit.all_tools()}
        tool = tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        # Inject memory context if not already in tool_input
        athlete_id = tool_input.get("athlete_id", "")
        if "twin_state" not in tool_input:
            twin_state = self.memory.get_twin_state_dict(athlete_id)
            if twin_state:
                tool_input = {**tool_input, "twin_state": twin_state}
        if "recent_sessions" not in tool_input:
            tool_input = {**tool_input, "recent_sessions": self.memory.get_recent_sessions(athlete_id, 28)}
        if "athlete_baseline" not in tool_input:
            athlete = self.memory.get_athlete_dict(athlete_id)
            if athlete and athlete.get("baseline"):
                tool_input = {**tool_input, "athlete_baseline": athlete["baseline"]}

        result = tool.run(tool_input)

        # Cache for programmatic access
        if tool_name == "compute_risk_score":
            self._last_assessments[athlete_id] = result

        return result
