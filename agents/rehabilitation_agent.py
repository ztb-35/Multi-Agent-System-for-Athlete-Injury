"""
RehabilitationAgent – post-injury monitoring and return-to-sport planning.

Responsibilities:
  - Listen to InjuryReportedEvent → generate rehabilitation plan.
  - Listen to RehabUpdateEvent → assess progress, update milestones.
  - Provide RTS readiness assessment on demand.
  - Support causal / counterfactual reasoning about recovery scenarios.
"""
from __future__ import annotations
import logging
from typing import Any

from agents.base_agent import BaseAgent
from events.event_types import InjuryReportedEvent, RehabUpdateEvent
from llm.prompts import SystemPrompts
from tools.rehab_tools import RehabToolkit

logger = logging.getLogger(__name__)


class RehabilitationAgent(BaseAgent):
    agent_name = "RehabilitationAgent"

    def __init__(self, llm, memory) -> None:
        super().__init__(llm, memory)
        self._toolkit = RehabToolkit()
        # In-memory plan cache (production: persist via MemoryManager)
        self._plans: dict[str, dict] = {}  # athlete_id → latest plan

    # ── EventBus handlers ─────────────────────────────────────────────────

    def on_injury_reported(self, event: InjuryReportedEvent) -> None:
        logger.info("RehabAgent creating plan for athlete %s – %s",
                    event.athlete_id, event.body_part)
        plan = self.create_plan(
            athlete_id=event.athlete_id,
            injury_id=event.injury_id,
            injury_description=f"{event.body_part}: {event.diagnosis}",
            severity=event.severity,
        )
        self._plans[event.athlete_id] = plan

    def on_rehab_update(self, event: RehabUpdateEvent) -> None:
        logger.info("RehabAgent updating progress for athlete %s day %d",
                    event.athlete_id, event.current_day)
        self.assess_progress(event.athlete_id, event.current_day, event.session_data)

    # ── Public API ────────────────────────────────────────────────────────

    def create_plan(
        self,
        athlete_id: str,
        injury_id: str,
        injury_description: str,
        severity: str,
    ) -> dict:
        """Generate a phased rehabilitation plan."""
        athlete_dict = self.memory.get_athlete_dict(athlete_id)
        context = {
            "athlete": athlete_dict,
            "injury_id": injury_id,
            "injury_description": injury_description,
            "severity": severity,
        }
        task = (
            f"Create a rehabilitation plan for injury: '{injury_description}' "
            f"(severity: {severity}, athlete_id: {athlete_id}, injury_id: {injury_id})"
        )
        user_msg = SystemPrompts.rehab_agent_user_prompt(context, task)
        self._run_loop(SystemPrompts.REHABILITATION_AGENT, user_msg)
        return self._plans.get(athlete_id, {})

    def assess_progress(self, athlete_id: str, current_day: int, session_data: dict) -> dict:
        """Evaluate rehabilitation progress for an athlete."""
        plan = self._plans.get(athlete_id, {})
        context = self.memory.build_context_for_agent(athlete_id)
        context["rehab_plan"] = plan
        context["current_rehab_day"] = current_day
        context["session_data"] = session_data
        task = f"Assess rehabilitation progress on day {current_day}."
        user_msg = SystemPrompts.rehab_agent_user_prompt(context, task)
        response = self._run_loop(SystemPrompts.REHABILITATION_AGENT, user_msg)
        return {"response": response, "current_day": current_day}

    def check_rts_readiness(self, athlete_id: str) -> dict:
        """Evaluate return-to-sport readiness."""
        plan = self._plans.get(athlete_id, {})
        context = self.memory.build_context_for_agent(athlete_id)
        context["rehab_plan"] = plan
        task = "Evaluate whether this athlete is ready to return to sport."
        user_msg = SystemPrompts.rehab_agent_user_prompt(context, task)
        response = self._run_loop(SystemPrompts.REHABILITATION_AGENT, user_msg)
        return {"response": response}

    # ── BaseAgent interface ───────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        return self._toolkit.schemas()

    def handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool_map = {t.name: t for t in self._toolkit.all_tools()}
        tool = tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        # Inject twin state / rehab plan if missing
        athlete_id = tool_input.get("athlete_id", "")
        if "twin_state" not in tool_input:
            ts = self.memory.get_twin_state_dict(athlete_id)
            if ts:
                tool_input = {**tool_input, "twin_state": ts}
        if "rehab_plan" not in tool_input and athlete_id in self._plans:
            tool_input = {**tool_input, "rehab_plan": self._plans[athlete_id]}

        result = tool.run(tool_input)

        # Cache generated plan
        if tool_name == "generate_rehab_plan":
            self._plans[athlete_id] = result

        return result
