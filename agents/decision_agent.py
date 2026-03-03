"""
DecisionAgent – orchestrator and natural-language interface.

Responsibilities:
  - Parse natural-language queries from coaches, physicians, and physios.
  - Determine which specialist agents to invoke and in what order.
  - Synthesise outputs into a role-appropriate, human-readable response.
  - Emit AgentResponseEvent back to the caller via the EventBus.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from events.event_types import AgentQueryEvent, AgentResponseEvent
from events.event_bus import EventBus
from llm.prompts import SystemPrompts

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """Orchestrator agent – coordinates TwinAgent, RiskAgent, RehabAgent."""

    agent_name = "DecisionAgent"

    def __init__(self, llm, memory, event_bus: EventBus | None = None) -> None:
        super().__init__(llm, memory)
        self.event_bus = event_bus
        # Populated by the system after all agents are initialised
        self._twin_agent = None
        self._risk_agent = None
        self._rehab_agent = None

    def register_agents(self, twin_agent, risk_agent, rehab_agent) -> None:
        self._twin_agent = twin_agent
        self._risk_agent = risk_agent
        self._rehab_agent = rehab_agent

    # ── EventBus handler ──────────────────────────────────────────────────

    def on_agent_query(self, event: AgentQueryEvent) -> None:
        response_text = self.answer(
            athlete_id=event.athlete_id,
            query=event.query,
            user_role=event.user_role,
            conversation_id=event.conversation_id,
        )
        if self.event_bus:
            resp_event = AgentResponseEvent(
                athlete_id=event.athlete_id,
                response=response_text,
                source_agents=["TwinAgent", "RiskAgent", "RehabilitationAgent"],
                conversation_id=event.conversation_id,
            )
            self.event_bus.publish(resp_event)

    # ── Public API ────────────────────────────────────────────────────────

    def answer(
        self,
        athlete_id: str,
        query: str,
        user_role: str = "coach",
        conversation_id: str = "",
    ) -> str:
        """
        Process a natural-language query and return a synthesised response.

        The DecisionAgent gathers context from all specialist agents and asks
        the LLM to produce a coherent, role-appropriate answer.
        """
        logger.info("DecisionAgent answering query for athlete %s: %s", athlete_id, query[:80])

        # Gather structured context from specialist agents
        context: dict[str, Any] = {}

        # Twin state
        twin_state = self.memory.get_twin_state_dict(athlete_id)
        if twin_state:
            context["twin_state"] = twin_state

        # Athlete profile
        athlete = self.memory.get_athlete_dict(athlete_id)
        if athlete:
            context["athlete"] = athlete

        # Risk assessment
        if self._risk_agent:
            latest_risk = self._risk_agent.get_latest_assessment(athlete_id)
            if latest_risk:
                context["latest_risk_assessment"] = latest_risk

        # Rehab plan
        if self._rehab_agent and athlete_id in self._rehab_agent._plans:
            context["rehab_plan"] = self._rehab_agent._plans[athlete_id]

        # Recent sessions summary
        recent = self.memory.get_recent_sessions(athlete_id, days=7)
        context["recent_sessions_count"] = len(recent)

        user_msg = SystemPrompts.decision_agent_user_prompt(query, context, user_role)
        response = self._run_loop(SystemPrompts.DECISION_AGENT, user_msg)
        return response

    # ── BaseAgent interface ───────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        # The DecisionAgent uses LLM synthesis only (no domain tools of its own)
        return []

    def handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        return {"error": f"DecisionAgent has no tool: {tool_name}"}
