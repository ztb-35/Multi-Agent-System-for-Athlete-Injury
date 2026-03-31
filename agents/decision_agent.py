"""
Decision Agent – Human-Facing NLP Coordinator (paper §2.5).

Responsibilities:
  - Parse natural language queries from coaches, medical staff, trainers
  - Route to appropriate specialist agents via Claude tool_use
  - Synthesize structured outputs into concise, role-aware responses
  - Does NOT perform primary inference – only coordination & explanation
  - Supports two query modes:
      * "How is A01 doing today?"        → risk + brief summary
      * "Show me A01's rehab progression" → rehab plan + trend
"""

from __future__ import annotations
import json
import os
from typing import Optional

import anthropic

from models.athlete_state import AthleteState, RiskAssessment, RehabPlan
from agents.twin_agent import TwinAgent
from agents.risk_agent import RiskAgent
from agents.rehab_agent import RehabAgent


# ---------------------------------------------------------------------------
# Tool definitions for Claude tool_use
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_athlete_state",
        "description": "Retrieve the latest digital twin state for an athlete, including their biomechanical snapshot, trends, deviations from baseline, and session history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "athlete_id": {
                    "type": "string",
                    "description": "The athlete's unique ID (e.g. 'A01')",
                }
            },
            "required": ["athlete_id"],
        },
    },
    {
        "name": "assess_injury_risk",
        "description": "Run injury risk assessment for an athlete based on their latest digital twin state. Returns risk level, confidence, top drivers, and reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "athlete_id": {
                    "type": "string",
                    "description": "The athlete's unique ID",
                }
            },
            "required": ["athlete_id"],
        },
    },
    {
        "name": "get_rehab_plan",
        "description": "Get the personalized rehabilitation plan for an injured athlete, including current stage, weekly exercises, restrictions, and progression criteria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "athlete_id": {
                    "type": "string",
                    "description": "The athlete's unique ID",
                }
            },
            "required": ["athlete_id"],
        },
    },
    {
        "name": "what_if_analysis",
        "description": "Perform counterfactual reasoning: what would happen if a specific intervention were applied to the athlete's rehabilitation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "athlete_id": {
                    "type": "string",
                    "description": "The athlete's unique ID",
                },
                "intervention": {
                    "type": "string",
                    "description": "The hypothetical intervention to analyze, e.g. 'reduce workload by 30% for 2 weeks'",
                },
            },
            "required": ["athlete_id", "intervention"],
        },
    },
]


class DecisionAgent:
    """
    NLP coordinator between human users and specialist agents.

    Usage:
        agent = DecisionAgent(twin_agent, risk_agent, rehab_agent)

        # Coach query
        response = agent.query("How is Abigail doing today?", athlete_id="A01")
        print(response)

        # Medical staff query
        response = agent.query("Show me Abigail's rehab progression and whether she can start lateral hops", athlete_id="A01")
        print(response)
    """

    SYSTEM_PROMPT = """You are a sports medicine coordination AI.
You help coaches, medical staff, and trainers understand athlete injury risk and rehabilitation status.

You have access to tools that retrieve athlete data and run specialist assessments.
Always use the appropriate tools before answering – never guess or hallucinate data.

Communication guidelines:
- For COACHES: be concise and action-oriented ("Can she train today? Yes/No and why")
- For MEDICAL STAFF: be detailed and evidence-backed with biomechanical specifics
- Always cite session IDs and metrics when making claims
- Never override medical safety constraints
- If data is insufficient, say so clearly"""

    def __init__(
        self,
        twin_agent: TwinAgent,
        risk_agent: RiskAgent,
        rehab_agent: RehabAgent,
        model: str = "claude-sonnet-4-6",
    ):
        self.twin_agent  = twin_agent
        self.risk_agent  = risk_agent
        self.rehab_agent = rehab_agent
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        athlete_id: Optional[str] = None,
        role: str = "coach",  # "coach" | "medical" | "trainer"
    ) -> str:
        """
        Handle a natural language query from a human user.

        Args:
            question:   Natural language question
            athlete_id: Optional hint; extracted from query if not provided
            role:       User role for response style

        Returns:
            Human-readable response string
        """
        user_content = question
        if athlete_id:
            user_content = f"[Athlete ID hint: {athlete_id}]\n{question}"
        if role != "coach":
            user_content += f"\n[User role: {role}]"

        print(f"\n[DecisionAgent] Query from {role}: {question!r}")

        messages = [{"role": "user", "content": user_content}]
        return self._agentic_loop(messages)

    # ------------------------------------------------------------------
    # Agentic loop with tool_use
    # ------------------------------------------------------------------

    def _agentic_loop(self, messages: list[dict]) -> str:
        """Run the Claude tool-use loop until a final text response."""
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # If Claude returns text, we're done
            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            # If Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Append assistant message
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Execute each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        print(f"[DecisionAgent] Tool: {block.name}({block.input}) → {str(result)[:120]}...")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                # Append tool results and continue
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue

            # Unexpected stop reason
            return self._extract_text(response)

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, inputs: dict) -> dict:
        """Dispatch a Claude tool call to the appropriate agent method."""
        try:
            if name == "get_athlete_state":
                return self._tool_get_state(inputs["athlete_id"])

            elif name == "assess_injury_risk":
                return self._tool_risk(inputs["athlete_id"])

            elif name == "get_rehab_plan":
                return self._tool_rehab(inputs["athlete_id"])

            elif name == "what_if_analysis":
                return self._tool_counterfactual(
                    inputs["athlete_id"], inputs["intervention"]
                )

            else:
                return {"error": f"Unknown tool: {name}"}

        except Exception as e:
            return {"error": str(e)}

    def _tool_get_state(self, athlete_id: str) -> dict:
        state = self.twin_agent.get_state(athlete_id)
        if state is None:
            return {"error": f"No digital twin found for athlete {athlete_id}"}

        # Return a condensed state (avoid huge JSON in context)
        return {
            "athlete_id": state.athlete_id,
            "name": state.name,
            "version": state.version,
            "active_injury": state.active_injury,
            "injury_history": state.injury_history,
            "sessions_count": len(state.session_ids),
            "latest_session": state.session_ids[-1] if state.session_ids else None,
            "pain_scores": state.pain_scores[-3:] if state.pain_scores else [],
            "latest_snapshot": state.latest_snapshot,
            "deviations": {
                k: v for k, v in state.deviations.items()
                if abs(v.get("pct_change", 0)) > 5
            } if state.deviations else {},
            "asymmetry_trend": state.trends.get("knee_asymmetry_index", [])[-5:],
        }

    def _tool_risk(self, athlete_id: str) -> dict:
        state = self.twin_agent.get_state(athlete_id)
        if state is None:
            return {"error": f"No digital twin found for athlete {athlete_id}"}
        session_id = state.session_ids[-1] if state.session_ids else "latest"
        assessment = self.risk_agent.assess(state, session_id)
        return assessment.to_dict()

    def _tool_rehab(self, athlete_id: str) -> dict:
        state = self.twin_agent.get_state(athlete_id)
        if state is None:
            return {"error": f"No digital twin found for athlete {athlete_id}"}
        if not state.active_injury:
            return {"message": f"Athlete {athlete_id} has no active injury recorded."}
        session_id = state.session_ids[-1] if state.session_ids else "latest"
        plan = self.rehab_agent.plan(state, session_id)
        return plan.to_dict()

    def _tool_counterfactual(self, athlete_id: str, intervention: str) -> dict:
        state = self.twin_agent.get_state(athlete_id)
        if state is None:
            return {"error": f"No digital twin found for athlete {athlete_id}"}
        analysis = self.rehab_agent.counterfactual(state, intervention)
        return {"analysis": analysis}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts).strip() or "[No response generated]"
