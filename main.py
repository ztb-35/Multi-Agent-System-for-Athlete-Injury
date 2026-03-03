"""
MASAI – Multi-Agent System for Athlete Injury (entry point)

Demonstrates the full system lifecycle:
  1. Register an athlete.
  2. Ingest a training session  →  TwinAgent updates twin, RiskAgent assesses risk.
  3. Report an injury           →  RehabilitationAgent generates a rehab plan.
  4. Submit a natural-language query via DecisionAgent.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py
"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime

import config
from agents import DecisionAgent, RehabilitationAgent, RiskAgent, TwinAgent
from events import (
    AgentQueryEvent,
    EventBus,
    EventType,
    InjuryReportedEvent,
    NewSessionEvent,
)
from llm.client import LLMClient
from memory.memory_manager import MemoryManager
from models.athlete import Athlete, AthleteBaseline
from models.session import TrainingSession, WearableData, MotionCaptureData

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System factory
# ─────────────────────────────────────────────────────────────────────────────

def build_system() -> tuple[EventBus, DecisionAgent]:
    """Instantiate and wire all MASAI components."""
    llm = LLMClient()
    memory = MemoryManager()
    bus = EventBus()

    twin_agent = TwinAgent(llm, memory)
    risk_agent = RiskAgent(llm, memory, event_bus=bus)
    rehab_agent = RehabilitationAgent(llm, memory)
    decision_agent = DecisionAgent(llm, memory, event_bus=bus)

    decision_agent.register_agents(twin_agent, risk_agent, rehab_agent)

    # Subscribe agents to events
    bus.subscribe(EventType.NEW_SESSION, twin_agent.on_new_session)
    bus.subscribe(EventType.NEW_SESSION, risk_agent.on_new_session)
    bus.subscribe(EventType.INJURY_REPORTED, rehab_agent.on_injury_reported)
    bus.subscribe(EventType.AGENT_QUERY, decision_agent.on_agent_query)

    logger.info("MASAI system initialised.")
    return bus, decision_agent


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    bus, decision_agent = build_system()
    memory = decision_agent.memory

    # ── 1. Register athlete ──────────────────────────────────────────────
    athlete = Athlete(
        athlete_id="ATH_001",
        name="Alex Johnson",
        sport="Soccer",
        position="Central Midfielder",
        date_of_birth=datetime(1998, 4, 15).date(),
        baseline=AthleteBaseline(
            resting_heart_rate=52.0,
            hrv_baseline=78.0,
            vo2_max=58.3,
            daily_training_load_avg=420.0,
            weekly_training_load_avg=2940.0,
            sleep_hours_avg=7.8,
            body_weight_kg=74.5,
        ),
    )
    memory.register_athlete(athlete)
    logger.info("Registered athlete: %s", athlete.name)

    # ── 2. Simulate a high-load training session ──────────────────────────
    session = TrainingSession(
        session_id=f"SES_{uuid.uuid4().hex[:8]}",
        athlete_id=athlete.athlete_id,
        timestamp=datetime.utcnow(),
        session_type="training",
        duration_minutes=95,
        wearable=WearableData(
            avg_heart_rate=158,
            max_heart_rate=187,
            hrv=62.0,                    # suppressed (baseline 78)
            training_load=680.0,         # well above daily avg of 420
            distance_km=11.2,
            sprint_count=31,
            acceleration_count=78,
            sleep_hours_prev_night=6.2,  # below avg
            sleep_quality_score=0.55,
        ),
        motion_capture=MotionCaptureData(
            system="IMU",
            joint_angles={"knee_flexion_peak_L": 68.1, "knee_flexion_peak_R": 82.4},
            ground_reaction_forces={"vertical_peak_N": 1620},
            symmetry_index=0.80,          # below 0.85 threshold
            anomalies=["Reduced left knee flexion during landing"],
        ),
    )

    event = NewSessionEvent(
        athlete_id=athlete.athlete_id,
        session_id=session.session_id,
        session_data=session.to_dict(),
    )
    logger.info("Publishing NewSessionEvent for session %s", session.session_id)
    bus.publish(event)

    # ── 3. Report an injury ───────────────────────────────────────────────
    injury_event = InjuryReportedEvent(
        athlete_id=athlete.athlete_id,
        injury_id=f"INJ_{uuid.uuid4().hex[:8]}",
        body_part="Left Knee",
        diagnosis="Grade II MCL Sprain",
        severity="moderate",
    )
    logger.info("Publishing InjuryReportedEvent: %s", injury_event.diagnosis)
    bus.publish(injury_event)

    # ── 4. Natural-language query via DecisionAgent ───────────────────────
    query = (
        "What is Alex's current injury risk level and "
        "when can he realistically return to full training?"
    )
    response = decision_agent.answer(
        athlete_id=athlete.athlete_id,
        query=query,
        user_role="coach",
    )
    print("\n" + "=" * 70)
    print("DECISION AGENT RESPONSE")
    print("=" * 70)
    print(response)
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
