"""
Event type definitions for the MASAI event-driven architecture.

Events flow through the EventBus and trigger agent pipelines:

  NewSessionEvent  ──►  TwinAgent (update twin)
                   ──►  RiskAgent (assess risk)

  InjuryReportedEvent  ──►  RehabilitationAgent (create plan)

  AgentQueryEvent  ──►  DecisionAgent (orchestrate response)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    NEW_SESSION = "new_session"
    INJURY_REPORTED = "injury_reported"
    REHAB_UPDATE = "rehab_update"
    AGENT_QUERY = "agent_query"
    AGENT_RESPONSE = "agent_response"
    RISK_ALERT = "risk_alert"


@dataclass
class BaseEvent:
    event_type: EventType
    athlete_id: str
    event_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NewSessionEvent(BaseEvent):
    """Fired when a new training/match/rehab session is ingested."""
    session_id: str = ""
    session_data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.NEW_SESSION


@dataclass
class InjuryReportedEvent(BaseEvent):
    """Fired when a new injury is recorded for an athlete."""
    injury_id: str = ""
    body_part: str = ""
    diagnosis: str = ""
    severity: str = ""

    def __post_init__(self):
        self.event_type = EventType.INJURY_REPORTED


@dataclass
class RehabUpdateEvent(BaseEvent):
    """Fired when rehabilitation progress data arrives."""
    plan_id: str = ""
    current_day: int = 0
    session_data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = EventType.REHAB_UPDATE


@dataclass
class AgentQueryEvent(BaseEvent):
    """Fired when a user submits a natural-language query."""
    query: str = ""
    user_role: str = "coach"           # "coach" | "physician" | "physio"
    conversation_id: str = ""

    def __post_init__(self):
        self.event_type = EventType.AGENT_QUERY


@dataclass
class AgentResponseEvent(BaseEvent):
    """Emitted by DecisionAgent after processing a query."""
    response: str = ""
    source_agents: list[str] = field(default_factory=list)
    conversation_id: str = ""

    def __post_init__(self):
        self.event_type = EventType.AGENT_RESPONSE


@dataclass
class RiskAlertEvent(BaseEvent):
    """Emitted by RiskAgent when risk exceeds a critical threshold."""
    risk_score: float = 0.0
    risk_level: str = ""
    risk_factors: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.event_type = EventType.RISK_ALERT
