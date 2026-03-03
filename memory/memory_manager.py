"""
MemoryManager – unified façade over SessionMemory and TwinStore.

Agents interact with this single object for all persistence needs.
Also handles memory lifecycle: compaction, retention policies, and
context window-friendly summarization for LLM prompts.
"""
from __future__ import annotations
import logging
from typing import Optional

from .session_memory import SessionMemory
from .twin_store import TwinStore
from models.athlete import Athlete
from models.twin import DigitalTwin

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        session_memory: Optional[SessionMemory] = None,
        twin_store: Optional[TwinStore] = None,
    ) -> None:
        self.sessions = session_memory or SessionMemory()
        self.twins = twin_store or TwinStore()

    # ── Athlete ───────────────────────────────────────────────────────────

    def register_athlete(self, athlete: Athlete) -> None:
        self.twins.save_athlete(athlete)
        logger.info("Registered athlete %s (%s)", athlete.name, athlete.athlete_id)

    def get_athlete_dict(self, athlete_id: str) -> Optional[dict]:
        return self.twins.load_athlete_dict(athlete_id)

    # ── Sessions ──────────────────────────────────────────────────────────

    def store_session(self, session_dict: dict) -> None:
        self.sessions.store(session_dict)

    def get_recent_sessions(self, athlete_id: str, days: int = 7) -> list[dict]:
        return self.sessions.load_recent(athlete_id, days)

    def get_all_sessions(self, athlete_id: str) -> list[dict]:
        return self.sessions.load_all(athlete_id)

    # ── Digital Twin ──────────────────────────────────────────────────────

    def get_twin(self, athlete_id: str) -> Optional[DigitalTwin]:
        return self.twins.load_twin(athlete_id)

    def save_twin(self, twin: DigitalTwin) -> None:
        self.twins.save_twin(twin)

    def get_twin_state_dict(self, athlete_id: str) -> Optional[dict]:
        return self.twins.get_current_snapshot_dict(athlete_id)

    # ── Context building for LLM prompts ──────────────────────────────────

    def build_context_for_agent(self, athlete_id: str, last_n_sessions: int = 5) -> dict:
        """Return a compact dict suitable for injection into LLM prompts."""
        return {
            "athlete": self.get_athlete_dict(athlete_id),
            "current_twin_state": self.get_twin_state_dict(athlete_id),
            "recent_sessions": self.get_recent_sessions(athlete_id, days=14)[-last_n_sessions:],
        }
