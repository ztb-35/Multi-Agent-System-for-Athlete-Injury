"""
Persistent Raw Session Memory – stores every SessionData event as JSON.
Provides auditability and retrospective analysis (paper §2.2).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

from models.athlete_state import SessionData


class SessionStore:
    """
    File-backed store for raw session events.
    Each athlete has one NDJSON file: memory/sessions/<athlete_id>.ndjson
    """

    def __init__(self, store_dir: str = "memory/sessions"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, athlete_id: str) -> Path:
        return self.store_dir / f"{athlete_id}.ndjson"

    def save(self, session: SessionData) -> None:
        """Append a session event to disk."""
        with open(self._path(session.athlete_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(session.to_dict()) + "\n")

    def load_all(self, athlete_id: str) -> list[SessionData]:
        """Load all sessions for an athlete, ordered by timestamp."""
        path = self._path(athlete_id)
        if not path.exists():
            return []
        sessions = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sessions.append(SessionData.from_dict(json.loads(line)))
        return sorted(sessions, key=lambda s: s.timestamp)

    def load_latest(self, athlete_id: str) -> Optional[SessionData]:
        """Return the most recent session for an athlete."""
        sessions = self.load_all(athlete_id)
        return sessions[-1] if sessions else None

    def get_session(self, athlete_id: str, session_id: str) -> Optional[SessionData]:
        """Fetch a specific session by ID."""
        for s in self.load_all(athlete_id):
            if s.session_id == session_id:
                return s
        return None

    def list_athletes(self) -> list[str]:
        return [p.stem for p in self.store_dir.glob("*.ndjson")]
