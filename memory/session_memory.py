"""
Raw Session Memory – append-only persistent store for every ingested session.

Corresponds to the "Raw Session Memory" component in the blueprint.
Stores raw multimodal data for audit, retrospective analysis, and replay.

Storage layout:
    <MEMORY_DIR>/sessions/<athlete_id>/<session_id>.json
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


class SessionMemory:
    """Append-only, file-backed store for raw training session data."""

    def __init__(self, base_dir: str = config.SESSION_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────

    def store(self, session_dict: dict) -> None:
        """Persist a session dict (from TrainingSession.to_dict())."""
        athlete_id = session_dict["athlete_id"]
        session_id = session_dict["session_id"]
        path = self._path(athlete_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(session_dict, f, indent=2, default=str)
        logger.info("Stored session %s for athlete %s", session_id, athlete_id)

    # ── Read ──────────────────────────────────────────────────────────────

    def load(self, athlete_id: str, session_id: str) -> Optional[dict]:
        path = self._path(athlete_id, session_id)
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    def load_all(self, athlete_id: str) -> list[dict]:
        """Return all sessions for an athlete, sorted by timestamp."""
        athlete_dir = self.base_dir / athlete_id
        if not athlete_dir.exists():
            return []
        sessions = []
        for p in athlete_dir.glob("*.json"):
            with p.open() as f:
                sessions.append(json.load(f))
        sessions.sort(key=lambda s: s.get("timestamp", ""))
        return sessions

    def load_recent(self, athlete_id: str, days: int) -> list[dict]:
        """Return sessions within the last `days` days."""
        from datetime import timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = []
        for s in self.load_all(athlete_id):
            ts_str = s.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    result.append(s)
            except ValueError:
                pass
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _path(self, athlete_id: str, session_id: str) -> Path:
        return self.base_dir / athlete_id / f"{session_id}.json"
