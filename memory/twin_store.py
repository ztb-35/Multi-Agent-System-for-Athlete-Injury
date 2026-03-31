"""
Versioned Athlete Digital Twin Store.
Each update increments the version and snapshots the full AthleteState to disk.
Provides longitudinal state tracking (paper §2.2).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from models.athlete_state import AthleteState


class TwinStore:
    """
    File-backed versioned store for AthleteState objects.
    Layout: memory/twins/<athlete_id>/v<version>.json
            memory/twins/<athlete_id>/latest.json  (symlinked to latest version)
    """

    def __init__(self, store_dir: str = "memory/twins"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _athlete_dir(self, athlete_id: str) -> Path:
        d = self.store_dir / athlete_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, state: AthleteState) -> None:
        """Persist a new version of the athlete's digital twin."""
        d = self._athlete_dir(state.athlete_id)
        version_file = d / f"v{state.version:04d}.json"
        latest_file  = d / "latest.json"

        payload = json.dumps(state.to_dict(), indent=2)
        version_file.write_text(payload, encoding="utf-8")
        latest_file.write_text(payload, encoding="utf-8")

    def load_latest(self, athlete_id: str) -> Optional[AthleteState]:
        """Load the most recent twin state."""
        latest_file = self._athlete_dir(athlete_id) / "latest.json"
        if not latest_file.exists():
            return None
        return AthleteState.from_dict(json.loads(latest_file.read_text(encoding="utf-8")))

    def load_version(self, athlete_id: str, version: int) -> Optional[AthleteState]:
        """Load a specific historical version."""
        vf = self._athlete_dir(athlete_id) / f"v{version:04d}.json"
        if not vf.exists():
            return None
        return AthleteState.from_dict(json.loads(vf.read_text(encoding="utf-8")))

    def list_versions(self, athlete_id: str) -> list[int]:
        """Return all available version numbers for an athlete."""
        d = self._athlete_dir(athlete_id)
        return sorted(
            int(p.stem[1:])
            for p in d.glob("v*.json")
        )

    def list_athletes(self) -> list[str]:
        return [p.name for p in self.store_dir.iterdir() if p.is_dir()]
