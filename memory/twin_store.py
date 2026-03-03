"""
TwinStore – versioned, file-backed storage for Digital Twins.

Storage layout:
    <MEMORY_DIR>/twins/<athlete_id>/
        athlete.json          – static athlete profile
        twin_v{N}.json        – each snapshot
        current.json          – symlink / copy of latest snapshot
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import config
from models.athlete import Athlete
from models.twin import DigitalTwin, TwinSnapshot

logger = logging.getLogger(__name__)


class TwinStore:
    """Versioned file-backed store for DigitalTwin objects."""

    def __init__(self, base_dir: str = config.TWIN_STORE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, DigitalTwin] = {}

    # ── Athlete profile ───────────────────────────────────────────────────

    def save_athlete(self, athlete: Athlete) -> None:
        path = self._athlete_dir(athlete.athlete_id) / "athlete.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(athlete.to_dict(), f, indent=2, default=str)

    def load_athlete_dict(self, athlete_id: str) -> Optional[dict]:
        path = self._athlete_dir(athlete_id) / "athlete.json"
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    # ── Twin snapshots ────────────────────────────────────────────────────

    def save_twin(self, twin: DigitalTwin) -> None:
        """Persist the latest snapshot (only the new one)."""
        if not twin.snapshots:
            return
        snapshot = twin.snapshots[-1]
        version = snapshot.version
        athlete_dir = self._athlete_dir(twin.athlete_id)
        athlete_dir.mkdir(parents=True, exist_ok=True)

        snap_path = athlete_dir / f"twin_v{version}.json"
        with snap_path.open("w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

        current_path = athlete_dir / "current.json"
        with current_path.open("w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

        self._cache[twin.athlete_id] = twin
        logger.info("Saved twin v%d for athlete %s", version, twin.athlete_id)

    def load_twin(self, athlete_id: str) -> Optional[DigitalTwin]:
        """Load all snapshots and reconstruct a DigitalTwin."""
        if athlete_id in self._cache:
            return self._cache[athlete_id]

        athlete_dir = self._athlete_dir(athlete_id)
        if not athlete_dir.exists():
            return None

        snap_files = sorted(
            athlete_dir.glob("twin_v*.json"),
            key=lambda p: int(p.stem.split("v")[1]),
        )
        if not snap_files:
            return None

        twin = DigitalTwin(athlete_id=athlete_id)
        for sf in snap_files:
            with sf.open() as f:
                d = json.load(f)
            from datetime import datetime
            snapshot = TwinSnapshot(
                version=d["version"],
                created_at=datetime.fromisoformat(d["created_at"]),
                athlete_id=d["athlete_id"],
                current_heart_rate_resting=d.get("current_heart_rate_resting"),
                current_hrv=d.get("current_hrv"),
                current_body_weight_kg=d.get("current_body_weight_kg"),
                acute_training_load=d.get("acute_training_load", 0.0),
                chronic_training_load=d.get("chronic_training_load", 0.0),
                active_injury=d.get("active_injury"),
                in_rehabilitation=d.get("in_rehabilitation", False),
                rehabilitation_day=d.get("rehabilitation_day", 0),
                symmetry_index=d.get("symmetry_index"),
                biomechanical_alerts=d.get("biomechanical_alerts", []),
                last_session_id=d.get("last_session_id"),
                derived_features=d.get("derived_features", {}),
            )
            twin.apply_snapshot(snapshot)

        self._cache[athlete_id] = twin
        return twin

    def get_current_snapshot_dict(self, athlete_id: str) -> Optional[dict]:
        current_path = self._athlete_dir(athlete_id) / "current.json"
        if not current_path.exists():
            return None
        with current_path.open() as f:
            return json.load(f)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _athlete_dir(self, athlete_id: str) -> Path:
        return self.base_dir / athlete_id
