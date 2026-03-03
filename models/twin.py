"""
Digital Twin models – versioned athlete state representation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TwinSnapshot:
    """Immutable, versioned snapshot of an athlete's state at a given moment."""
    version: int
    created_at: datetime
    athlete_id: str

    # Current physiological state
    current_heart_rate_resting: Optional[float] = None
    current_hrv: Optional[float] = None
    current_body_weight_kg: Optional[float] = None

    # Cumulative load metrics
    acute_training_load: float = 0.0   # rolling 7-day AU
    chronic_training_load: float = 0.0 # rolling 28-day AU

    @property
    def acwr(self) -> Optional[float]:
        """Acute:Chronic Workload Ratio – a well-known injury risk proxy."""
        if self.chronic_training_load > 0:
            return self.acute_training_load / self.chronic_training_load
        return None

    # Injury & rehabilitation state
    active_injury: Optional[str] = None         # description or None
    in_rehabilitation: bool = False
    rehabilitation_day: int = 0

    # Biomechanical flags
    symmetry_index: Optional[float] = None
    biomechanical_alerts: list[str] = field(default_factory=list)

    # Metadata
    last_session_id: Optional[str] = None
    derived_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "athlete_id": self.athlete_id,
            "current_heart_rate_resting": self.current_heart_rate_resting,
            "current_hrv": self.current_hrv,
            "current_body_weight_kg": self.current_body_weight_kg,
            "acute_training_load": self.acute_training_load,
            "chronic_training_load": self.chronic_training_load,
            "acwr": self.acwr,
            "active_injury": self.active_injury,
            "in_rehabilitation": self.in_rehabilitation,
            "rehabilitation_day": self.rehabilitation_day,
            "symmetry_index": self.symmetry_index,
            "biomechanical_alerts": self.biomechanical_alerts,
            "last_session_id": self.last_session_id,
            "derived_features": self.derived_features,
        }


@dataclass
class DigitalTwin:
    """Live digital twin for an athlete, maintaining a full version history."""
    athlete_id: str
    snapshots: list[TwinSnapshot] = field(default_factory=list)

    @property
    def current(self) -> Optional[TwinSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def version(self) -> int:
        return len(self.snapshots)

    def apply_snapshot(self, snapshot: TwinSnapshot) -> None:
        self.snapshots.append(snapshot)

    def get_snapshot(self, version: int) -> Optional[TwinSnapshot]:
        if 1 <= version <= len(self.snapshots):
            return self.snapshots[version - 1]
        return None

    def history_as_dicts(self, last_n: int = 10) -> list[dict]:
        return [s.to_dict() for s in self.snapshots[-last_n:]]
