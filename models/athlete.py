"""
Athlete core data models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class AthleteBaseline:
    """Per-athlete physiological and performance baselines.

    Used by RiskAgent to evaluate deviations relative to the individual
    rather than population-level thresholds.
    """
    resting_heart_rate: float          # bpm
    hrv_baseline: float                # ms (RMSSD)
    vo2_max: float                     # ml/kg/min
    daily_training_load_avg: float     # AU (arbitrary units, e.g. sRPE)
    weekly_training_load_avg: float    # AU
    sleep_hours_avg: float             # hours/night
    body_weight_kg: float
    # Injury-specific baselines (populated after first full assessment)
    acl_laxity_mm: Optional[float] = None
    shoulder_rom_deg: Optional[float] = None


@dataclass
class InjuryRecord:
    injury_id: str
    date_of_injury: date
    body_part: str                     # e.g. "left knee – ACL"
    diagnosis: str
    severity: str                      # "mild" | "moderate" | "severe"
    expected_recovery_days: int
    actual_recovery_days: Optional[int] = None
    notes: str = ""


@dataclass
class Athlete:
    athlete_id: str
    name: str
    sport: str
    position: str
    date_of_birth: date
    baseline: Optional[AthleteBaseline] = None
    injury_history: list[InjuryRecord] = field(default_factory=list)

    def age(self) -> int:
        today = date.today()
        b = self.date_of_birth
        return today.year - b.year - ((today.month, today.day) < (b.month, b.day))

    def to_dict(self) -> dict:
        return {
            "athlete_id": self.athlete_id,
            "name": self.name,
            "sport": self.sport,
            "position": self.position,
            "age": self.age(),
            "baseline": self.baseline.__dict__ if self.baseline else None,
            "injury_history": [
                {**ir.__dict__, "date_of_injury": ir.date_of_injury.isoformat()}
                for ir in self.injury_history
            ],
        }
