"""
Data models for the Multi-Agent Digital Twin system.
Based on: "A Multi-Agent Digital Twin Blueprint for Athlete Injury Risk Assessment
           and Rehabilitation Planning" (ICLR 2026 Workshop on MALGAI)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


# ---------------------------------------------------------------------------
# Biomechanical feature snapshot extracted from one session
# ---------------------------------------------------------------------------

@dataclass
class BiomechanicsSnapshot:
    """Kinematic summary stats computed from an OpenSim .mot file."""
    session_id: str

    # Knee angles (degrees)
    knee_angle_r_mean: float = 0.0
    knee_angle_r_std: float = 0.0
    knee_angle_r_min: float = 0.0
    knee_angle_r_max: float = 0.0

    knee_angle_l_mean: float = 0.0
    knee_angle_l_std: float = 0.0
    knee_angle_l_min: float = 0.0
    knee_angle_l_max: float = 0.0

    # Hip flexion angles (degrees)
    hip_flexion_r_mean: float = 0.0
    hip_flexion_l_mean: float = 0.0

    # Hip adduction (valgus proxy)
    hip_adduction_r_mean: float = 0.0
    hip_adduction_l_mean: float = 0.0

    # Ankle angles (degrees)
    ankle_angle_r_mean: float = 0.0
    ankle_angle_l_mean: float = 0.0

    # Pelvis
    pelvis_tilt_mean: float = 0.0
    pelvis_list_mean: float = 0.0

    # Lumbar
    lumbar_extension_mean: float = 0.0

    # Derived metrics
    knee_asymmetry_index: float = 0.0   # % asymmetry (R vs L)
    hip_asymmetry_index: float = 0.0

    # Workload proxy
    session_duration_s: float = 0.0
    n_frames: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BiomechanicsSnapshot":
        return cls(**d)


# ---------------------------------------------------------------------------
# Session event
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    """Raw session container – the NewSessionEvent payload."""
    athlete_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    mot_file: Optional[str] = None      # path to .mot kinematics
    trc_file: Optional[str] = None      # path to .trc marker data
    yaml_file: Optional[str] = None     # path to session metadata
    injury_notes: str = ""              # free-text clinical note
    pain_score: Optional[float] = None  # 0-10 VAS scale

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionData":
        return cls(**d)


# ---------------------------------------------------------------------------
# Versioned Athlete Digital Twin state
# ---------------------------------------------------------------------------

@dataclass
class AthleteState:
    """
    Versioned digital twin state – the single source of truth.
    Updated by the TwinAgent; consumed by Risk/Rehab Agents.
    """
    athlete_id: str
    version: int = 0
    last_updated: float = field(default_factory=time.time)

    # Static profile
    name: str = ""
    age: Optional[int] = None
    height_m: float = 0.0
    mass_kg: float = 0.0
    gender: str = ""
    sport: str = ""
    injury_history: list[str] = field(default_factory=list)
    active_injury: Optional[str] = None   # e.g. "ACL_recon"

    # Session history (session_id list)
    session_ids: list[str] = field(default_factory=list)

    # Latest snapshot
    latest_snapshot: Optional[dict] = None   # BiomechanicsSnapshot.to_dict()

    # Personalized baseline (rolling average across sessions)
    baseline: dict = field(default_factory=dict)

    # Deviations from baseline (latest_snapshot - baseline)
    deviations: dict = field(default_factory=dict)

    # Temporal trends (list of per-session values for key metrics)
    trends: dict = field(default_factory=dict)
    # e.g. trends = {
    #   "knee_asymmetry_index": [4.2, 5.1, 6.8, 8.3],
    #   "knee_angle_r_mean":    [-15.3, -14.1, -16.2, -18.0],
    # }

    # Clinical notes history
    pain_scores: list[float] = field(default_factory=list)
    injury_notes_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AthleteState":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Risk Assessment output
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """Structured output from the Risk Agent."""
    athlete_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)

    risk_level: str = "Unknown"         # Low / Moderate / High / Critical
    confidence: float = 0.0             # 0–1
    top_risk_drivers: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    reasoning: str = ""                 # LLM chain-of-thought summary

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        drivers = "\n  - ".join(self.top_risk_drivers) if self.top_risk_drivers else "None identified"
        refs    = ", ".join(self.evidence_refs) if self.evidence_refs else "N/A"
        return (
            f"Risk Level : {self.risk_level}  (confidence: {self.confidence:.2f})\n"
            f"Top Drivers:\n  - {drivers}\n"
            f"Evidence   : {refs}\n"
            f"Reasoning  : {self.reasoning}"
        )


# ---------------------------------------------------------------------------
# Rehabilitation Plan output
# ---------------------------------------------------------------------------

@dataclass
class RehabPlan:
    """Structured output from the Rehabilitation Agent."""
    athlete_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)

    current_stage: str = ""             # e.g. "Early", "Mid", "Late", "Return-to-Play"
    progress_status: str = ""           # e.g. "On Track", "Delayed", "At Risk"
    weekly_exercises: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    progression_criteria: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        exs   = "\n  - ".join(self.weekly_exercises) if self.weekly_exercises else "None"
        rests = "\n  - ".join(self.restrictions) if self.restrictions else "None"
        crit  = "\n  - ".join(self.progression_criteria) if self.progression_criteria else "None"
        return (
            f"Stage      : {self.current_stage}\n"
            f"Progress   : {self.progress_status}\n"
            f"Exercises  :\n  - {exs}\n"
            f"Restrictions:\n  - {rests}\n"
            f"Criteria   :\n  - {crit}\n"
            f"Reasoning  : {self.reasoning}"
        )
