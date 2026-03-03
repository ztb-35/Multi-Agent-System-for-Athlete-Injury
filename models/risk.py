"""
Risk assessment output models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FatigueReport:
    athlete_id: str
    assessed_at: datetime
    fatigue_score: float               # 0–1
    level: RiskLevel
    contributing_factors: list[str] = field(default_factory=list)
    recommended_rest_days: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "athlete_id": self.athlete_id,
            "assessed_at": self.assessed_at.isoformat(),
            "fatigue_score": self.fatigue_score,
            "level": self.level.value,
            "contributing_factors": self.contributing_factors,
            "recommended_rest_days": self.recommended_rest_days,
            "notes": self.notes,
        }


@dataclass
class RiskAssessment:
    athlete_id: str
    assessed_at: datetime
    overall_risk_score: float          # 0–1
    overall_risk_level: RiskLevel

    # Granular risk factors
    musculoskeletal_risk: float = 0.0
    cardiovascular_risk: float = 0.0
    neurological_risk: float = 0.0
    overtraining_risk: float = 0.0

    # Explanations (for human-in-the-loop transparency)
    risk_factors: list[str] = field(default_factory=list)
    protective_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Model ensemble metadata
    model_votes: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0           # agreement across ensemble

    fatigue: FatigueReport | None = None
    session_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "athlete_id": self.athlete_id,
            "assessed_at": self.assessed_at.isoformat(),
            "overall_risk_score": self.overall_risk_score,
            "overall_risk_level": self.overall_risk_level.value,
            "musculoskeletal_risk": self.musculoskeletal_risk,
            "cardiovascular_risk": self.cardiovascular_risk,
            "neurological_risk": self.neurological_risk,
            "overtraining_risk": self.overtraining_risk,
            "risk_factors": self.risk_factors,
            "protective_factors": self.protective_factors,
            "recommendations": self.recommendations,
            "model_votes": self.model_votes,
            "confidence": self.confidence,
            "fatigue": self.fatigue.to_dict() if self.fatigue else None,
            "session_id": self.session_id,
        }
