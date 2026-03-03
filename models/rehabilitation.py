"""
Rehabilitation planning and return-to-sport models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class RehabMilestone:
    milestone_id: str
    name: str                              # e.g. "Full weight-bearing"
    target_day: int                        # days from injury
    criteria: list[str]                    # measurable criteria to pass
    achieved: bool = False
    achieved_date: Optional[date] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "target_day": self.target_day,
            "criteria": self.criteria,
            "achieved": self.achieved,
            "achieved_date": self.achieved_date.isoformat() if self.achieved_date else None,
            "notes": self.notes,
        }


@dataclass
class RehabPlan:
    plan_id: str
    athlete_id: str
    injury_id: str
    created_at: datetime
    injury_description: str
    total_expected_days: int
    phases: list[dict]             # [{phase, duration_days, goals, exercises}]
    milestones: list[RehabMilestone] = field(default_factory=list)
    current_phase: int = 0         # 0-indexed
    progress_notes: list[str] = field(default_factory=list)

    def next_milestones(self) -> list[RehabMilestone]:
        return [m for m in self.milestones if not m.achieved]

    def completion_pct(self, current_day: int) -> float:
        if self.total_expected_days <= 0:
            return 0.0
        return min(1.0, current_day / self.total_expected_days)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "athlete_id": self.athlete_id,
            "injury_id": self.injury_id,
            "created_at": self.created_at.isoformat(),
            "injury_description": self.injury_description,
            "total_expected_days": self.total_expected_days,
            "phases": self.phases,
            "milestones": [m.to_dict() for m in self.milestones],
            "current_phase": self.current_phase,
            "progress_notes": self.progress_notes,
        }


@dataclass
class ReturnToSportAssessment:
    athlete_id: str
    assessed_at: datetime
    readiness_score: float         # 0–1; >= threshold → cleared
    cleared: bool
    criteria_met: list[str]
    criteria_not_met: list[str]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "athlete_id": self.athlete_id,
            "assessed_at": self.assessed_at.isoformat(),
            "readiness_score": self.readiness_score,
            "cleared": self.cleared,
            "criteria_met": self.criteria_met,
            "criteria_not_met": self.criteria_not_met,
            "notes": self.notes,
        }
