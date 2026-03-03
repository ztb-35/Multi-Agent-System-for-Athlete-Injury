"""
RehabToolkit – tools available to the RehabilitationAgent.

Tools:
  - generate_rehab_plan             create phased rehabilitation plan
  - assess_recovery_progress        evaluate milestone achievement
  - check_return_to_sport_readiness RTS clearance decision support
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any

import config
from .base_tool import BaseTool


class GenerateRehabPlan(BaseTool):
    name = "generate_rehab_plan"
    description = (
        "Generate a phased rehabilitation plan for a given injury. "
        "Returns a structured plan with phases, milestones, and criteria."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "injury_id": {"type": "string"},
            "injury_description": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["mild", "moderate", "severe"],
            },
            "sport": {"type": "string"},
            "athlete_age": {"type": "integer"},
        },
        "required": ["athlete_id", "injury_id", "injury_description", "severity"],
    }

    _PHASES = {
        "mild": [
            {"phase": 1, "name": "Pain & Swelling Control", "duration_days": 5,
             "goals": ["Reduce inflammation", "Restore ROM"],
             "exercises": ["RICE", "Gentle ROM", "Isometrics"]},
            {"phase": 2, "name": "Strength & Proprioception", "duration_days": 10,
             "goals": ["Restore 80% strength", "Neuromuscular control"],
             "exercises": ["Progressive resistance", "Balance board"]},
            {"phase": 3, "name": "Sport-Specific Training", "duration_days": 7,
             "goals": ["Full sport readiness"],
             "exercises": ["Sport drills", "Agility"]},
        ],
        "moderate": [
            {"phase": 1, "name": "Acute Management", "duration_days": 10,
             "goals": ["Protect tissue", "Reduce pain"],
             "exercises": ["POLICE protocol", "Pool therapy"]},
            {"phase": 2, "name": "Functional Restoration", "duration_days": 21,
             "goals": ["Restore strength symmetry"],
             "exercises": ["OKC & CKC strengthening", "Neuromuscular re-education"]},
            {"phase": 3, "name": "Return to Training", "duration_days": 14,
             "goals": ["Full training load tolerance"],
             "exercises": ["Progressive running", "Team drills"]},
            {"phase": 4, "name": "Return to Sport", "duration_days": 7,
             "goals": ["Match readiness"],
             "exercises": ["Full practice", "Contact drills"]},
        ],
        "severe": [
            {"phase": 1, "name": "Post-Surgical / Immobilisation", "duration_days": 21,
             "goals": ["Wound healing", "Prevent atrophy"],
             "exercises": ["Quad sets", "Ankle pumps", "CPM"]},
            {"phase": 2, "name": "Early Mobilisation", "duration_days": 28,
             "goals": ["Full weight-bearing", "0–90° ROM"],
             "exercises": ["Gait training", "Pool walking", "SLR"]},
            {"phase": 3, "name": "Strength Building", "duration_days": 42,
             "goals": [">90% limb symmetry index"],
             "exercises": ["Leg press", "Step-ups", "Hamstring curls"]},
            {"phase": 4, "name": "Plyometrics & Agility", "duration_days": 28,
             "goals": ["Single-leg hop tests", "Agility benchmarks"],
             "exercises": ["Jump training", "Cutting drills"]},
            {"phase": 5, "name": "Return to Sport", "duration_days": 14,
             "goals": ["Full competitive readiness"],
             "exercises": ["Full training", "Match simulation"]},
        ],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        severity = tool_input["severity"]
        phases = self._PHASES.get(severity, self._PHASES["moderate"])
        total_days = sum(p["duration_days"] for p in phases)

        milestones = []
        cumulative = 0
        for p in phases:
            cumulative += p["duration_days"]
            milestones.append({
                "milestone_id": str(uuid.uuid4())[:8],
                "name": f"Complete {p['name']}",
                "target_day": cumulative,
                "criteria": p["goals"],
                "achieved": False,
            })

        return {
            "plan_id": str(uuid.uuid4())[:12],
            "athlete_id": tool_input["athlete_id"],
            "injury_id": tool_input["injury_id"],
            "created_at": datetime.utcnow().isoformat(),
            "injury_description": tool_input["injury_description"],
            "severity": severity,
            "total_expected_days": total_days,
            "phases": phases,
            "milestones": milestones,
        }


class AssessRecoveryProgress(BaseTool):
    name = "assess_recovery_progress"
    description = (
        "Evaluate current rehabilitation progress against milestones and the plan. "
        "Returns achieved milestones, next targets, and a progress percentage."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "rehab_plan": {"type": "object", "description": "RehabPlan dict"},
            "current_day": {"type": "integer"},
            "session_data": {"type": "object", "description": "Latest session or clinical note"},
        },
        "required": ["athlete_id", "rehab_plan", "current_day"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        plan = tool_input["rehab_plan"]
        current_day = tool_input["current_day"]
        total = plan.get("total_expected_days", 1)
        progress_pct = min(100, round(current_day / total * 100, 1))

        achieved = [m for m in plan.get("milestones", []) if m["target_day"] <= current_day]
        pending = [m for m in plan.get("milestones", []) if m["target_day"] > current_day]

        return {
            "athlete_id": tool_input["athlete_id"],
            "current_day": current_day,
            "progress_pct": progress_pct,
            "milestones_achieved": len(achieved),
            "milestones_total": len(plan.get("milestones", [])),
            "next_milestone": pending[0] if pending else None,
            "on_track": current_day <= total,
        }


class CheckReturnToSportReadiness(BaseTool):
    name = "check_return_to_sport_readiness"
    description = (
        "Evaluate athlete readiness to return to full sport participation. "
        "Applies evidence-based RTS criteria and returns a readiness score + clearance decision."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "twin_state": {"type": "object"},
            "rehab_plan": {"type": "object"},
            "clinical_assessment": {
                "type": "object",
                "description": "Latest clinical note or functional test results",
            },
        },
        "required": ["athlete_id", "twin_state"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        twin = tool_input.get("twin_state", {})
        criteria_met: list[str] = []
        criteria_not_met: list[str] = []
        score = 0.0
        weight_per_criterion = 0.2  # 5 criteria → max 1.0

        # 1. Symmetry
        si = twin.get("symmetry_index", 0)
        if si >= 0.90:
            criteria_met.append(f"Limb symmetry index ≥ 90% (actual: {si:.0%})")
            score += weight_per_criterion
        else:
            criteria_not_met.append(f"Limb symmetry index < 90% (actual: {si:.0%})")

        # 2. No active biomechanical alerts
        alerts = twin.get("biomechanical_alerts", [])
        if not alerts:
            criteria_met.append("No biomechanical movement alerts")
            score += weight_per_criterion
        else:
            criteria_not_met.append(f"Unresolved biomechanical alerts: {alerts}")

        # 3. Active injury cleared
        if not twin.get("active_injury"):
            criteria_met.append("No active injury recorded")
            score += weight_per_criterion
        else:
            criteria_not_met.append(f"Active injury still recorded: {twin['active_injury']}")

        # 4. HRV recovery
        if twin.get("current_hrv") and twin.get("current_hrv", 0) > 30:
            criteria_met.append("HRV within acceptable range")
            score += weight_per_criterion
        else:
            criteria_not_met.append("HRV below acceptable range")

        # 5. Rehab plan completion
        plan = tool_input.get("rehab_plan", {})
        rehab_day = twin.get("rehabilitation_day", 0)
        total_days = plan.get("total_expected_days", 0)
        if total_days > 0 and rehab_day >= total_days * 0.9:
            criteria_met.append("≥ 90% of rehabilitation plan completed")
            score += weight_per_criterion
        else:
            criteria_not_met.append("Rehabilitation plan not sufficiently completed")

        cleared = score >= config.RETURN_TO_SPORT_MIN_SCORE
        return {
            "athlete_id": tool_input["athlete_id"],
            "assessed_at": datetime.utcnow().isoformat(),
            "readiness_score": round(score, 2),
            "cleared": cleared,
            "criteria_met": criteria_met,
            "criteria_not_met": criteria_not_met,
        }


class RehabToolkit:
    """Container exposing all RehabilitationAgent tools."""
    def __init__(self) -> None:
        self.generate_rehab_plan = GenerateRehabPlan()
        self.assess_recovery_progress = AssessRecoveryProgress()
        self.check_return_to_sport_readiness = CheckReturnToSportReadiness()

    def all_tools(self) -> list[BaseTool]:
        return [
            self.generate_rehab_plan,
            self.assess_recovery_progress,
            self.check_return_to_sport_readiness,
        ]

    def schemas(self) -> list[dict]:
        return [t.to_anthropic_schema() for t in self.all_tools()]
