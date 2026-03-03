"""
RiskToolkit – tools available to the RiskAgent.

Tools:
  - compute_risk_score        ensemble injury risk estimation
  - analyze_fatigue           fatigue / recovery state analysis
  - compare_to_baseline       deviation from individual baseline
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any

import config
from models.risk import RiskLevel
from .base_tool import BaseTool


def _risk_level(score: float) -> RiskLevel:
    if score >= 0.85:
        return RiskLevel.CRITICAL
    if score >= config.DEFAULT_RISK_HIGH_THRESHOLD:
        return RiskLevel.HIGH
    if score >= config.DEFAULT_RISK_MEDIUM_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class ComputeRiskScore(BaseTool):
    name = "compute_risk_score"
    description = (
        "Compute an ensemble injury risk score for an athlete given their current "
        "twin state and recent session metrics. Returns an overall 0–1 score, "
        "per-domain scores, risk factors, and recommendations."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "twin_state": {"type": "object", "description": "Current DigitalTwin snapshot dict"},
            "recent_sessions": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of recent session dicts (last 7–28 days)",
            },
            "athlete_baseline": {"type": "object", "description": "AthleteBaseline dict"},
        },
        "required": ["athlete_id", "twin_state"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        twin = tool_input.get("twin_state", {})
        baseline = tool_input.get("athlete_baseline", {})
        sessions = tool_input.get("recent_sessions", [])

        factors: list[str] = []
        protective: list[str] = []
        recommendations: list[str] = []
        model_votes: dict[str, float] = {}

        # ── ACWR model ───────────────────────────────────────────────────
        acwr = twin.get("acwr")
        if acwr is not None:
            if acwr > 1.5:
                acwr_score = min(1.0, (acwr - 1.5) / 0.5 * 0.8 + 0.5)
                factors.append(f"ACWR = {acwr:.2f} (spike zone > 1.5)")
                recommendations.append("Reduce training load for 3–5 days")
            elif acwr > 1.3:
                acwr_score = 0.4
                factors.append(f"ACWR = {acwr:.2f} (elevated)")
            elif 0.8 <= acwr <= 1.3:
                acwr_score = 0.15
                protective.append(f"ACWR = {acwr:.2f} (sweet spot 0.8–1.3)")
            else:
                acwr_score = 0.25
                factors.append(f"ACWR = {acwr:.2f} (underloading)")
        else:
            acwr_score = 0.3  # unknown
        model_votes["acwr_model"] = acwr_score

        # ── HRV model ────────────────────────────────────────────────────
        current_hrv = twin.get("current_hrv")
        baseline_hrv = baseline.get("hrv_baseline")
        if current_hrv and baseline_hrv and baseline_hrv > 0:
            hrv_ratio = current_hrv / baseline_hrv
            hrv_score = max(0.0, min(1.0, (1 - hrv_ratio) * 1.5))
            if hrv_ratio < 0.8:
                factors.append(f"HRV suppressed ({current_hrv:.1f} vs baseline {baseline_hrv:.1f} ms)")
            elif hrv_ratio > 1.1:
                protective.append("HRV above baseline – good recovery")
        else:
            hrv_score = 0.3
        model_votes["hrv_model"] = hrv_score

        # ── Biomechanical model ──────────────────────────────────────────
        si = twin.get("symmetry_index")
        bio_alerts = twin.get("biomechanical_alerts", [])
        bio_score = 0.0
        if si is not None and si < 0.85:
            bio_score = (0.85 - si) * 3.0
            factors.append(f"Low movement symmetry index: {si:.2f}")
        if bio_alerts:
            bio_score = min(1.0, bio_score + 0.1 * len(bio_alerts))
            factors.extend(bio_alerts)
        model_votes["biomechanical_model"] = min(1.0, bio_score)

        # ── Injury history model ─────────────────────────────────────────
        active_injury = twin.get("active_injury")
        if active_injury:
            factors.append(f"Active injury: {active_injury}")
            model_votes["injury_history_model"] = 0.75
        else:
            model_votes["injury_history_model"] = 0.1

        # ── Ensemble aggregation (simple weighted average) ───────────────
        weights = {
            "acwr_model": 0.35,
            "hrv_model": 0.25,
            "biomechanical_model": 0.25,
            "injury_history_model": 0.15,
        }
        overall = sum(model_votes[m] * weights[m] for m in model_votes)
        confidence = 1.0 - (max(model_votes.values()) - min(model_votes.values()))

        level = _risk_level(overall)
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            recommendations.append("Flag for immediate clinical review")

        return {
            "athlete_id": tool_input["athlete_id"],
            "assessed_at": datetime.utcnow().isoformat(),
            "overall_risk_score": round(overall, 3),
            "overall_risk_level": level.value,
            "model_votes": {k: round(v, 3) for k, v in model_votes.items()},
            "confidence": round(confidence, 3),
            "risk_factors": factors,
            "protective_factors": protective,
            "recommendations": recommendations,
        }


class AnalyzeFatigue(BaseTool):
    name = "analyze_fatigue"
    description = (
        "Analyse fatigue and recovery state from recent session data and HRV trends. "
        "Returns a fatigue score 0–1 and recommended rest days."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "twin_state": {"type": "object"},
            "recent_sessions": {"type": "array", "items": {"type": "object"}},
            "athlete_baseline": {"type": "object"},
        },
        "required": ["athlete_id", "twin_state"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        twin = tool_input.get("twin_state", {})
        baseline = tool_input.get("athlete_baseline", {})
        sessions = tool_input.get("recent_sessions", [])

        factors: list[str] = []
        score = 0.0

        # Sleep deficit
        sleep_scores = [
            s.get("wearable", {}).get("sleep_quality_score", 1.0)
            for s in sessions if s.get("wearable")
        ]
        if sleep_scores:
            avg_sleep = sum(sleep_scores) / len(sleep_scores)
            if avg_sleep < 0.6:
                score += 0.3
                factors.append(f"Poor average sleep quality ({avg_sleep:.2f})")

        # HRV suppression
        current_hrv = twin.get("current_hrv")
        baseline_hrv = baseline.get("hrv_baseline")
        if current_hrv and baseline_hrv:
            hrv_ratio = current_hrv / baseline_hrv
            if hrv_ratio < 0.85:
                score += 0.3
                factors.append(f"HRV suppressed to {hrv_ratio:.0%} of baseline")

        # Consecutive high-load days
        high_load_days = sum(
            1 for s in sessions[-7:]
            if s.get("wearable", {}).get("training_load", 0) >
               baseline.get("daily_training_load_avg", 500) * 1.2
        )
        if high_load_days >= 4:
            score += 0.25
            factors.append(f"{high_load_days} high-load days in past week")

        score = min(1.0, score)
        level = _risk_level(score)
        rest_days = 0
        if score > 0.7:
            rest_days = 2
        elif score > 0.4:
            rest_days = 1

        return {
            "athlete_id": tool_input["athlete_id"],
            "assessed_at": datetime.utcnow().isoformat(),
            "fatigue_score": round(score, 3),
            "level": level.value,
            "contributing_factors": factors,
            "recommended_rest_days": rest_days,
        }


class CompareToBaseline(BaseTool):
    name = "compare_to_baseline"
    description = (
        "Compare current athlete metrics against their individual baseline. "
        "Returns per-metric deviation scores and flags significant changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "athlete_id": {"type": "string"},
            "current_metrics": {
                "type": "object",
                "description": "Dict of metric_name → current value",
            },
            "baseline": {
                "type": "object",
                "description": "AthleteBaseline dict",
            },
        },
        "required": ["athlete_id", "current_metrics", "baseline"],
    }

    def run(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        current = tool_input["current_metrics"]
        baseline = tool_input["baseline"]

        comparisons: list[dict] = []
        for metric, baseline_val in baseline.items():
            if baseline_val is None or not isinstance(baseline_val, (int, float)):
                continue
            current_val = current.get(metric)
            if current_val is None:
                continue
            pct_change = (current_val - baseline_val) / baseline_val * 100 if baseline_val else 0
            comparisons.append({
                "metric": metric,
                "baseline": baseline_val,
                "current": current_val,
                "pct_change": round(pct_change, 1),
                "flagged": abs(pct_change) > 15,
            })

        return {
            "athlete_id": tool_input["athlete_id"],
            "comparisons": comparisons,
            "flagged_metrics": [c["metric"] for c in comparisons if c["flagged"]],
        }


class RiskToolkit:
    """Container exposing all RiskAgent tools."""
    def __init__(self) -> None:
        self.compute_risk_score = ComputeRiskScore()
        self.analyze_fatigue = AnalyzeFatigue()
        self.compare_to_baseline = CompareToBaseline()

    def all_tools(self) -> list[BaseTool]:
        return [self.compute_risk_score, self.analyze_fatigue, self.compare_to_baseline]

    def schemas(self) -> list[dict]:
        return [t.to_anthropic_schema() for t in self.all_tools()]
