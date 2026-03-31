"""
Rehabilitation Agent – Recovery Planning & Progression (paper §2.4).

Responsibilities:
  - Receive structured AthleteState from TwinAgent
  - Stage-aware reasoning (Early / Mid / Late / Return-to-Play)
  - Knowledge retrieval: built-in protocol guidelines + evidence refs
  - Causal/counterfactual reasoning: what-if workload adjustments
  - Generate personalized weekly plans with progression criteria
  - Human-in-the-loop feedback for medical staff corrections
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Optional

import anthropic

from models.athlete_state import AthleteState, RehabPlan


# ---------------------------------------------------------------------------
# Embedded rehabilitation knowledge base (knowledge retrieval, paper §2.4)
# ---------------------------------------------------------------------------

REHAB_PROTOCOLS = {
    "ACL_recon": {
        "stages": {
            "Early (0-6 weeks)": {
                "criteria_to_progress": [
                    "Full ROM (0-120°)",
                    "Quad strength >60% of contralateral",
                    "No effusion",
                    "Normal gait pattern",
                ],
                "allowed_exercises": [
                    "Straight leg raises",
                    "Quad sets",
                    "Heel slides",
                    "Stationary cycling (low resistance)",
                    "Pool walking",
                ],
                "restrictions": [
                    "No cutting/pivoting",
                    "No running",
                    "No deep squats (>90°)",
                ],
            },
            "Mid (6-14 weeks)": {
                "criteria_to_progress": [
                    "Quad strength >80% of contralateral",
                    "Hop test symmetry >85%",
                    "Knee asymmetry index <10%",
                    "No pain with jogging",
                ],
                "allowed_exercises": [
                    "Single-leg squats",
                    "Lateral band walks",
                    "Step-downs",
                    "Light jogging (straight line)",
                    "Eccentric hamstring curls",
                ],
                "restrictions": [
                    "No cutting drills",
                    "No lateral hops",
                    "Max load 80% BW on reconstructed leg",
                ],
            },
            "Late (14-24 weeks)": {
                "criteria_to_progress": [
                    "Quad strength >90% of contralateral",
                    "Triple hop symmetry >90%",
                    "Knee asymmetry index <5%",
                    "Psychological readiness score >65/100",
                ],
                "allowed_exercises": [
                    "Lateral hops",
                    "Cutting drills (controlled speed)",
                    "Agility ladders",
                    "Sport-specific movement patterns",
                    "Deceleration training",
                ],
                "restrictions": [
                    "No full-speed cutting",
                    "No contact sport participation",
                ],
            },
            "Return-to-Play (24+ weeks)": {
                "criteria_to_progress": [
                    "Quad/Ham strength within 10% of contralateral",
                    "All hop tests >95% symmetry",
                    "Full sport-specific training without pain",
                    "Team physician clearance",
                ],
                "allowed_exercises": [
                    "Full training load",
                    "Contact drills",
                    "Match simulations",
                ],
                "restrictions": [],
            },
        }
    },
    "hamstring_strain": {
        "stages": {
            "Acute (0-2 weeks)": {
                "criteria_to_progress": ["No pain at rest", "Pain-free passive ROM"],
                "allowed_exercises": ["Isometric holds", "Gentle stretching", "Pool therapy"],
                "restrictions": ["No sprinting", "No heavy loading"],
            },
            "Subacute (2-6 weeks)": {
                "criteria_to_progress": ["Pain-free jogging", "80% eccentric strength"],
                "allowed_exercises": ["Nordic curls", "RDLs", "Progressive running"],
                "restrictions": ["No maximal sprinting"],
            },
            "Return-to-Sport (6+ weeks)": {
                "criteria_to_progress": ["Full sprinting without pain", "Hop symmetry >90%"],
                "allowed_exercises": ["Maximal sprinting", "Sport drills"],
                "restrictions": [],
            },
        }
    },
}

_GENERIC_PROTOCOL_NOTE = """No specific protocol loaded for this injury type.
Provide conservative, evidence-based general rehabilitation principles."""


class RehabAgent:
    """
    Stage-aware rehabilitation planner with Claude LLM reasoning.

    Usage:
        agent = RehabAgent()
        plan = agent.plan(athlete_state, session_id="S2026_02_03")
        print(plan.pretty())
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        feedback_dir: str = "memory/rehab_feedback",
    ):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, state: AthleteState, session_id: str) -> RehabPlan:
        """
        Generate a personalized rehabilitation plan for the athlete's current state.
        """
        return self._llm_plan(state, session_id)

    def counterfactual(
        self,
        state: AthleteState,
        intervention: str,
        session_id: str = "hypothetical",
    ) -> str:
        """
        Causal/counterfactual reasoning (paper §2.4):
        Ask "what would happen if <intervention>?"
        Example intervention: "reduce workload by 30% for 2 weeks"
        """
        prompt = self._build_counterfactual_prompt(state, intervention)
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"[RehabAgent] Counterfactual reasoning failed: {e}"

    def submit_feedback(
        self,
        athlete_id: str,
        session_id: str,
        corrected_stage: str,
        notes: str = "",
    ) -> None:
        """Medical staff feedback for HITL refinement."""
        record = {
            "athlete_id": athlete_id,
            "session_id": session_id,
            "corrected_stage": corrected_stage,
            "notes": notes,
            "timestamp": time.time(),
        }
        fb_path = self.feedback_dir / f"{athlete_id}.ndjson"
        with open(fb_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[RehabAgent] Feedback recorded for {athlete_id}/{session_id}")

    # ------------------------------------------------------------------
    # LLM planning
    # ------------------------------------------------------------------

    def _get_protocol(self, injury: Optional[str]) -> str:
        """Retrieve relevant rehabilitation protocol knowledge."""
        if not injury:
            return _GENERIC_PROTOCOL_NOTE
        for key, proto in REHAB_PROTOCOLS.items():
            if key.lower() in (injury or "").lower():
                return json.dumps(proto["stages"], indent=2)
        return _GENERIC_PROTOCOL_NOTE

    def _build_prompt(self, state: AthleteState, session_id: str) -> str:
        protocol_text = self._get_protocol(state.active_injury)

        # Key asymmetry metrics for stage classification
        asym_info = ""
        if state.latest_snapshot:
            snap = state.latest_snapshot
            asym_info = (
                f"Knee asymmetry index: {snap.get('knee_asymmetry_index', 'N/A'):.1f}%\n"
                f"Hip asymmetry index:  {snap.get('hip_asymmetry_index', 'N/A'):.1f}%\n"
                f"Knee angle R mean:    {snap.get('knee_angle_r_mean', 'N/A'):.1f}°\n"
                f"Knee angle L mean:    {snap.get('knee_angle_l_mean', 'N/A'):.1f}°\n"
                f"Hip adduction R mean: {snap.get('hip_adduction_r_mean', 'N/A'):.1f}°\n"
                f"Session duration:     {snap.get('session_duration_s', 'N/A'):.1f}s"
            )

        trend_asym = state.trends.get("knee_asymmetry_index", [])[-5:]

        return f"""You are an expert sports rehabilitation AI. Generate a personalized rehabilitation plan.

## Athlete Profile
- ID: {state.athlete_id} | Name: {state.name}
- Age: {state.age} | Sport: {state.sport or 'Not specified'}
- Active Injury: {state.active_injury or 'None'}
- Injury History: {', '.join(state.injury_history) or 'None'}
- Sessions since injury onset: {len(state.session_ids)}
- Recent pain scores (VAS 0-10): {state.pain_scores[-3:] if state.pain_scores else 'N/A'}
- Latest notes: {state.injury_notes_history[-1] if state.injury_notes_history else 'None'}

## Current Biomechanical Status
{asym_info}

## Asymmetry Trend (last 5 sessions)
knee_asymmetry_index: {trend_asym}
{'↑ WORSENING' if len(trend_asym) >= 2 and trend_asym[-1] > trend_asym[-2] else '↓ Improving or stable'}

## Evidence-Based Protocol for "{state.active_injury or 'general'}"
{protocol_text}

## Task
Generate a personalized rehabilitation plan. Respond with ONLY a JSON object:
{{
  "current_stage": "<stage name>",
  "progress_status": "On Track" | "Delayed" | "At Risk" | "Ready to Progress",
  "weekly_exercises": [<string>, ...],
  "restrictions": [<string>, ...],
  "progression_criteria": [<string>, ...],
  "evidence_refs": ["{session_id}", ...],
  "reasoning": "<2-3 sentence clinical rationale>"
}}

Base stage classification on biomechanical data (asymmetry index, strength proxy) vs. protocol criteria.
Prioritize safety and conservative progression."""

    def _build_counterfactual_prompt(self, state: AthleteState, intervention: str) -> str:
        return f"""You are a sports medicine AI performing causal/counterfactual reasoning.

## Athlete
- ID: {state.athlete_id} | Active Injury: {state.active_injury}
- Current stage sessions: {len(state.session_ids)}
- Knee asymmetry trend: {state.trends.get('knee_asymmetry_index', [])[-5:]}
- Pain scores: {state.pain_scores[-3:] if state.pain_scores else 'N/A'}

## Hypothetical Intervention
"{intervention}"

Reason through how this intervention would likely affect:
1. Recovery trajectory (weeks to next stage)
2. Biomechanical adaptations (asymmetry, strength)
3. Re-injury risk
4. Return-to-play timeline

Provide a concise 3-5 sentence counterfactual analysis."""

    def _llm_plan(self, state: AthleteState, session_id: str) -> RehabPlan:
        prompt = self._build_prompt(state, session_id)
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)

            return RehabPlan(
                athlete_id=state.athlete_id,
                session_id=session_id,
                current_stage=data.get("current_stage", "Unknown"),
                progress_status=data.get("progress_status", "Unknown"),
                weekly_exercises=data.get("weekly_exercises", []),
                restrictions=data.get("restrictions", []),
                progression_criteria=data.get("progression_criteria", []),
                evidence_refs=data.get("evidence_refs", [session_id]),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            print(f"[RehabAgent] LLM call failed: {e}. Using fallback.")
            return self._fallback_plan(state, session_id)

    def _fallback_plan(self, state: AthleteState, session_id: str) -> RehabPlan:
        """Rule-based fallback when LLM is unavailable."""
        n = len(state.session_ids)
        if n < 6:
            stage = "Early"
        elif n < 15:
            stage = "Mid"
        elif n < 24:
            stage = "Late"
        else:
            stage = "Return-to-Play"

        return RehabPlan(
            athlete_id=state.athlete_id,
            session_id=session_id,
            current_stage=stage,
            progress_status="On Track",
            weekly_exercises=["Consult rehabilitation protocol for stage-appropriate exercises"],
            restrictions=["Follow medical staff guidance"],
            progression_criteria=["Assessed by licensed physiotherapist"],
            evidence_refs=[session_id],
            reasoning="Fallback plan – LLM unavailable. Stage estimated from session count.",
        )
