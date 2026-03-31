"""
Twin Agent – the central state manager and single source of truth.

Responsibilities (paper §2.2):
  - Ingest heterogeneous raw data (mot / trc / yaml / json)
  - Abstract raw session → BiomechanicsSnapshot
  - Maintain versioned AthleteState: baselines, trends, deviations
  - Persist every state version via TwinStore
  - Log every SessionData to SessionStore
  - Trigger downstream specialist agents on NewSessionEvent
"""

from __future__ import annotations
import time
from typing import Optional, TYPE_CHECKING

from models.athlete_state import (
    AthleteState,
    BiomechanicsSnapshot,
    SessionData,
)
from memory.session_store import SessionStore
from memory.twin_store import TwinStore
from utils.data_loader import mot_to_snapshot, load_session_yaml

if TYPE_CHECKING:
    from agents.risk_agent import RiskAgent
    from agents.rehab_agent import RehabAgent


# Metrics tracked in temporal trend history
_TREND_KEYS = [
    "knee_angle_r_mean",
    "knee_angle_l_mean",
    "knee_asymmetry_index",
    "hip_flexion_r_mean",
    "hip_flexion_l_mean",
    "hip_adduction_r_mean",
    "hip_adduction_l_mean",
    "hip_asymmetry_index",
    "ankle_angle_r_mean",
    "ankle_angle_l_mean",
    "pelvis_tilt_mean",
    "lumbar_extension_mean",
    "session_duration_s",
]

# How many sessions to use for baseline rolling average
_BASELINE_WINDOW = 5


class TwinAgent:
    """
    Maintains versioned digital twins for all athletes.

    Usage:
        agent = TwinAgent()

        # Register a new athlete profile (once)
        agent.register_athlete(athlete_id="A01", name="Abigail Savoy",
                               age=22, height_m=1.70, mass_kg=81.6, gender="f")

        # Process a session event (NewSessionEvent)
        session = SessionData(
            athlete_id="A01",
            session_id="S2026_02_03",
            mot_file="data/OpenSimData/Kinematics/Abigail_1.mot",
            yaml_file="data/sessionMetadata.yaml",
        )
        state = agent.process_session(session)
    """

    def __init__(
        self,
        session_store: Optional[SessionStore] = None,
        twin_store: Optional[TwinStore] = None,
        risk_agent: Optional["RiskAgent"] = None,
        rehab_agent: Optional["RehabAgent"] = None,
    ):
        self.session_store = session_store or SessionStore()
        self.twin_store    = twin_store    or TwinStore()
        self.risk_agent    = risk_agent
        self.rehab_agent   = rehab_agent

    # ------------------------------------------------------------------
    # Athlete registration
    # ------------------------------------------------------------------

    def register_athlete(
        self,
        athlete_id: str,
        name: str = "",
        age: Optional[int] = None,
        height_m: float = 0.0,
        mass_kg: float = 0.0,
        gender: str = "",
        sport: str = "",
        injury_history: Optional[list[str]] = None,
        active_injury: Optional[str] = None,
    ) -> AthleteState:
        """
        Create or update an athlete's static profile in the twin store.
        Safe to call multiple times – will merge with existing state.
        """
        existing = self.twin_store.load_latest(athlete_id)
        if existing:
            state = existing
        else:
            state = AthleteState(athlete_id=athlete_id)

        if name:             state.name    = name
        if age is not None:  state.age     = age
        if height_m:         state.height_m = height_m
        if mass_kg:          state.mass_kg  = mass_kg
        if gender:           state.gender   = gender
        if sport:            state.sport    = sport
        if injury_history:   state.injury_history = injury_history
        if active_injury is not None:
            state.active_injury = active_injury

        state.last_updated = time.time()
        self.twin_store.save(state)
        print(f"[TwinAgent] Registered / updated athlete {athlete_id}: {name}")
        return state

    # ------------------------------------------------------------------
    # NewSessionEvent handler
    # ------------------------------------------------------------------

    def process_session(self, session: SessionData) -> AthleteState:
        """
        Handle a NewSessionEvent:
          1. Log raw session to SessionStore
          2. Extract BiomechanicsSnapshot from .mot file
          3. Merge metadata from .yaml if available
          4. Update AthleteState: baseline, trends, deviations
          5. Persist new twin version
          6. Optionally trigger Risk + Rehab agents

        Returns the updated AthleteState.
        """
        print(f"\n[TwinAgent] NewSessionEvent ← athlete={session.athlete_id}  session={session.session_id}")

        # 1. Persist raw session
        self.session_store.save(session)

        # 2. Load current twin state (or create fresh)
        state = self.twin_store.load_latest(session.athlete_id)
        if state is None:
            state = AthleteState(athlete_id=session.athlete_id)
            print(f"[TwinAgent] No existing twin – creating new for {session.athlete_id}")

        # 3. Extract biomechanics snapshot from .mot
        snap: Optional[BiomechanicsSnapshot] = None
        if session.mot_file:
            snap = mot_to_snapshot(session.session_id, session.mot_file)
            print(f"[TwinAgent] Extracted snapshot: {snap.n_frames} frames, "
                  f"knee_asym={snap.knee_asymmetry_index:.1f}%")
        else:
            print("[TwinAgent] No .mot file provided – skipping biomechanics snapshot")

        # 4. Merge session metadata from .yaml
        if session.yaml_file:
            meta = load_session_yaml(session.yaml_file)
            if not state.name and meta.get("subjectID"):
                state.name = meta["subjectID"]
            if not state.height_m and meta.get("height_m"):
                state.height_m = float(meta["height_m"])
            if not state.mass_kg and meta.get("mass_kg"):
                state.mass_kg = float(meta["mass_kg"])
            if not state.gender and meta.get("gender_mf"):
                state.gender = meta["gender_mf"]

        # 5. Update session ID list
        if session.session_id not in state.session_ids:
            state.session_ids.append(session.session_id)

        # 6. Pain / injury notes
        if session.pain_score is not None:
            state.pain_scores.append(session.pain_score)
        if session.injury_notes:
            state.injury_notes_history.append(session.injury_notes)

        # 7. Update snapshot + trends + baseline + deviations
        if snap:
            snap_dict = snap.to_dict()
            state.latest_snapshot = snap_dict

            # Append to trends
            for key in _TREND_KEYS:
                if key not in state.trends:
                    state.trends[key] = []
                state.trends[key].append(snap_dict.get(key, 0.0))

            # Recompute baseline (rolling mean over last N sessions)
            state.baseline = self._compute_baseline(state.trends)

            # Compute deviations (latest − baseline)
            state.deviations = self._compute_deviations(snap_dict, state.baseline)

        # 8. Bump version and persist
        state.version     += 1
        state.last_updated = time.time()
        self.twin_store.save(state)
        print(f"[TwinAgent] Twin updated → version {state.version}")

        # 9. Trigger specialist agents (event-driven)
        self._trigger_specialists(state, session.session_id)

        return state

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_state(self, athlete_id: str) -> Optional[AthleteState]:
        """Return the latest digital twin state for an athlete."""
        return self.twin_store.load_latest(athlete_id)

    def get_state_version(self, athlete_id: str, version: int) -> Optional[AthleteState]:
        """Return a specific historical version."""
        return self.twin_store.load_version(athlete_id, version)

    def list_athletes(self) -> list[str]:
        return self.twin_store.list_athletes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_baseline(self, trends: dict) -> dict:
        """Rolling mean over the last _BASELINE_WINDOW sessions."""
        baseline = {}
        for key, values in trends.items():
            window = values[-_BASELINE_WINDOW:]
            baseline[key] = sum(window) / len(window) if window else 0.0
        return baseline

    def _compute_deviations(self, snap_dict: dict, baseline: dict) -> dict:
        """Latest snapshot value minus baseline for each metric."""
        deviations = {}
        for key in _TREND_KEYS:
            latest_val   = snap_dict.get(key, 0.0)
            baseline_val = baseline.get(key, 0.0)
            deviations[key] = {
                "current":  round(latest_val, 3),
                "baseline": round(baseline_val, 3),
                "delta":    round(latest_val - baseline_val, 3),
                "pct_change": round(
                    (latest_val - baseline_val) / abs(baseline_val) * 100
                    if baseline_val != 0 else 0.0, 1
                ),
            }
        return deviations

    def _trigger_specialists(self, state: AthleteState, session_id: str) -> None:
        """Asynchronously notify Risk and Rehab agents after twin update."""
        if self.risk_agent:
            print(f"[TwinAgent] → triggering RiskAgent for {state.athlete_id}")
            result = self.risk_agent.assess(state, session_id)
            print(f"[TwinAgent]   Risk result: {result.risk_level} ({result.confidence:.2f})")

        if self.rehab_agent and state.active_injury:
            print(f"[TwinAgent] → triggering RehabAgent for {state.athlete_id}")
            plan = self.rehab_agent.plan(state, session_id)
            print(f"[TwinAgent]   Rehab stage: {plan.current_stage} / {plan.progress_status}")
