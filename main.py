"""
Multi-Agent Digital Twin System – Demo Entry Point
===================================================
Implements the blueprint from:
  "A Multi-Agent Digital Twin Blueprint for Athlete Injury Risk Assessment
   and Rehabilitation Planning" (ICLR 2026 Workshop on MALGAI)

Architecture:
  Raw Data → TwinAgent → AthleteState
                ↓               ↓
           RiskAgent      RehabAgent
                ↘               ↙
              DecisionAgent (NLP)
                    ↕
            Coaches / Medical Staff

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python main.py

Or run without an API key for offline demo mode.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.athlete_state import SessionData
from memory.session_store import SessionStore
from memory.twin_store import TwinStore
from agents.twin_agent import TwinAgent
from agents.risk_agent import RiskAgent, VideoKneePoseAnalyzer, _MEDIAPIPE_AVAILABLE
from agents.rehab_agent import RehabAgent
from agents.decision_agent import DecisionAgent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ATHLETE_ID   = "A01"
ATHLETE_NAME = "Abigail Savoy"

DATA_DIR     = Path("data")
KINEMATIC_SESSIONS = [
    ("S2026_01_01", "OpenSimData/Kinematics/Abigail.mot"),
    ("S2026_01_08", "OpenSimData/Kinematics/Abigail_1.mot"),
    ("S2026_01_15", "OpenSimData/Kinematics/Abigail_2.mot"),
    ("S2026_01_22", "OpenSimData/Kinematics/Abigail_3.mot"),
    ("S2026_01_29", "OpenSimData/Kinematics/Abigail_4.mot"),
    ("S2026_02_03", "OpenSimData/Kinematics/Abigail_5.mot"),  # ← "today's" session
]

YAML_FILE = str(DATA_DIR / "sessionMetadata.yaml")

# ---------------------------------------------------------------------------
# Video inputs for ACL knee-flexion analysis
# Map session_id → video file path (set to None to skip a session)
# ---------------------------------------------------------------------------
VIDEO_INPUTS: dict[str, str | None] = {
    "S2026_01_01": "data/Videos/Cam1/InputMedia/Abigail/Abigail_sync.mp4"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(title: str = "") -> None:
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


def demo_offline_mode(twin_agent: TwinAgent) -> None:
    """Run a quick offline demo without the Claude API."""
    separator("OFFLINE DEMO – No Claude API key detected")

    state = twin_agent.get_state(ATHLETE_ID)
    if state is None:
        print("No athlete data. Run with sessions first.")
        return

    print(f"\nAthlete      : {state.name} ({state.athlete_id})")
    print(f"Twin Version : v{state.version}")
    print(f"Sessions     : {len(state.session_ids)}")
    print(f"Active Injury: {state.active_injury or 'None'}")

    if state.latest_snapshot:
        s = state.latest_snapshot
        print(f"\nLatest Biomechanics Snapshot:")
        print(f"  knee_angle_r_mean  : {s.get('knee_angle_r_mean', 0):.1f}°")
        print(f"  knee_angle_l_mean  : {s.get('knee_angle_l_mean', 0):.1f}°")
        print(f"  knee_asymmetry_idx : {s.get('knee_asymmetry_index', 0):.1f}%")
        print(f"  hip_adduction_r    : {s.get('hip_adduction_r_mean', 0):.1f}°")
        print(f"  session_duration   : {s.get('session_duration_s', 0):.1f}s")

    if state.deviations:
        print(f"\nNotable Deviations from Baseline:")
        for k, v in state.deviations.items():
            pct = v.get("pct_change", 0)
            if abs(pct) > 10:
                print(f"  {k:35s}: {v['current']:.2f} vs baseline {v['baseline']:.2f} ({pct:+.1f}%)")

    asym_trend = state.trends.get("knee_asymmetry_index", [])
    if asym_trend:
        print(f"\nKnee Asymmetry Trend: {[round(x, 1) for x in asym_trend]}")
        direction = "↑ Worsening" if asym_trend[-1] > asym_trend[0] else "↓ Improving"
        print(f"Trend Direction     : {direction}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    separator("Multi-Agent Digital Twin System")
    print("Paper: 'A Multi-Agent Digital Twin Blueprint for Athlete Injury")
    print("        Risk Assessment and Rehabilitation Planning'")
    print("       ICLR 2026 Workshop on MALGAI | Taibiao Zhao, LSU")

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"\nANTHROPIC_API_KEY : {'✓ detected' if has_api_key else '✗ not set (offline demo mode)'}")

    # ------------------------------------------------------------------
    # 1. Instantiate memory stores
    # ------------------------------------------------------------------
    session_store = SessionStore(store_dir="memory/sessions")
    twin_store    = TwinStore(store_dir="memory/twins")

    # ------------------------------------------------------------------
    # 2. Instantiate specialist agents
    # ------------------------------------------------------------------
    risk_agent  = RiskAgent(model="claude-haiku-4-5-20251001")
    rehab_agent = RehabAgent(model="claude-haiku-4-5-20251001")

    # ------------------------------------------------------------------
    # 3. Instantiate Twin Agent (wires specialist agents for event triggers)
    # ------------------------------------------------------------------
    twin_agent = TwinAgent(
        session_store=session_store,
        twin_store=twin_store,
        risk_agent=risk_agent  if has_api_key else None,
        rehab_agent=rehab_agent if has_api_key else None,
    )

    # ------------------------------------------------------------------
    # 4. Register athlete profile
    # ------------------------------------------------------------------
    separator("Step 1: Register Athlete Profile")
    twin_agent.register_athlete(
        athlete_id=ATHLETE_ID,
        name=ATHLETE_NAME,
        age=22,
        height_m=1.70,
        mass_kg=81.6,
        gender="f",
        sport="soccer",
        injury_history=["Left ACL reconstruction (6 months ago)"],
        active_injury="ACL_recon",
    )

    # ------------------------------------------------------------------
    # 5. Process session events (NewSessionEvent)
    # ------------------------------------------------------------------
    separator("Step 2: Process Session Events (NewSessionEvent)")

    pain_progression = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]  # improving pain
    notes_progression = [
        "Early stage – quad weakness noted",
        "ROM improving, slight knee flexion deficit",
        "Gait pattern normalizing",
        "Single-leg squat attempted – mild valgus",
        "Strength asymmetry reducing",
        "Feels ready for lateral drills – physician review needed",
    ]

    for i, (session_id, rel_mot_path) in enumerate(KINEMATIC_SESSIONS):
        mot_path = str(DATA_DIR / rel_mot_path)
        session = SessionData(
            athlete_id=ATHLETE_ID,
            session_id=session_id,
            mot_file=mot_path if Path(mot_path).exists() else None,
            yaml_file=YAML_FILE if Path(YAML_FILE).exists() else None,
            pain_score=pain_progression[i],
            injury_notes=notes_progression[i],
        )

        # Only trigger LLM agents on the latest session (for demo speed)
        if i < len(KINEMATIC_SESSIONS) - 1:
            # Process silently for older sessions
            twin_agent.risk_agent  = None
            twin_agent.rehab_agent = None

        if i == len(KINEMATIC_SESSIONS) - 1 and has_api_key:
            # Re-attach agents for the final (latest) session
            twin_agent.risk_agent  = risk_agent
            twin_agent.rehab_agent = rehab_agent

        twin_agent.process_session(session)

    # ------------------------------------------------------------------
    # 6. Video-based ACL knee-flexion analysis (if video paths provided)
    # ------------------------------------------------------------------
    video_sessions = {sid: p for sid, p in VIDEO_INPUTS.items() if p}
    if video_sessions:
        separator("Step 2b: Video ACL Knee-Flexion Analysis (MediaPipe)")

        if not _MEDIAPIPE_AVAILABLE:
            print("  [SKIP] mediapipe / opencv not installed.")
            print("         Run: pip install mediapipe opencv-python")
        else:
            state = twin_agent.get_state(ATHLETE_ID)
            for session_id, video_path in video_sessions.items():
                if not Path(video_path).exists():
                    print(f"  [SKIP] {session_id}: video not found at {video_path}")
                    continue

                separator(f"Video: {session_id}")
                print(f"  File: {video_path}")
                try:
                    knee_result, assessment = risk_agent.assess_from_video(
                        video_path=video_path,
                        state=state,
                        session_id=session_id,
                    )
                    print()
                    print(knee_result.pretty())
                    print()
                    print(assessment.pretty())
                except Exception as e:
                    print(f"  [ERROR] {e}")
    else:
        print("\n[Video Analysis] No video paths configured in VIDEO_INPUTS — skipping.")
        print("  To enable: set paths in VIDEO_INPUTS at the top of main.py")

    # ------------------------------------------------------------------
    # 7. Offline demo (if no API key)
    # ------------------------------------------------------------------
    if not has_api_key:
        demo_offline_mode(twin_agent)
        separator()
        print("\nTo run with full Claude AI reasoning, set ANTHROPIC_API_KEY and re-run.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # ------------------------------------------------------------------
    # 7. Decision Agent – NLP query interface
    # ------------------------------------------------------------------
    separator("Step 3: Decision Agent – Natural Language Interface")  # Step numbering: 2b is video

    decision_agent = DecisionAgent(
        twin_agent=twin_agent,
        risk_agent=risk_agent,
        rehab_agent=rehab_agent,
        model="claude-sonnet-4-6",
    )

    # Scenario 1: Coach query (paper Figure 1 – left side)
    separator("Coach Query")
    q1 = f"How is {ATHLETE_NAME} doing today? Should she join today's training session?"
    print(f"Coach: {q1}")
    print()
    r1 = decision_agent.query(q1, athlete_id=ATHLETE_ID, role="coach")
    print(f"AI Response:\n{r1}")

    # Scenario 2: Medical staff query (paper Figure 1 – right side)
    separator("Medical Staff Query")
    q2 = f"Show me {ATHLETE_NAME}'s rehab progression timeline. Is she ready to start cutting drills?"
    print(f"Medical Staff: {q2}")
    print()
    r2 = decision_agent.query(q2, athlete_id=ATHLETE_ID, role="medical")
    print(f"AI Response:\n{r2}")

    # Scenario 3: Counterfactual / what-if (paper §2.4 causal reasoning)
    separator("What-If Analysis (Causal Reasoning)")
    q3 = f"What would happen if we reduced {ATHLETE_NAME}'s workload by 20% for the next 2 weeks?"
    print(f"Trainer: {q3}")
    print()
    r3 = decision_agent.query(q3, athlete_id=ATHLETE_ID, role="trainer")
    print(f"AI Response:\n{r3}")

    # ------------------------------------------------------------------
    # 8. Human feedback simulation (HITL loop)
    # ------------------------------------------------------------------
    separator("Step 4: Human-in-the-Loop Feedback")
    state = twin_agent.get_state(ATHLETE_ID)
    if state and state.session_ids:
        latest_sid = state.session_ids[-1]
        print(f"Medical staff reviews and corrects risk level for session {latest_sid}:")
        risk_agent.submit_feedback(
            athlete_id=ATHLETE_ID,
            session_id=latest_sid,
            corrected_risk_level="Moderate",
            notes="Patient reports pain reduced significantly; asymmetry trending down.",
        )
        rehab_agent.submit_feedback(
            athlete_id=ATHLETE_ID,
            session_id=latest_sid,
            corrected_stage="Late",
            notes="Cleared for lateral hops after in-person assessment.",
        )

    separator("Demo Complete")
    print(f"\nDigital twin versions: {twin_store.list_versions(ATHLETE_ID)}")
    print(f"Total sessions logged: {len(session_store.load_all(ATHLETE_ID))}")
    print(f"Memory stored in     : memory/twins/{ATHLETE_ID}/ and memory/sessions/")


if __name__ == "__main__":
    main()
