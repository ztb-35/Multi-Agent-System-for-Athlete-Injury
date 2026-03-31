"""
Risk Agent – Injury Risk & Fatigue Assessment (paper §2.3).

Responsibilities:
  - Receive structured AthleteState from TwinAgent
  - Identify deviations from personalized baselines (not population thresholds)
  - Ensemble-style reasoning: rule-based anomaly detection + LLM analysis
  - Produce interpretable RiskAssessment with evidence references
  - Support RLHF-style feedback loop (store human corrections)
"""

from __future__ import annotations
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import numpy as np

from models.athlete_state import AthleteState, RiskAssessment

# Optional dependencies — fail gracefully if not installed
try:
    import cv2
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# ACL knee-flexion risk thresholds from biomechanics literature
# Hewett TE et al. (2005) AJSM; Myer GD et al. (2011) JOSPT
# Thresholds apply to minimum knee flexion at initial contact during landing/cutting.
# Lower flexion = more upright knee = higher ACL strain.
# ---------------------------------------------------------------------------

ACL_FLEXION_THRESHOLDS = {
    # (min_flexion_deg, max_flexion_deg): (risk_level, description)
    (0,   20): ("Critical", "Near-full extension at contact — peak ACL load (Hewett 2005)"),
    (20,  30): ("High",     "Low knee flexion — significantly elevated ACL stress (Myer 2011)"),
    (30,  45): ("Moderate", "Sub-optimal flexion — some ACL risk, technique correction advised"),
    (45, 180): ("Low",      "Adequate knee flexion — within safe landing mechanics range"),
}


@dataclass
class CutInWindow:
    """Knee angle data for one detected cut-in motion window."""
    window_idx: int            # 1-based event number
    start_frame: int
    peak_frame: int            # frame of maximum lateral deceleration
    end_frame: int
    time_sec: float            # timestamp of peak frame in video (seconds)

    knee_r_angles: list[float] = field(default_factory=list)
    knee_l_angles: list[float] = field(default_factory=list)
    knee_r_min: float = 0.0
    knee_l_min: float = 0.0
    acl_risk_level: str = "Unknown"
    acl_risk_description: str = ""

    def pretty(self) -> str:
        return (
            f"  Cut-in #{self.window_idx}  @{self.time_sec:.2f}s  "
            f"frames [{self.start_frame}–{self.end_frame}]  "
            f"R_min={self.knee_r_min:.1f}°  L_min={self.knee_l_min:.1f}°  "
            f"→ {self.acl_risk_level}"
        )


@dataclass
class KneeAngleResult:
    """Per-video knee flexion analysis output from VideoKneePoseAnalyzer."""
    video_path: str
    n_frames_analyzed: int
    n_cut_ins_detected: int = 0

    # Per-cut-in window breakdown
    cut_in_windows: list[CutInWindow] = field(default_factory=list)

    # Flattened angles across all cut-in windows (degrees; 0° = fully extended)
    knee_r_angles: list[float] = field(default_factory=list)
    knee_l_angles: list[float] = field(default_factory=list)

    # Summary stats — right knee (across all cut-in windows)
    knee_r_mean: float = 0.0
    knee_r_min: float = 0.0
    knee_r_max: float = 0.0

    # Summary stats — left knee
    knee_l_mean: float = 0.0
    knee_l_min: float = 0.0
    knee_l_max: float = 0.0

    # Overall ACL risk from worst-case cut-in window
    acl_risk_level: str = "Unknown"
    acl_risk_description: str = ""
    risk_flags: list[str] = field(default_factory=list)

    def pretty(self) -> str:
        windows_str = "\n".join(w.pretty() for w in self.cut_in_windows) or "  None"
        return (
            f"Frames analyzed : {self.n_frames_analyzed}\n"
            f"Cut-ins detected: {self.n_cut_ins_detected}\n"
            f"Cut-in windows  :\n{windows_str}\n"
            f"Right knee flex : min={self.knee_r_min:.1f}°  mean={self.knee_r_mean:.1f}°  max={self.knee_r_max:.1f}°\n"
            f"Left  knee flex : min={self.knee_l_min:.1f}°  mean={self.knee_l_mean:.1f}°  max={self.knee_l_max:.1f}°\n"
            f"ACL Risk Level  : {self.acl_risk_level}\n"
            f"Description     : {self.acl_risk_description}\n"
            f"Flags           : {'; '.join(self.risk_flags) or 'None'}"
        )


# ---------------------------------------------------------------------------
# Risk thresholds for rule-based anomaly signal (ensemble component 1)
# ---------------------------------------------------------------------------

RISK_RULES = {
    "knee_asymmetry_index":   {"warn": 10.0,  "high": 20.0,  "label": "Knee asymmetry"},
    "hip_asymmetry_index":    {"warn": 15.0,  "high": 25.0,  "label": "Hip asymmetry"},
    "hip_adduction_r_mean":   {"warn": 10.0,  "high": 18.0,  "label": "Hip adduction R (valgus)"},
    "hip_adduction_l_mean":   {"warn": 10.0,  "high": 18.0,  "label": "Hip adduction L (valgus)"},
    "session_duration_s":     {"pct_warn": 20, "pct_high": 35, "label": "Workload spike"},
}

# How many trend points to include in the LLM context
_TREND_HISTORY = 5


def _compute_knee_flexion(
    hip: tuple[float, float, float],
    knee: tuple[float, float, float],
    ankle: tuple[float, float, float],
) -> float:
    """
    Compute the knee flexion angle (degrees) from 3-D joint positions.
    Returns the angle at the knee vertex, between vectors knee→hip and knee→ankle.
    0° = fully extended, 90° = right angle, >90° = deep flexion.
    """
    v_hip   = np.array(hip)   - np.array(knee)
    v_ankle = np.array(ankle) - np.array(knee)
    cos_a   = np.dot(v_hip, v_ankle) / (np.linalg.norm(v_hip) * np.linalg.norm(v_ankle) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def _acl_risk_from_flexion(min_flexion_deg: float) -> tuple[str, str]:
    """Look up ACL risk level and description from the minimum knee flexion angle."""
    for (lo, hi), (level, desc) in ACL_FLEXION_THRESHOLDS.items():
        if lo <= min_flexion_deg < hi:
            return level, desc
    return "Unknown", "Angle outside expected range"


class VideoKneePoseAnalyzer:
    """
    Extract knee flexion angles from a video clip using MediaPipe Pose.

    MediaPipe landmark indices used:
      LEFT_HIP=23  RIGHT_HIP=24  LEFT_KNEE=25  RIGHT_KNEE=26
      LEFT_ANKLE=27  RIGHT_ANKLE=28

    Usage:
        analyzer = VideoKneePoseAnalyzer()
        result = analyzer.analyze("clip.mp4")
        print(result.pretty())
    """

    # MediaPipe landmark indices
    _LM = {
        "r_hip": 24, "l_hip": 23,
        "r_knee": 26, "l_knee": 25,
        "r_ankle": 28, "l_ankle": 27,
    }

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe and opencv-python are required: pip install mediapipe opencv-python")
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def _detect_cut_in_windows(
        self,
        knee_angles: list[float],
        fps: float,
        window_seconds: float,
    ) -> list[tuple[int, int, int]]:
        """
        Detect cut-in peak frames from per-frame minimum knee angle trajectory.

        Cut-in landing = deepest knee flexion (smallest angle value).
        This is camera-angle-independent and directly biomechanically meaningful.

        Returns list of (start_frame, peak_frame, end_frame).
        """
        n = len(knee_angles)
        window_frames = max(1, int(window_seconds * fps))

        angles = np.array(knee_angles, dtype=float)
        # Forward-fill NaN gaps
        for i in range(1, n):
            if np.isnan(angles[i]):
                angles[i] = angles[i - 1]
        if np.isnan(angles[0]):
            angles[0] = 180.0

        # Smooth to suppress per-frame jitter
        kernel_size = min(9, (n // 4) * 2 + 1)
        smoothed = np.convolve(angles, np.ones(kernel_size) / kernel_size, mode='same')

        # Find local minima (most flexed frames)
        local_mins: list[tuple[int, float]] = []
        for t in range(1, n - 1):
            if smoothed[t] < smoothed[t - 1] and smoothed[t] < smoothed[t + 1]:
                local_mins.append((t, float(smoothed[t])))

        if not local_mins:
            # No local minima — use global minimum
            peak_frame = int(np.argmin(smoothed))
            local_mins = [(peak_frame, float(smoothed[peak_frame]))]

        # Keep only minima within 15° of the global minimum (most significant dips)
        global_min_angle = min(a for _, a in local_mins)
        significant = [(f, a) for f, a in local_mins if a <= global_min_angle + 15.0]

        # Sort by angle ascending (most flexed first) then merge well-separated peaks
        significant.sort(key=lambda x: x[1])
        merged: list[int] = []
        for frame, _ in significant:
            if not any(abs(frame - m) < window_frames for m in merged):
                merged.append(frame)
        merged.sort()

        return [
            (max(0, p - window_frames), p, min(n - 1, p + window_frames))
            for p in merged
        ]

    def analyze(
        self,
        video_path: str,
        window_seconds: float = 0.5,
    ) -> KneeAngleResult:
        """
        Two-pass analysis focused on cut-in motion windows.

        Pass 1: Extract all landmarks; build hip x-trajectory.
        Detect: Find cut-in events from lateral velocity direction reversals.
        Pass 2: Compute knee flexion angles only within cut-in windows.

        Parameters
        ----------
        window_seconds   : half-width of sliding window around each cut-in peak (default 0.5s)
        min_lateral_speed: minimum hip lateral speed to count as a real cut-in (normalized units/frame)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # ── Pass 1: extract all landmarks and compute per-frame knee angles ──
        all_landmarks: list = []  # pose_world_landmarks per frame (or None)
        knee_angles_all: list[float] = []  # min(R, L) knee angle per frame

        with mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_world_landmarks:
                    lm = result.pose_world_landmarks.landmark
                    all_landmarks.append(lm)
                    r_angle = _compute_knee_flexion(
                        (lm[self._LM["r_hip"]].x,   lm[self._LM["r_hip"]].y,   lm[self._LM["r_hip"]].z),
                        (lm[self._LM["r_knee"]].x,  lm[self._LM["r_knee"]].y,  lm[self._LM["r_knee"]].z),
                        (lm[self._LM["r_ankle"]].x, lm[self._LM["r_ankle"]].y, lm[self._LM["r_ankle"]].z),
                    )
                    l_angle = _compute_knee_flexion(
                        (lm[self._LM["l_hip"]].x,   lm[self._LM["l_hip"]].y,   lm[self._LM["l_hip"]].z),
                        (lm[self._LM["l_knee"]].x,  lm[self._LM["l_knee"]].y,  lm[self._LM["l_knee"]].z),
                        (lm[self._LM["l_ankle"]].x, lm[self._LM["l_ankle"]].y, lm[self._LM["l_ankle"]].z),
                    )
                    knee_angles_all.append(min(r_angle, l_angle))
                else:
                    all_landmarks.append(None)
                    knee_angles_all.append(float("nan"))

        cap.release()

        n_detected = sum(1 for lm in all_landmarks if lm is not None)
        if n_detected == 0:
            raise RuntimeError("No pose landmarks detected — check video quality or athlete visibility.")

        # ── Detect cut-in windows from knee angle trajectory ─────────────────
        windows = self._detect_cut_in_windows(
            knee_angles=knee_angles_all,
            fps=fps,
            window_seconds=window_seconds,
        )

        if not windows:
            raise RuntimeError(
                "No cut-in motion detected. "
                "Try lowering min_lateral_speed or verify the video contains lateral cutting."
            )

        print(f"[VideoKneePoseAnalyzer] {len(windows)} cut-in window(s) detected "
              f"from {n_detected} pose frames ({fps:.1f} fps)")

        # ── Pass 2: compute knee angles within each cut-in window ────────────
        cut_in_windows: list[CutInWindow] = []

        for idx, (start, peak, end) in enumerate(windows):
            knee_r: list[float] = []
            knee_l: list[float] = []

            for f in range(start, end + 1):
                lm = all_landmarks[f]
                if lm is None:
                    continue

                r_hip   = (lm[self._LM["r_hip"]].x,   lm[self._LM["r_hip"]].y,   lm[self._LM["r_hip"]].z)
                r_knee  = (lm[self._LM["r_knee"]].x,  lm[self._LM["r_knee"]].y,  lm[self._LM["r_knee"]].z)
                r_ankle = (lm[self._LM["r_ankle"]].x, lm[self._LM["r_ankle"]].y, lm[self._LM["r_ankle"]].z)
                knee_r.append(_compute_knee_flexion(r_hip, r_knee, r_ankle))

                l_hip   = (lm[self._LM["l_hip"]].x,   lm[self._LM["l_hip"]].y,   lm[self._LM["l_hip"]].z)
                l_knee  = (lm[self._LM["l_knee"]].x,  lm[self._LM["l_knee"]].y,  lm[self._LM["l_knee"]].z)
                l_ankle = (lm[self._LM["l_ankle"]].x, lm[self._LM["l_ankle"]].y, lm[self._LM["l_ankle"]].z)
                knee_l.append(_compute_knee_flexion(l_hip, l_knee, l_ankle))

            if not knee_r:
                continue

            min_flexion = min(min(knee_r), min(knee_l))
            level, desc = _acl_risk_from_flexion(min_flexion)

            cut_in_windows.append(CutInWindow(
                window_idx=idx + 1,
                start_frame=start,
                peak_frame=peak,
                end_frame=end,
                time_sec=peak / fps,
                knee_r_angles=knee_r,
                knee_l_angles=knee_l,
                knee_r_min=float(np.min(knee_r)),
                knee_l_min=float(np.min(knee_l)),
                acl_risk_level=level,
                acl_risk_description=desc,
            ))

        if not cut_in_windows:
            raise RuntimeError("No valid knee angles computed in detected cut-in windows.")

        # ── Aggregate across all windows ─────────────────────────────────────
        all_r = [a for w in cut_in_windows for a in w.knee_r_angles]
        all_l = [a for w in cut_in_windows for a in w.knee_l_angles]
        global_min = min(min(w.knee_r_min for w in cut_in_windows),
                         min(w.knee_l_min for w in cut_in_windows))
        acl_level, acl_desc = _acl_risk_from_flexion(global_min)

        flags: list[str] = []
        worst_r = min(w.knee_r_min for w in cut_in_windows)
        worst_l = min(w.knee_l_min for w in cut_in_windows)
        if worst_r < worst_l:
            flags.append(f"Right knee most extended during cut-in ({worst_r:.1f}°)")
        else:
            flags.append(f"Left knee most extended during cut-in ({worst_l:.1f}°)")
        if abs(float(np.mean(all_r)) - float(np.mean(all_l))) > 10:
            flags.append(f"Bilateral asymmetry {abs(np.mean(all_r) - np.mean(all_l)):.1f}° during cut-ins")
        if global_min < 30:
            flags.append("Minimum cut-in flexion < 30° — high ACL risk (Hewett 2005)")

        return KneeAngleResult(
            video_path=video_path,
            n_frames_analyzed=n_detected,
            n_cut_ins_detected=len(cut_in_windows),
            cut_in_windows=cut_in_windows,
            knee_r_angles=all_r,
            knee_l_angles=all_l,
            knee_r_mean=float(np.mean(all_r)),
            knee_r_min=float(np.min(all_r)),
            knee_r_max=float(np.max(all_r)),
            knee_l_mean=float(np.mean(all_l)),
            knee_l_min=float(np.min(all_l)),
            knee_l_max=float(np.max(all_l)),
            acl_risk_level=acl_level,
            acl_risk_description=acl_desc,
            risk_flags=flags,
        )


class RiskAgent:
    """
    Ensemble injury risk assessor.

    Component 1: Rule-based anomaly flags (fast, deterministic)
    Component 2: Claude LLM reasoning over athlete state + flags

    Usage:
        agent = RiskAgent()
        assessment = agent.assess(athlete_state, session_id="S2026_02_03")
        print(assessment.pretty())
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        feedback_dir: str = "memory/risk_feedback",
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

    def assess(self, state: AthleteState, session_id: str) -> RiskAssessment:
        """
        Run ensemble risk assessment over the given AthleteState.
        Returns a RiskAssessment with level, confidence, drivers, and evidence.
        """
        # Component 1: rule-based anomaly flags
        flags = self._rule_based_flags(state)

        # Component 2: LLM reasoning
        assessment = self._llm_assess(state, session_id, flags)

        return assessment

    def assess_from_video(
        self,
        video_path: str,
        state: AthleteState,
        session_id: str,
    ) -> tuple[KneeAngleResult, RiskAssessment]:
        """
        Full pipeline: video → MediaPipe pose → knee angles → ACL risk assessment.

        1. Extract per-frame knee flexion angles with VideoKneePoseAnalyzer
        2. Inject angle stats into athlete state deviations
        3. Run ensemble assessment (rule-based + LLM)

        Returns (KneeAngleResult, RiskAssessment).
        """
        analyzer = VideoKneePoseAnalyzer()
        knee_result = analyzer.analyze(video_path)

        print(f"[RiskAgent] Video analysis complete — {knee_result.n_frames_analyzed} frames")
        print(knee_result.pretty())

        # Inject knee angle stats into state so existing rule-based + LLM pipeline sees them
        state.deviations["knee_flexion_r_min"] = {
            "current": knee_result.knee_r_min,
            "baseline": state.baseline.get("knee_flexion_r_min", 45.0),
            "delta": knee_result.knee_r_min - state.baseline.get("knee_flexion_r_min", 45.0),
            "pct_change": 0.0,
        }
        state.deviations["knee_flexion_l_min"] = {
            "current": knee_result.knee_l_min,
            "baseline": state.baseline.get("knee_flexion_l_min", 45.0),
            "delta": knee_result.knee_l_min - state.baseline.get("knee_flexion_l_min", 45.0),
            "pct_change": 0.0,
        }

        # Run full ensemble assessment with video context injected
        flags = self._rule_based_flags(state)

        # Append video-derived ACL flags
        for flag_text in knee_result.risk_flags:
            flags.append({
                "metric": "Video knee flexion (MediaPipe)",
                "value": knee_result.knee_r_min,
                "baseline": 45.0,
                "delta": knee_result.knee_r_min - 45.0,
                "pct_change": 0.0,
                "severity": "high" if knee_result.acl_risk_level in ("Critical", "High") else "moderate",
                "detail": flag_text,
            })

        assessment = self._llm_assess(state, session_id, flags)
        return knee_result, assessment

    def submit_feedback(
        self,
        athlete_id: str,
        session_id: str,
        corrected_risk_level: str,
        notes: str = "",
    ) -> None:
        """
        Human-in-the-loop feedback mechanism (paper §2.3 RLHF hook).
        Stores corrections for future fine-tuning or prompt refinement.
        """
        record = {
            "athlete_id": athlete_id,
            "session_id": session_id,
            "corrected_risk_level": corrected_risk_level,
            "notes": notes,
            "timestamp": time.time(),
        }
        fb_path = self.feedback_dir / f"{athlete_id}.ndjson"
        with open(fb_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[RiskAgent] Feedback recorded for {athlete_id}/{session_id}")

    # ------------------------------------------------------------------
    # Component 1: Rule-based anomaly detection
    # ------------------------------------------------------------------

    def _rule_based_flags(self, state: AthleteState) -> list[dict]:
        """
        Scan deviations for threshold violations.
        Returns a list of flag dicts with keys: metric, value, baseline, delta, severity.
        """
        flags = []
        devs  = state.deviations
        snap  = state.latest_snapshot or {}

        for metric, thresholds in RISK_RULES.items():
            if metric not in devs and metric not in snap:
                continue

            current  = devs.get(metric, {}).get("current",  snap.get(metric, 0.0))
            baseline = devs.get(metric, {}).get("baseline", 0.0)
            delta    = devs.get(metric, {}).get("delta",    0.0)
            pct      = devs.get(metric, {}).get("pct_change", 0.0)

            severity = None

            # Absolute threshold check (for asymmetry / adduction)
            if "high" in thresholds and abs(current) >= thresholds["high"]:
                severity = "high"
            elif "warn" in thresholds and abs(current) >= thresholds["warn"]:
                severity = "moderate"

            # Percentage-change check (for workload)
            if "pct_high" in thresholds and abs(pct) >= thresholds["pct_high"]:
                severity = "high"
            elif "pct_warn" in thresholds and abs(pct) >= thresholds["pct_warn"]:
                severity = severity or "moderate"

            if severity:
                flags.append({
                    "metric":   thresholds["label"],
                    "value":    round(current, 2),
                    "baseline": round(baseline, 2),
                    "delta":    round(delta, 2),
                    "pct_change": round(pct, 1),
                    "severity": severity,
                })

        return flags

    # ------------------------------------------------------------------
    # Component 2: Claude LLM reasoning
    # ------------------------------------------------------------------

    def _build_prompt(self, state: AthleteState, session_id: str, flags: list[dict]) -> str:
        """Construct the system + user prompt for the risk assessment."""

        # Subset trends to last N sessions for brevity
        trend_snippet = {}
        for k, v in state.trends.items():
            trend_snippet[k] = v[-_TREND_HISTORY:]

        flags_text = json.dumps(flags, indent=2) if flags else "No threshold violations detected."
        deviations_text = json.dumps(
            {k: v for k, v in state.deviations.items()
             if abs(v.get("pct_change", 0)) > 5},  # only noteworthy
            indent=2
        ) if state.deviations else "{}"

        return f"""You are an expert sports medicine AI performing injury risk assessment.
You receive a structured athlete digital twin state and anomaly flags from a rule-based detector.
Your job is to synthesize these signals into an interpretable risk assessment.

## Athlete Profile
- ID: {state.athlete_id}
- Name: {state.name or 'Unknown'}
- Age: {state.age}, Gender: {state.gender}, Height: {state.height_m}m, Mass: {state.mass_kg}kg
- Sport: {state.sport or 'Not specified'}
- Injury History: {', '.join(state.injury_history) or 'None'}
- Active Injury: {state.active_injury or 'None'}
- Sessions completed: {len(state.session_ids)}
- Recent pain scores (VAS 0-10): {state.pain_scores[-5:] if state.pain_scores else 'N/A'}

## Current Session
- Session ID: {session_id}

## Biomechanical Deviations from Personalized Baseline (>5% change shown)
{deviations_text}

## Temporal Trends (last {_TREND_HISTORY} sessions)
{json.dumps(trend_snippet, indent=2)}

## Rule-Based Anomaly Flags
{flags_text}

## Task
Perform injury risk assessment. Respond with ONLY a JSON object in this exact schema:
{{
  "risk_level": "Low" | "Moderate" | "High" | "Critical",
  "confidence": <float 0.0-1.0>,
  "top_risk_drivers": [<string>, ...],
  "evidence_refs": [<session_id>, ...],
  "reasoning": "<concise 2-3 sentence explanation>"
}}

Focus on personalized deviations and trends rather than population norms.
Be conservative: when uncertain, prefer a higher risk level for safety."""

    def _llm_assess(
        self, state: AthleteState, session_id: str, flags: list[dict]
    ) -> RiskAssessment:
        """Call Claude to synthesize a risk assessment."""
        prompt = self._build_prompt(state, session_id, flags)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)

            return RiskAssessment(
                athlete_id=state.athlete_id,
                session_id=session_id,
                risk_level=data.get("risk_level", "Unknown"),
                confidence=float(data.get("confidence", 0.5)),
                top_risk_drivers=data.get("top_risk_drivers", []),
                evidence_refs=data.get("evidence_refs", [session_id]),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            print(f"[RiskAgent] LLM call failed: {e}. Using rule-based fallback.")
            return self._fallback_assessment(state, session_id, flags)

    def _fallback_assessment(
        self, state: AthleteState, session_id: str, flags: list[dict]
    ) -> RiskAssessment:
        """Rule-based fallback when LLM is unavailable."""
        high_flags     = [f for f in flags if f["severity"] == "high"]
        moderate_flags = [f for f in flags if f["severity"] == "moderate"]

        if high_flags:
            level, conf = "High", 0.70
        elif moderate_flags:
            level, conf = "Moderate", 0.65
        else:
            level, conf = "Low", 0.75

        drivers = [f"{f['metric']} ({f['value']:.1f} vs baseline {f['baseline']:.1f})" for f in flags]

        return RiskAssessment(
            athlete_id=state.athlete_id,
            session_id=session_id,
            risk_level=level,
            confidence=conf,
            top_risk_drivers=drivers or ["No significant deviations detected"],
            evidence_refs=[session_id],
            reasoning=f"Rule-based fallback: {len(high_flags)} high + {len(moderate_flags)} moderate flags.",
        )
