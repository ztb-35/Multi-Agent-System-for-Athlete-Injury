"""
Training session and multimodal sensor data models.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class WearableData:
    """Data captured from wearable sensors during a session."""
    avg_heart_rate: float              # bpm
    max_heart_rate: float              # bpm
    hrv: float                         # RMSSD ms (post-session)
    training_load: float               # sRPE × duration (AU)
    distance_km: float
    sprint_count: int
    acceleration_count: int
    sleep_hours_prev_night: float
    sleep_quality_score: float         # 0–1
    body_temp_celsius: Optional[float] = None
    skin_conductance: Optional[float] = None  # proxy for stress


@dataclass
class MotionCaptureData:
    """Biomechanical data from motion capture or video analysis."""
    system: str                        # e.g. "Vicon", "IMU", "video-AI"
    joint_angles: dict[str, float]     # e.g. {"knee_flexion_peak": 72.3}
    ground_reaction_forces: dict[str, float]   # e.g. {"vertical_peak_N": 1450}
    symmetry_index: float              # 0–1; 1 = perfect bilateral symmetry
    anomalies: list[str] = field(default_factory=list)
    raw_file_path: Optional[str] = None


@dataclass
class ImagingReport:
    """Structured summary of an imaging study (MRI, X-ray, ultrasound)."""
    modality: str                      # "MRI" | "X-ray" | "Ultrasound"
    body_part: str
    date: str                          # ISO-8601
    radiologist_summary: str
    findings: list[str] = field(default_factory=list)
    severity_grade: Optional[str] = None   # "Grade I" | "Grade II" | "Grade III"
    raw_report_path: Optional[str] = None


@dataclass
class ClinicalNote:
    """Physician or physiotherapist clinical note."""
    author: str
    role: str                          # "physician" | "physio" | "athletic_trainer"
    date: str
    subjective: str                    # athlete-reported symptoms
    objective: str                     # measurable findings
    assessment: str
    plan: str


@dataclass
class TrainingSession:
    session_id: str
    athlete_id: str
    timestamp: datetime
    session_type: str                  # "training" | "match" | "rehab" | "test"
    duration_minutes: int
    wearable: Optional[WearableData] = None
    motion_capture: Optional[MotionCaptureData] = None
    imaging: Optional[ImagingReport] = None
    clinical_note: Optional[ClinicalNote] = None
    coach_rpe: Optional[float] = None  # coach-perceived RPE 1–10

    def to_dict(self) -> dict:
        d: dict = {
            "session_id": self.session_id,
            "athlete_id": self.athlete_id,
            "timestamp": self.timestamp.isoformat(),
            "session_type": self.session_type,
            "duration_minutes": self.duration_minutes,
            "coach_rpe": self.coach_rpe,
        }
        if self.wearable:
            d["wearable"] = self.wearable.__dict__
        if self.motion_capture:
            d["motion_capture"] = self.motion_capture.__dict__
        if self.imaging:
            d["imaging"] = self.imaging.__dict__
        if self.clinical_note:
            d["clinical_note"] = self.clinical_note.__dict__
        return d
