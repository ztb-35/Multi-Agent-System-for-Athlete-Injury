"""
Data loader for heterogeneous athlete data sources:
  - OpenSim kinematics (.mot)  → BiomechanicsSnapshot
  - Marker data (.trc)         → raw DataFrame
  - Session metadata (.yaml)   → dict
  - Pose landmarks (.json)     → raw list

Corresponds to the multimodal data ingestion layer (paper §2.2).
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional
import yaml

from models.athlete_state import BiomechanicsSnapshot


# ---------------------------------------------------------------------------
# MOT (OpenSim kinematics) loader
# ---------------------------------------------------------------------------

def load_mot(path: str | Path) -> list[dict]:
    """
    Parse an OpenSim .mot file into a list of row-dicts.
    Returns [] on failure.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows: list[dict] = []
    headers: list[str] = []
    in_data = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "endheader":
                in_data = True
                continue
            if not in_data:
                continue
            if not headers:
                headers = line.split("\t")
                continue
            parts = line.split()
            if len(parts) == len(headers):
                rows.append({h: float(v) for h, v in zip(headers, parts)})

    return rows


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

def _asymmetry_index(r: float, l: float) -> float:
    """Bilateral asymmetry index (%) : |R−L| / max(|R|, |L|) * 100"""
    denom = max(abs(r), abs(l))
    if denom == 0:
        return 0.0
    return abs(r - l) / denom * 100.0


def mot_to_snapshot(session_id: str, path: str | Path) -> BiomechanicsSnapshot:
    """
    Compute a BiomechanicsSnapshot from an OpenSim .mot file.
    Uses the 34-column coordinate format produced by the OpenSim pipeline.
    """
    rows = load_mot(path)
    snap = BiomechanicsSnapshot(session_id=session_id)

    if not rows:
        return snap

    def col(key: str) -> list[float]:
        return [r[key] for r in rows if key in r]

    # --- Knee angles ---
    kr = col("knee_angle_r")
    kl = col("knee_angle_l")
    snap.knee_angle_r_mean = _safe_mean(kr)
    snap.knee_angle_r_std  = _safe_std(kr)
    snap.knee_angle_r_min  = min(kr) if kr else 0.0
    snap.knee_angle_r_max  = max(kr) if kr else 0.0
    snap.knee_angle_l_mean = _safe_mean(kl)
    snap.knee_angle_l_std  = _safe_std(kl)
    snap.knee_angle_l_min  = min(kl) if kl else 0.0
    snap.knee_angle_l_max  = max(kl) if kl else 0.0

    # --- Hip flexion ---
    snap.hip_flexion_r_mean = _safe_mean(col("hip_flexion_r"))
    snap.hip_flexion_l_mean = _safe_mean(col("hip_flexion_l"))

    # --- Hip adduction (dynamic knee valgus proxy) ---
    snap.hip_adduction_r_mean = _safe_mean(col("hip_adduction_r"))
    snap.hip_adduction_l_mean = _safe_mean(col("hip_adduction_l"))

    # --- Ankle angles ---
    snap.ankle_angle_r_mean = _safe_mean(col("ankle_angle_r"))
    snap.ankle_angle_l_mean = _safe_mean(col("ankle_angle_l"))

    # --- Pelvis ---
    snap.pelvis_tilt_mean = _safe_mean(col("pelvis_tilt"))
    snap.pelvis_list_mean = _safe_mean(col("pelvis_list"))

    # --- Lumbar ---
    snap.lumbar_extension_mean = _safe_mean(col("lumbar_extension"))

    # --- Derived: asymmetry indices ---
    snap.knee_asymmetry_index = _asymmetry_index(
        snap.knee_angle_r_mean, snap.knee_angle_l_mean
    )
    snap.hip_asymmetry_index = _asymmetry_index(
        snap.hip_flexion_r_mean, snap.hip_flexion_l_mean
    )

    # --- Workload proxy ---
    snap.n_frames = len(rows)
    times = col("time")
    snap.session_duration_s = (times[-1] - times[0]) if len(times) >= 2 else 0.0

    return snap


# ---------------------------------------------------------------------------
# YAML session metadata loader
# ---------------------------------------------------------------------------

def load_session_yaml(path: str | Path) -> dict:
    """Parse a session metadata YAML file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Pose landmarks JSON loader
# ---------------------------------------------------------------------------

def load_pose_landmarks(path: str | Path) -> list[dict]:
    """Load MediaPipe pose landmarks JSON produced by human_motion.py."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TRC marker data loader (simple tab-separated)
# ---------------------------------------------------------------------------

def load_trc(path: str | Path) -> list[dict]:
    """
    Minimal TRC parser – returns rows as dicts.
    TRC files have a 6-line header before the data columns.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Row 3 (0-indexed 2) has the marker names; row 4 has X/Y/Z labels
    # Row 6+ is data
    if len(lines) < 7:
        return rows

    marker_line = lines[3].rstrip("\n").split("\t")
    # Skip first two columns (Frame, Time), then group by 3 (X,Y,Z)
    markers = [m for m in marker_line[2:] if m.strip()]

    for line in lines[6:]:
        parts = line.strip().split("\t")
        if not parts or not parts[0].strip():
            continue
        try:
            frame = int(float(parts[0]))
            t     = float(parts[1])
        except (ValueError, IndexError):
            continue
        data_vals = []
        for v in parts[2:]:
            try:
                data_vals.append(float(v))
            except ValueError:
                data_vals.append(0.0)
        row: dict = {"frame": frame, "time": t}
        for i, m in enumerate(markers):
            base = i * 3
            row[m + "_x"] = data_vals[base]     if base     < len(data_vals) else 0.0
            row[m + "_y"] = data_vals[base + 1] if base + 1 < len(data_vals) else 0.0
            row[m + "_z"] = data_vals[base + 2] if base + 2 < len(data_vals) else 0.0
        rows.append(row)

    return rows
