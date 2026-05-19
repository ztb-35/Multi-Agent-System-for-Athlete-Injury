# Multi-Agent Digital Twin System
### ACL Injury Risk Assessment & Rehabilitation Planning

A multi-agent AI system that tracks athlete biomechanics across rehab sessions, assesses injury risk, and generates personalized rehabilitation plans. Built with Claude (Anthropic) as the LLM backbone.

**Contact:** Taibiao Zhao · tzhao3@lsu.edu

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Data](#4-data)
5. [Running the System](#5-running-the-system)
6. [Running the Dashboard](#6-running-the-dashboard)
7. [Deploying Online](#7-deploying-online)
8. [How Each Agent Works](#8-how-each-agent-works)
9. [Adding a New Athlete](#9-adding-a-new-athlete)
10. [Common Issues](#10-common-issues)
11. [What's Next](#11-whats-next)

---

## 1. System Overview

```
Raw Data (.mot / .trc / .yaml)
           ↓
      [Twin Agent]          ← versioned state manager, single source of truth
      /           \
 [Risk Agent]  [Rehab Agent]   ← specialist LLM agents (Claude Haiku)
      \           /
   [Decision Agent]            ← NLP coordinator (Claude Sonnet), tool_use routing
           ↕
   Coaches / Medical Staff
           ↕
     Streamlit Dashboard       ← web UI (app.py)
```

**Four agents:**

| Agent | Model | Role |
|-------|-------|------|
| TwinAgent | — | Ingests session data, maintains versioned athlete state |
| RiskAgent | claude-haiku-4-5 | Ensemble risk assessment (rule-based + LLM) |
| RehabAgent | claude-haiku-4-5 | ACL rehab planning, counterfactual reasoning |
| DecisionAgent | claude-sonnet-4-6 | NLP interface, routes queries to specialist agents via tool_use |

---

## 2. Project Structure

```
PoseAngle/
├── app.py                     ← Streamlit dashboard (main entry point for UI)
├── main.py                    ← CLI demo entry point (runs full pipeline)
├── requirements.txt
│
├── agents/
│   ├── twin_agent.py          ← versioned digital twin, NewSessionEvent handler
│   ├── risk_agent.py          ← injury risk assessment + MediaPipe video analysis
│   ├── rehab_agent.py         ← ACL rehab planner + counterfactual reasoning
│   └── decision_agent.py      ← Claude tool_use NLP coordinator
│
├── models/
│   └── athlete_state.py       ← data classes: AthleteState, RiskAssessment, RehabPlan, etc.
│
├── memory/
│   ├── session_store.py       ← NDJSON session log
│   ├── twin_store.py          ← versioned JSON twin snapshots
│   ├── sessions/              ← A01.ndjson (raw session log)
│   └── twins/                 ← A01/v0001.json ... latest.json
│
├── utils/
│   └── data_loader.py         ← .mot / .trc / .yaml parser → BiomechanicsSnapshot
│
└── data/
    ├── OpenSimData/Kinematics/ ← Abigail.mot, Abigail_1.mot ... (6 sessions)
    ├── MarkerData/             ← .trc marker files
    ├── Videos/Cam1/            ← .mp4 video files per session
    └── sessionMetadata.yaml
```

---

## 3. Environment Setup

**Python version:** 3.11 (tested). Use the `pytorch` conda environment or create a new one.

```bash
# Option A: conda
conda create -n digitaltwn python=3.11
conda activate digitaltwn

# Option B: existing pytorch env
conda activate pytorch
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `anthropic`, `streamlit`, `plotly`, `pandas`, `numpy`, `pyyaml`

**Optional (for video analysis only):**

```bash
pip install mediapipe opencv-python
```

**Set your Anthropic API key:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Get a key at [console.anthropic.com](https://console.anthropic.com). The system uses:
- `claude-haiku-4-5-20251001` for Risk and Rehab agents (cheap, fast)
- `claude-sonnet-4-6` for the Decision agent (smarter, used only when queried)

---

## 4. Data

All data is in `data/`. The current athlete is **Abigail Savoy (A01)**, 6 sessions of ACL reconstruction rehab.

| File type | Location | Contents |
|-----------|----------|----------|
| `.mot` | `data/OpenSimData/Kinematics/` | OpenSim joint angles, 34 columns, ~400-900 rows per session |
| `.trc` | `data/MarkerData/` | 3D marker positions |
| `.yaml` | `data/sessionMetadata.yaml` | Subject demographics |
| `.mp4` | `data/Videos/Cam1/InputMedia/` | Sync'd video per session |

The `.mot` files are the primary data source. Each row is one time frame; columns include `knee_angle_r`, `hip_flexion_l`, `pelvis_tilt`, etc. `data_loader.py` parses these into `BiomechanicsSnapshot` objects.

**Already processed data** is cached in `memory/twins/A01/`. If you just want to run the dashboard without re-processing, this is enough — you do NOT need to re-run `main.py`.

---

## 5. Running the System

### Full pipeline (CLI)

This processes all 6 sessions, builds the digital twin, runs risk + rehab agents on the last session, and demonstrates the Decision agent:

```bash
python main.py
```

Output: versioned twin snapshots saved to `memory/twins/A01/`, session logs to `memory/sessions/A01.ndjson`.

> **Note:** Running `main.py` multiple times appends data. The twin store handles versioning correctly, but `pain_scores` and `injury_notes_history` in the state will accumulate duplicates. This is a known issue — for demo purposes it doesn't affect the dashboard display (the dashboard takes only the last N values aligned to session count).

### Offline mode (no API key)

Running without `ANTHROPIC_API_KEY` skips LLM calls and shows rule-based outputs only.

---

## 6. Running the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Five tabs:**

| Tab | What it shows | Needs API key? |
|-----|--------------|----------------|
| Overview | KPI cards, 4 trend charts, session table | No |
| Risk Assessment | Run risk agent, see risk level + drivers | Yes |
| Rehab Plan | Run rehab agent, see stage + exercises | Yes |
| AI Assistant | Chat with DecisionAgent (Coach/Medical/Trainer roles) | Yes |
| What-If Analysis | Counterfactual reasoning (e.g. "reduce workload 30%") | Yes |

---

## 7. Deploying Online

The app is hosted on **Streamlit Community Cloud** (free). The GitHub repo is:
`https://github.com/ztb-35/Multi-Agent-System-for-Athlete-Injury`

### To redeploy after code changes:

```bash
git add .
git commit -m "your message"
git push origin main
```

Streamlit Cloud auto-detects the push and redeploys within ~2 minutes.

### To set the API key on Streamlit Cloud:

1. Go to [share.streamlit.io](https://share.streamlit.io) → your app → **Settings** → **Secrets**
2. Add:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

The app reads from `st.secrets` on Cloud and from the environment variable locally — both are handled automatically in `app.py`.

### To add access control (LSU email only):

This is not yet implemented. Options:
- **Google OAuth** (proper): Set up at Google Cloud Console, restrict to `@lsu.edu` domain
- **streamlit-authenticator** (simple): Pre-register usernames/passwords in a YAML config

---

## 8. How Each Agent Works

### TwinAgent (`agents/twin_agent.py`)

The central hub. Call `process_session(SessionData)` to:
1. Log raw session to NDJSON
2. Parse `.mot` file → `BiomechanicsSnapshot`
3. Update athlete state: rolling baseline, deviations, trends
4. Save versioned JSON snapshot
5. Trigger RiskAgent + RehabAgent

```python
twin_agent.register_athlete("A01", name="Abigail", age=22, ...)
session = SessionData(athlete_id="A01", session_id="S001", mot_file="data/...")
state = twin_agent.process_session(session)
```

### RiskAgent (`agents/risk_agent.py`)

Two-component ensemble:
1. **Rule-based flags**: checks knee asymmetry, hip adduction, workload spike against thresholds
2. **Claude LLM**: receives structured state + flags, returns `RiskAssessment` JSON

```python
assessment = risk_agent.assess(state, session_id)
# assessment.risk_level → "Low" / "Moderate" / "High" / "Critical"
```

Also has video analysis via MediaPipe (`assess_from_video()`), if `mediapipe` and `opencv-python` are installed.

### RehabAgent (`agents/rehab_agent.py`)

Embedded ACL protocol knowledge base (4 stages: Early / Mid / Late / Return-to-Play). Claude reads the athlete's biomechanics and determines current stage + exercises + restrictions.

```python
plan = rehab_agent.plan(state, session_id)
result = rehab_agent.counterfactual(state, "reduce workload by 30%")
```

To add new injury protocols: edit the `REHAB_PROTOCOLS` dict at the top of `rehab_agent.py`.

### DecisionAgent (`agents/decision_agent.py`)

Claude Sonnet with 4 tools:
- `get_athlete_state` → reads twin store
- `assess_injury_risk` → calls RiskAgent
- `get_rehab_plan` → calls RehabAgent
- `what_if_analysis` → calls RehabAgent.counterfactual()

```python
response = decision_agent.query("Can Abigail train today?", athlete_id="A01", role="coach")
```

---

## 9. Adding a New Athlete

1. **Add data files** to `data/OpenSimData/Kinematics/`, `data/MarkerData/`, `data/Videos/`

2. **Edit `main.py`** — update these constants at the top:
```python
ATHLETE_ID   = "A02"
ATHLETE_NAME = "New Athlete Name"
KINEMATIC_SESSIONS = [
    ("S2026_01_01", "OpenSimData/Kinematics/NewAthlete.mot"),
    ...
]
```

3. **Run the pipeline:**
```bash
python main.py
```

4. **Update `app.py`** — change `ATHLETE_ID = "A02"` near the top (or add a dropdown to select athlete).

---

## 10. Common Issues

**`ModuleNotFoundError: No module named 'anthropic'`**
→ Run `pip install -r requirements.txt`

**LLM calls fail with "credit balance too low"**
→ Top up at [console.anthropic.com](https://console.anthropic.com) → Plans & Billing

**Dashboard shows wrong/repeated data (pain scores appear 60+ times)**
→ `main.py` has been run multiple times. The dashboard handles this by taking only the last `N` values aligned to session count. To fully reset: delete `memory/twins/A01/` and `memory/sessions/A01.ndjson`, then re-run `main.py` once.

**`FileNotFoundError` for `.mot` files**
→ Check `DATA_DIR` in `main.py` points to the correct `data/` folder. Run from the project root directory.

**Streamlit shows blank page after deploy**
→ Check that `ANTHROPIC_API_KEY` is set in Streamlit Cloud Secrets (Settings → Secrets).

---

## 11. What's Next

Suggested improvements for the next person:

- **More athletes**: the system is built for multi-athlete, only one (A01) is wired up in `main.py`
- **Authentication**: add Google OAuth restricted to `@lsu.edu` domain
- **Longitudinal evaluation**: quantitative metrics comparing predicted vs. actual progression
- **Richer ACL protocol**: add specific load parameters (sets/reps/intensity) to `REHAB_PROTOCOLS`
- **Risk → Rehab linkage**: when RiskAgent outputs "High", automatically flag it in the RehabPlan
- **Real-time data**: replace `.mot` file parsing with a live OpenSim streaming connection
- **Multimodal risk agent**: pass annotated video frames directly to Claude vision API for visual ACL risk assessment (skeleton extraction code already exists in `risk_agent.py`)
