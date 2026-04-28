"""
ACL Digital Twin – Streamlit Dashboard
Multi-Agent System for Athlete Injury Risk & Rehabilitation (ICLR 2026 / MALGAI)

Run:
    streamlit run app.py

Deploy (free public link):
    Push repo to GitHub → streamlit.io/cloud → Connect repo → Deploy
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Inject API key from Streamlit Secrets (cloud deploy) if not already in env
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from memory.twin_store import TwinStore
from memory.session_store import SessionStore
from agents.twin_agent import TwinAgent
from agents.risk_agent import RiskAgent
from agents.rehab_agent import RehabAgent
from agents.decision_agent import DecisionAgent
from models.athlete_state import AthleteState

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ACL Digital Twin Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { font-size: 13px; color: #64748b; margin-bottom: 4px; }
  .metric-value { font-size: 26px; font-weight: 700; color: #0f172a; }
  .metric-sub   { font-size: 12px; color: #94a3b8; margin-top: 2px; }

  .risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.5px;
  }
  .tag {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 13px;
    margin: 3px 2px;
  }
  .restriction-tag {
    background: #fef2f2;
    color: #b91c1c;
  }
  .criteria-tag {
    background: #f0fdf4;
    color: #15803d;
  }
  .chat-user   { background:#eff6ff; border-radius:10px; padding:10px 14px; margin:6px 0; }
  .chat-ai     { background:#f8fafc; border-radius:10px; padding:10px 14px; margin:6px 0; border-left:3px solid #6366f1; }
  .tool-call   { background:#fefce8; border-radius:8px; padding:8px 12px; font-size:12px; color:#713f12; margin:4px 0; }
  .section-header { font-size:15px; font-weight:600; color:#374151; margin:16px 0 8px 0; border-bottom:1px solid #e5e7eb; padding-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────

ATHLETE_ID = "A01"
SESSION_IDS = [
    "S2026_01_01", "S2026_01_08", "S2026_01_15",
    "S2026_01_22", "S2026_01_29", "S2026_02_03",
]
SESSION_LABELS = ["Jan 1", "Jan 8", "Jan 15", "Jan 22", "Jan 29", "Feb 3"]

RISK_COLORS = {
    "Low":      "#22c55e",
    "Moderate": "#f59e0b",
    "High":     "#ef4444",
    "Critical": "#7c3aed",
    "Unknown":  "#94a3b8",
}

STAGE_ORDER = ["Early (0-6 weeks)", "Mid (6-14 weeks)", "Late (14-24 weeks)", "Return-to-Play (24+ weeks)"]

# ─── Caching ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_stores():
    return TwinStore("memory/twins"), SessionStore("memory/sessions")

@st.cache_resource
def _load_agents():
    twin_store, session_store = _load_stores()
    risk_agent   = RiskAgent(model="claude-haiku-4-5-20251001")
    rehab_agent  = RehabAgent(model="claude-haiku-4-5-20251001")
    twin_agent   = TwinAgent(
        session_store=session_store,
        twin_store=twin_store,
        risk_agent=None,
        rehab_agent=None,
    )
    decision_agent = DecisionAgent(
        twin_agent=twin_agent,
        risk_agent=risk_agent,
        rehab_agent=rehab_agent,
        model="claude-sonnet-4-6",
    )
    return twin_agent, risk_agent, rehab_agent, decision_agent

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _aligned(state: AthleteState, key: str) -> list:
    """Return the last N values from a trend list, aligned to session count."""
    n    = len(SESSION_IDS)
    vals = state.trends.get(key, [])
    return (vals[-n:] if len(vals) >= n else vals)


def _pain_scores(state: AthleteState) -> list:
    n = len(SESSION_IDS)
    s = state.pain_scores
    return s[-n:] if len(s) >= n else s


def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _sparkline(values: list, labels: list, color: str = "#6366f1", title: str = "") -> go.Figure:
    fill_color = _hex_to_rgba(color, 0.08) if color.startswith("#") else color
    fig = go.Figure(go.Scatter(
        x=labels[:len(values)], y=values,
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color),
        fill="tozeroy",
        fillcolor=fill_color,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=8, r=8, t=30, b=8),
        height=200,
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _bar_deviations(deviations: dict) -> go.Figure:
    items = [
        (k, v["pct_change"])
        for k, v in deviations.items()
        if abs(v.get("pct_change", 0)) > 3
    ]
    if not items:
        return None
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    keys, pcts = zip(*items[:10])
    colors = ["#ef4444" if p > 0 else "#22c55e" for p in pcts]
    fig = go.Figure(go.Bar(
        x=list(pcts), y=list(keys),
        orientation="h",
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Deviations from Personalized Baseline (%)", font=dict(size=13)),
        margin=dict(l=8, r=60, t=30, b=8),
        height=max(180, len(keys) * 28 + 60),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏃 Digital Twin System")
    st.caption("Multi-Agent ACL Injury Risk & Rehab")
    st.divider()

    twin_store, _ = _load_stores()
    state = twin_store.load_latest(ATHLETE_ID)

    if state:
        st.markdown(f"""
        <div style="background:#eff6ff;border-radius:10px;padding:14px 16px;margin-bottom:8px">
          <div style="font-size:18px;font-weight:700;color:#1e40af">{state.name}</div>
          <div style="font-size:13px;color:#3b82f6;margin-top:2px">{state.athlete_id} · {state.sport.title()}</div>
          <div style="font-size:12px;color:#64748b;margin-top:6px">
            Age {state.age} · {state.height_m:.2f} m · {state.mass_kg:.0f} kg · {state.gender.upper()}
          </div>
        </div>
        """, unsafe_allow_html=True)

        injury_color = "#fef2f2" if state.active_injury else "#f0fdf4"
        injury_text_color = "#b91c1c" if state.active_injury else "#15803d"
        st.markdown(f"""
        <div style="background:{injury_color};border-radius:8px;padding:10px 14px;font-size:13px;color:{injury_text_color};font-weight:600">
          {"⚠️ " + (state.active_injury or "").replace("_", " ").title() if state.active_injury else "✅ No Active Injury"}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:10px;font-size:12px;color:#64748b">
          <b>Injury history:</b><br>{"<br>".join(f"• {h}" for h in state.injury_history) or "None"}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No athlete data found. Run `python main.py` first.")

    st.divider()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        st.success("✓ Claude API connected")
    else:
        st.warning("⚠ No API key – AI tabs need ANTHROPIC_API_KEY")

    st.divider()
    st.caption("Twin version: " + (f"v{state.version}" if state else "—"))
    st.caption("Sessions: " + (str(len(state.session_ids)) if state else "—"))


# ─── Guard ───────────────────────────────────────────────────────────────────

if state is None:
    st.error("No athlete data found. Please run `python main.py` first to initialize the digital twin.")
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⚠️ Risk Assessment",
    "🩺 Rehab Plan",
    "💬 AI Assistant",
    "🔬 What-If Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Digital Twin Overview")
    st.caption("Longitudinal biomechanics tracking across all 6 rehab sessions · Abigail Savoy · ACL Reconstruction")

    # KPI row
    snap = state.latest_snapshot or {}
    pain_list  = _pain_scores(state)
    asym_trend = _aligned(state, "knee_asymmetry_index")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Sessions Completed</div>
          <div class="metric-value">{len(state.session_ids)}</div>
          <div class="metric-sub">of planned 6</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Latest Pain Score (VAS)</div>
          <div class="metric-value">{pain_list[-1] if pain_list else "—"}</div>
          <div class="metric-sub">0 = no pain · 10 = worst</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        asym_now = asym_trend[-1] if asym_trend else 0
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Knee Asymmetry Index</div>
          <div class="metric-value">{asym_now:.1f}%</div>
          <div class="metric-sub">target &lt;5%</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        hip_add = snap.get("hip_adduction_r_mean", 0)
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Hip Adduction (R) Mean</div>
          <div class="metric-value">{hip_add:.1f}°</div>
          <div class="metric-sub">valgus proxy</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        dur = snap.get("session_duration_s", 0)
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Last Session Duration</div>
          <div class="metric-value">{dur:.0f}s</div>
          <div class="metric-sub">workload indicator</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        if asym_trend:
            fig = _sparkline(asym_trend, SESSION_LABELS, "#6366f1", "Knee Asymmetry Index (%) per Session")
            fig.add_hline(y=5, line_dash="dot", line_color="#22c55e",
                          annotation_text="Target <5%", annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if pain_list:
            fig = _sparkline(pain_list, SESSION_LABELS, "#f59e0b", "Pain Score (VAS 0–10) per Session")
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 2 — bilateral knee angles
    c3, c4 = st.columns(2)
    knee_r = _aligned(state, "knee_angle_r_mean")
    knee_l = _aligned(state, "knee_angle_l_mean")

    with c3:
        if knee_r and knee_l:
            labels = SESSION_LABELS[:max(len(knee_r), len(knee_l))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels[:len(knee_r)], y=knee_r,
                name="Right Knee", mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5), marker=dict(size=7),
            ))
            fig.add_trace(go.Scatter(
                x=labels[:len(knee_l)], y=knee_l,
                name="Left Knee", mode="lines+markers",
                line=dict(color="#a855f7", width=2.5, dash="dash"), marker=dict(size=7),
            ))
            fig.update_layout(
                title=dict(text="Bilateral Knee Flexion Mean (°)", font=dict(size=13)),
                margin=dict(l=8, r=8, t=36, b=8), height=220,
                legend=dict(
                    orientation="h", x=1, y=1,
                    xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.75)",
                    font=dict(size=11),
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        hip_r = _aligned(state, "hip_adduction_r_mean")
        hip_l = _aligned(state, "hip_adduction_l_mean")
        if hip_r and hip_l:
            labels = SESSION_LABELS[:max(len(hip_r), len(hip_l))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels[:len(hip_r)], y=hip_r,
                name="Right Hip Adduction", mode="lines+markers",
                line=dict(color="#ef4444", width=2.5), marker=dict(size=7),
            ))
            fig.add_trace(go.Scatter(
                x=labels[:len(hip_l)], y=hip_l,
                name="Left Hip Adduction", mode="lines+markers",
                line=dict(color="#f97316", width=2.5, dash="dash"), marker=dict(size=7),
            ))
            fig.add_hline(y=10, line_dash="dot", line_color="#ef4444",
                          annotation_text="Warn threshold 10°", annotation_position="top right")
            fig.update_layout(
                title=dict(text="Hip Adduction (Valgus Proxy) (°)", font=dict(size=13)),
                margin=dict(l=8, r=8, t=36, b=8), height=220,
                legend=dict(
                    orientation="h", x=1, y=1,
                    xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.75)",
                    font=dict(size=11),
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Session table
    st.markdown('<div class="section-header">Session Summary Table</div>', unsafe_allow_html=True)
    rows = []
    for i, sid in enumerate(SESSION_IDS):
        n_sessions = len(SESSION_IDS)
        def _get(key):
            vals = _aligned(state, key)
            if i < len(vals):
                return round(vals[i], 2)
            return "—"

        pain_vals = _pain_scores(state)
        pain_val  = round(pain_vals[i], 1) if i < len(pain_vals) else "—"

        notes_all = state.injury_notes_history
        n = len(SESSION_IDS)
        notes_list = notes_all[-n:] if len(notes_all) >= n else notes_all
        note = notes_list[i] if i < len(notes_list) else "—"

        rows.append({
            "Session": f"{SESSION_LABELS[i]}",
            "Session ID": sid,
            "Pain (VAS)": pain_val,
            "Knee Asym (%)": _get("knee_asymmetry_index"),
            "Hip Add R (°)": _get("hip_adduction_r_mean"),
            "Knee R Mean (°)": _get("knee_angle_r_mean"),
            "Knee L Mean (°)": _get("knee_angle_l_mean"),
            "Clinical Note": note,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Injury Risk Assessment")
    st.caption("Ensemble: rule-based anomaly detection + Claude LLM reasoning")

    has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_api:
        st.info("Set ANTHROPIC_API_KEY to enable live LLM risk assessment. Showing rule-based results below.")

    col_btn, col_space = st.columns([2, 5])
    with col_btn:
        run_risk = st.button("▶ Run Risk Assessment", type="primary", use_container_width=True)

    if run_risk or "risk_result" in st.session_state:
        if run_risk:
            _, risk_agent, _, _ = _load_agents()
            with st.spinner("Running ensemble risk assessment…"):
                try:
                    session_id = state.session_ids[-1] if state.session_ids else "latest"
                    result = risk_agent.assess(state, session_id)
                    st.session_state["risk_result"] = result
                except Exception as e:
                    err = str(e)
                    if "credit balance" in err.lower() or "billing" in err.lower():
                        st.error("⚠️ API 余额不足，请前往 console.anthropic.com → Plans & Billing 充值。")
                    else:
                        st.error(f"请求失败：{err}")
                    st.stop()

        result = st.session_state["risk_result"]
        color  = RISK_COLORS.get(result.risk_level, "#94a3b8")

        # Risk level hero
        c_level, c_conf, c_drivers = st.columns([2, 1, 3])
        with c_level:
            st.markdown(f"""
            <div style="text-align:center;padding:24px 0">
              <div style="font-size:13px;color:#64748b;margin-bottom:8px">OVERALL RISK LEVEL</div>
              <span class="risk-badge" style="background:{color};color:white;font-size:24px;padding:10px 28px">
                {result.risk_level}
              </span>
            </div>
            """, unsafe_allow_html=True)

        with c_conf:
            st.metric("Confidence", f"{result.confidence:.0%}")
            st.metric("Session", result.session_id)

        with c_drivers:
            st.markdown('<div class="section-header">Top Risk Drivers</div>', unsafe_allow_html=True)
            for driver in result.top_risk_drivers:
                st.markdown(f'<span class="tag restriction-tag">⚠ {driver}</span>', unsafe_allow_html=True)

        # Reasoning
        st.markdown('<div class="section-header">Clinical Reasoning (LLM)</div>', unsafe_allow_html=True)
        st.info(result.reasoning or "No reasoning provided.")

        # Deviation bar chart
        if state.deviations:
            fig = _bar_deviations(state.deviations)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Raw deviations expander
        with st.expander("Raw deviation data"):
            st.json({k: v for k, v in state.deviations.items()
                     if abs(v.get("pct_change", 0)) > 1})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: REHAB PLAN
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Rehabilitation Plan")
    st.caption("ACL reconstruction protocol · Stage-aware planning · Evidence-based progression criteria")

    if not state.active_injury:
        st.success("No active injury recorded for this athlete.")
    else:
        col_btn2, _ = st.columns([2, 5])
        with col_btn2:
            run_rehab = st.button("▶ Generate Rehab Plan", type="primary", use_container_width=True)

        if run_rehab or "rehab_result" in st.session_state:
            if run_rehab:
                _, _, rehab_agent, _ = _load_agents()
                with st.spinner("Generating personalized rehab plan…"):
                    try:
                        session_id = state.session_ids[-1] if state.session_ids else "latest"
                        plan = rehab_agent.plan(state, session_id)
                        st.session_state["rehab_result"] = plan
                    except Exception as e:
                        err = str(e)
                        if "credit balance" in err.lower() or "billing" in err.lower():
                            st.error("⚠️ API 余额不足，请前往 console.anthropic.com → Plans & Billing 充值。")
                        else:
                            st.error(f"请求失败：{err}")
                        st.stop()

            plan = st.session_state["rehab_result"]

            # Stage progress bar
            stage_idx = next(
                (i for i, s in enumerate(STAGE_ORDER) if s.lower().startswith(plan.current_stage.lower().split()[0])),
                0
            )
            st.markdown('<div class="section-header">Recovery Stage</div>', unsafe_allow_html=True)
            prog_cols = st.columns(len(STAGE_ORDER))
            for i, s in enumerate(STAGE_ORDER):
                with prog_cols[i]:
                    is_current = (i == stage_idx)
                    bg  = "#6366f1" if is_current else ("#e0e7ff" if i < stage_idx else "#f1f5f9")
                    txt = "white" if is_current else ("#6366f1" if i < stage_idx else "#94a3b8")
                    st.markdown(f"""
                    <div style="background:{bg};color:{txt};border-radius:8px;padding:8px 6px;
                                text-align:center;font-size:12px;font-weight:{'700' if is_current else '400'}">
                      {"✓ " if i < stage_idx else ("● " if is_current else "")}{s.split("(")[0].strip()}
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Status + session
            status_colors = {
                "On Track": "#22c55e", "Delayed": "#f59e0b",
                "At Risk": "#ef4444", "Ready to Progress": "#6366f1",
            }
            sc = status_colors.get(plan.progress_status, "#94a3b8")
            st.markdown(f"""
            <span style="background:{sc};color:white;padding:5px 14px;border-radius:20px;font-weight:600;font-size:14px">
              {plan.progress_status}
            </span>
            &nbsp;&nbsp;<span style="font-size:13px;color:#64748b">Session: {plan.session_id}</span>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Three columns: exercises, restrictions, criteria
            col_ex, col_re, col_cr = st.columns(3)

            with col_ex:
                st.markdown('<div class="section-header">📋 Weekly Exercises</div>', unsafe_allow_html=True)
                for ex in plan.weekly_exercises:
                    st.markdown(f'<span class="tag">✓ {ex}</span>', unsafe_allow_html=True)

            with col_re:
                st.markdown('<div class="section-header">🚫 Restrictions</div>', unsafe_allow_html=True)
                if plan.restrictions:
                    for r in plan.restrictions:
                        st.markdown(f'<span class="tag restriction-tag">✗ {r}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag criteria-tag">No restrictions at this stage</span>',
                                unsafe_allow_html=True)

            with col_cr:
                st.markdown('<div class="section-header">🎯 Progression Criteria</div>', unsafe_allow_html=True)
                for c in plan.progression_criteria:
                    st.markdown(f'<span class="tag criteria-tag">→ {c}</span>', unsafe_allow_html=True)

            # Reasoning
            st.markdown('<div class="section-header">Clinical Reasoning (LLM)</div>', unsafe_allow_html=True)
            st.info(plan.reasoning or "No reasoning provided.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### AI Assistant – Natural Language Query Interface")
    st.caption("Powered by Claude Sonnet · Tool-use routing to specialist agents")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY not set. This tab requires the Claude API.")
        st.stop()

    # Role selector
    role = st.selectbox(
        "Your role",
        ["coach", "medical", "trainer"],
        format_func=lambda r: {"coach": "🏋 Coach", "medical": "🩺 Medical Staff", "trainer": "💪 Trainer"}[r],
    )

    role_hints = {
        "coach": "Try: \"Can Abigail join today's training session?\"",
        "medical": "Try: \"Show me Abigail's biomechanical deviations and rehab progression.\"",
        "trainer": "Try: \"What cutting drills are safe for Abigail right now?\"",
    }
    st.caption(role_hints[role])

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "tool_calls_log" not in st.session_state:
        st.session_state["tool_calls_log"] = []

    # Display history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <b>You ({role}):</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 <b>AI:</b><br>{msg["content"]}</div>',
                        unsafe_allow_html=True)

    # Tool calls log
    if st.session_state["tool_calls_log"]:
        with st.expander(f"🔧 Tool calls ({len(st.session_state['tool_calls_log'])})"):
            for tc in st.session_state["tool_calls_log"]:
                st.markdown(f'<div class="tool-call">{tc}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask the AI…", placeholder=role_hints[role].replace("Try: ", ""))
        submitted  = st.form_submit_button("Send", use_container_width=False)

    if submitted and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        _, _, _, decision_agent = _load_agents()

        # Monkey-patch to capture tool calls for display
        _orig_execute = decision_agent._execute_tool
        def _patched_execute(name, inputs):
            log = f"→ {name}({', '.join(f'{k}={v!r}' for k, v in inputs.items())})"
            st.session_state["tool_calls_log"].append(log)
            return _orig_execute(name, inputs)
        decision_agent._execute_tool = _patched_execute

        with st.spinner("Thinking…"):
            try:
                response = decision_agent.query(user_input, athlete_id=ATHLETE_ID, role=role)
            except Exception as e:
                err = str(e)
                if "credit balance" in err.lower() or "billing" in err.lower():
                    response = "⚠️ **API 余额不足** — 请前往 [console.anthropic.com](https://console.anthropic.com) → Plans & Billing 充值后重试。"
                elif "api_key" in err.lower() or "authentication" in err.lower():
                    response = "⚠️ **API Key 无效** — 请检查 ANTHROPIC_API_KEY 是否正确。"
                else:
                    response = f"⚠️ **请求失败：** {err}"

        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("Clear chat", type="secondary"):
        st.session_state["chat_history"] = []
        st.session_state["tool_calls_log"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: WHAT-IF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### What-If / Counterfactual Analysis")
    st.caption("Causal reasoning: how would a given intervention change the recovery trajectory?")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY not set. This tab requires the Claude API.")
    else:
        examples = [
            "Reduce training workload by 30% for the next 2 weeks",
            "Add plyometric box jumps to the weekly program starting next session",
            "Skip the next 2 rehab sessions due to travel",
            "Increase session frequency from once to twice per week",
            "Start full-speed cutting drills immediately",
        ]

        # Persist intervention text across reruns
        if "whatif_text" not in st.session_state:
            st.session_state["whatif_text"] = ""
        if "whatif_result" not in st.session_state:
            st.session_state["whatif_result"] = None

        st.markdown("**Quick examples:**")
        ex_cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            with ex_cols[i]:
                if st.button(ex[:40] + "…" if len(ex) > 40 else ex, key=f"ex_{i}"):
                    st.session_state["whatif_text"] = ex
                    st.session_state["whatif_result"] = None
                    st.rerun()

        # Text area bound to session_state key — value persists across reruns
        intervention = st.text_area(
            "Describe the intervention:",
            key="whatif_text",
            height=80,
            placeholder="e.g. Reduce workload by 20% for 2 weeks",
        )

        if st.button("🔬 Analyze", type="primary"):
            if not intervention.strip():
                st.warning("Please enter or select an intervention first.")
            else:
                _, _, rehab_agent, _ = _load_agents()
                with st.spinner("Reasoning through counterfactual scenario…"):
                    try:
                        result = rehab_agent.counterfactual(state, intervention)
                        st.session_state["whatif_result"] = (intervention, result)
                    except Exception as e:
                        err = str(e)
                        if "credit balance" in err.lower() or "billing" in err.lower():
                            st.error("⚠️ API 余额不足，请前往 console.anthropic.com → Plans & Billing 充值。")
                        else:
                            st.error(f"请求失败：{err}")

        if st.session_state["whatif_result"]:
            saved_intervention, result = st.session_state["whatif_result"]
            st.markdown('<div class="section-header">Counterfactual Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#f8fafc;border-left:4px solid #6366f1;border-radius:0 10px 10px 0;padding:16px 20px;font-size:14px;line-height:1.7">
              <b>Intervention:</b> <i>{saved_intervention}</i><br><br>{result}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">About This Feature</div>', unsafe_allow_html=True)
        st.markdown("""
        The What-If module uses **causal reasoning** to predict downstream effects of clinical interventions.
        It reasons over:
        - Recovery trajectory (weeks to next stage)
        - Biomechanical adaptation expectations
        - Re-injury risk change
        - Estimated return-to-play timeline shift

        *Counterfactual reasoning component of the Digital Twin Framework*
        """)
