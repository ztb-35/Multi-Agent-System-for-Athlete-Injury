"""
Global configuration for the Multi-Agent Athlete Injury System (MASAI).
"""
import os

# ── LLM ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = "claude-sonnet-4-6"
LLM_MAX_TOKENS: int = 4096
LLM_TEMPERATURE: float = 0.0          # deterministic for medical reasoning

# ── Memory / Storage ─────────────────────────────────────────────────────────
MEMORY_DIR: str = os.getenv("MASAI_MEMORY_DIR", "./data/memory")
TWIN_STORE_DIR: str = os.path.join(MEMORY_DIR, "twins")
SESSION_STORE_DIR: str = os.path.join(MEMORY_DIR, "sessions")

# ── Risk thresholds (per-athlete baselines override these defaults) ───────────
DEFAULT_RISK_HIGH_THRESHOLD: float = 0.70
DEFAULT_RISK_MEDIUM_THRESHOLD: float = 0.40
DEFAULT_FATIGUE_HIGH_THRESHOLD: float = 0.75

# ── Rehabilitation ────────────────────────────────────────────────────────────
REHAB_CHECK_INTERVAL_DAYS: int = 7    # how often to auto-evaluate progress
RETURN_TO_SPORT_MIN_SCORE: float = 0.85

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("MASAI_LOG_LEVEL", "INFO")
