"""
Configuration loader — AI Decision Engine.
Reads from .env, falls back to environment variables, then safe defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# A placeholder key (the old env.example value) must behave like no key at
# all — otherwise `cp env.example .env` silently disables simulation mode
# and every AI call fails with 401 (audit finding B1).
PLACEHOLDER_API_KEYS = {"your_openai_api_key_here"}


class Config:
    # ── OpenAI ────────────────────────────────────────────────────────────
    _RAW_OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_API_KEY: str = (
        "" if _RAW_OPENAI_API_KEY.lower() in PLACEHOLDER_API_KEYS
        else _RAW_OPENAI_API_KEY
    )
    OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Standard OpenAI SDK override — lets ops point at a proxy/mock, and
    # lets tests point at an unreachable endpoint to exercise the abort path.
    OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", "").strip() or None

    # ── Pipeline ──────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

    # ── Evaluation thresholds ─────────────────────────────────────────────
    MIN_CONVERSION_RATE: float  = float(os.getenv("MIN_CONVERSION_RATE", "0.40"))
    MAX_FALSE_POSITIVE_RATE: float = float(os.getenv("MAX_FALSE_POSITIVE_RATE", "0.30"))
    MAX_MANUAL_REVIEW_RATE: float  = float(os.getenv("MAX_MANUAL_REVIEW_RATE", "0.25"))

    # ── Paths ─────────────────────────────────────────────────────────────
    DB_PATH: Path = Path(os.getenv("DB_PATH", "data/decisions.db"))

    @classmethod
    def simulation_mode(cls) -> bool:
        return not bool(cls.OPENAI_API_KEY)

    @classmethod
    def simulation_reason(cls) -> str | None:
        """Why the system is in simulation mode; None when a real key is set."""
        if not cls.simulation_mode():
            return None
        if cls._RAW_OPENAI_API_KEY:
            return "OPENAI_API_KEY is the env.example placeholder - treated as unset"
        return "OPENAI_API_KEY is not set"

    @classmethod
    def summary(cls) -> dict:
        return {
            "openai_model":            cls.OPENAI_MODEL,
            "simulation_mode":         cls.simulation_mode(),
            "confidence_threshold":    cls.CONFIDENCE_THRESHOLD,
            "min_conversion_rate":     cls.MIN_CONVERSION_RATE,
            "max_false_positive_rate": cls.MAX_FALSE_POSITIVE_RATE,
            "max_manual_review_rate":  cls.MAX_MANUAL_REVIEW_RATE,
            "db_path":                 str(cls.DB_PATH),
        }


config = Config()
