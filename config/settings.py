"""
Configuration loader — AI Decision Engine.
Reads from .env, falls back to environment variables, then safe defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
    def summary(cls) -> dict:
        return {
            "openai_model":            cls.OPENAI_MODEL,
            "simulation_mode":         cls.simulation_mode(),
            "confidence_threshold":    cls.CONFIDENCE_THRESHOLD,
            "min_conversion_rate":     cls.MIN_CONVERSION_RATE,
            "max_false_positive_rate": cls.MAX_FALSE_POSITIVE_RATE,
            "db_path":                 str(cls.DB_PATH),
        }


config = Config()
