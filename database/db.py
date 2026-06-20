"""
SQLite persistence layer — AI Decision Engine.
Stores decisions and outcome feedback for performance evaluation.
Extended from P1 with new outcomes table and evaluation queries.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from config.settings import config
from utils.logger import logger


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables. Safe to call on every startup."""
    with _connect() as conn:

        # Decisions table — one row per lead processed
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id         TEXT NOT NULL UNIQUE,
                run_id          TEXT NOT NULL,
                raw_text        TEXT,
                received_at     TEXT,
                category        TEXT,
                confidence      REAL,
                reason          TEXT,
                final_decision  TEXT,
                fallback_action TEXT,
                processing_ms   REAL,
                created_at      TEXT NOT NULL
            )
        """)

        # Outcomes table — feedback loop, linked to decisions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id     TEXT NOT NULL,
                decision    TEXT NOT NULL,
                outcome     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES decisions(lead_id)
            )
        """)

        for idx, tbl, col in [
            ("idx_decisions_lead",     "decisions", "lead_id"),
            ("idx_decisions_decision", "decisions", "final_decision"),
            ("idx_outcomes_lead",      "outcomes",  "lead_id"),
            ("idx_outcomes_outcome",   "outcomes",  "outcome"),
        ]:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl}({col})")

    logger.debug("Database initialised")


def save_decision(result: dict, run_id: str):
    ai  = result.get("ai_output") or {}
    inp = result.get("input") or {}

    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO decisions
              (lead_id, run_id, raw_text, received_at, category, confidence, reason,
               final_decision, fallback_action, processing_ms, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            inp.get("id"),
            run_id,
            inp.get("raw_text"),
            inp.get("received_at"),
            ai.get("category"),
            ai.get("confidence"),
            ai.get("reason"),
            result.get("final_decision"),
            result.get("fallback_action"),
            result.get("processing_ms"),
            datetime.now(timezone.utc).isoformat()
        ))


def lead_exists(lead_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM decisions WHERE lead_id = ?", (lead_id,)
        ).fetchone()
    return row is not None


def get_lead_decision(lead_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE lead_id = ?", (lead_id,)
        ).fetchone()
    return dict(row) if row else None


def save_outcome(lead_id: str, decision: str, outcome: str, timestamp: str):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO outcomes (lead_id, decision, outcome, timestamp)
            VALUES (?, ?, ?, ?)
        """, (lead_id, decision, outcome, timestamp))


def get_evaluation_data() -> dict:
    """
    Pull raw data needed for the evaluation engine.
    Returns decision counts and outcome breakdowns.
    """
    with _connect() as conn:
        total_decisions = conn.execute(
            "SELECT COUNT(*) FROM decisions"
        ).fetchone()[0]

        by_decision = conn.execute("""
            SELECT final_decision, COUNT(*) as count
            FROM decisions GROUP BY final_decision
        """).fetchall()

        total_outcomes = conn.execute(
            "SELECT COUNT(*) FROM outcomes"
        ).fetchone()[0]

        outcome_by_decision = conn.execute("""
            SELECT d.final_decision, o.outcome, COUNT(*) as count
            FROM outcomes o
            JOIN decisions d ON o.lead_id = d.lead_id
            GROUP BY d.final_decision, o.outcome
        """).fetchall()

        # False positives: sent to sales but NOT converted
        false_positives = conn.execute("""
            SELECT COUNT(*) FROM outcomes o
            JOIN decisions d ON o.lead_id = d.lead_id
            WHERE d.final_decision = 'send_to_sales'
            AND o.outcome != 'converted'
        """).fetchone()[0]

        # Missed opportunities: manual_review or archive but WAS converted
        missed = conn.execute("""
            SELECT COUNT(*) FROM outcomes o
            JOIN decisions d ON o.lead_id = d.lead_id
            WHERE d.final_decision IN ('manual_review', 'archive')
            AND o.outcome = 'converted'
        """).fetchone()[0]

    return {
        "total_decisions":      total_decisions,
        "by_decision":          {row["final_decision"]: row["count"] for row in by_decision},
        "total_outcomes":       total_outcomes,
        "outcome_by_decision":  [dict(r) for r in outcome_by_decision],
        "false_positives":      false_positives,
        "missed_opportunities": missed,
    }


def get_recent_decisions(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT d.lead_id, d.final_decision, d.category, d.confidence,
                   d.fallback_action, d.created_at,
                   o.outcome, o.timestamp as outcome_timestamp
            FROM decisions d
            LEFT JOIN outcomes o ON d.lead_id = o.lead_id
            ORDER BY d.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def test_connection() -> bool:
    try:
        with _connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def generate_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
