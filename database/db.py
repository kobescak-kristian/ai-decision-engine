"""
SQLite persistence layer — AI Decision Engine.
Stores decisions and outcome feedback for performance evaluation.
Includes decisions table and outcomes table for the feedback evaluation loop.

Decisions are append-only and versioned per lead_id (audit finding M2:
`INSERT OR REPLACE` destroyed the original decision on re-qualification,
contradicting ADR-0001's "written once"). Re-qualifying a lead inserts a
new (lead_id, version) row; no UPDATE or DELETE path exists on decisions.
All reads that represent "the current decision" (metrics, the default
audit list) use the latest version per lead via the `latest_decisions`
view; the raw `decisions` table preserves full history.

This is a fresh schema, not a migration. The demo DB (data/decisions.db)
is generated and gitignored — delete it before running against this
version of the code; a pre-existing DB from the old (unversioned) schema
will not be altered by CREATE TABLE IF NOT EXISTS and will error on the
new INSERT.

Outcomes are one-per-lead (audit finding M3: duplicate POSTs to /outcome
let a single lead count as both a conversion and a false positive,
rates summing over 100%). `outcomes.lead_id` is UNIQUE; a second outcome
for an already-recorded lead is rejected in pipeline/outcome_handler.py,
not silently replaced. Same fresh-schema caveat as above applies to a
pre-existing DB from before this constraint existed.

decisions.validation_passed/validation_errors persist the ORIGINAL
validation outcome for a record, before any safe-default fallback was
applied (the validation-overwrite pattern from the Reliability engine
audit, M1 there: re-validating the safe default always passes, and
reassigning the validation result to that pass silently erased the
real failure). A fallback record is written with validation_passed=0
and the real failure reason in validation_errors; that state is set
once at INSERT time and is never revised afterward, by design (see
save_decision()). Clean records are validation_passed=1,
validation_errors=NULL.
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

        # Decisions table — append-only, one row per (lead_id, version).
        # Re-qualifying a lead adds a new version; nothing is ever replaced.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id         TEXT NOT NULL,
                version         INTEGER NOT NULL,
                run_id          TEXT NOT NULL,
                raw_text        TEXT,
                received_at     TEXT,
                category        TEXT,
                confidence      REAL,
                reason          TEXT,
                final_decision  TEXT,
                fallback_action TEXT,
                processing_ms   REAL,
                validation_passed INTEGER NOT NULL,
                validation_errors TEXT,
                created_at      TEXT NOT NULL,
                UNIQUE(lead_id, version)
            )
        """)

        # Outcomes table — feedback loop, linked to decisions by lead_id
        # (never by row id), so it stays valid across decision versions.
        # One outcome per lead: lead_id is UNIQUE, so a duplicate INSERT
        # fails at the DB level even if the application-level check in
        # pipeline/outcome_handler.py is ever bypassed.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id     TEXT NOT NULL UNIQUE,
                decision    TEXT NOT NULL,
                outcome     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES decisions(lead_id)
            )
        """)

        # Current decision per lead = highest version. Everything that
        # reads "the" decision (metrics, default audit list) goes through
        # this view; full history stays queryable straight off `decisions`.
        conn.execute("""
            CREATE VIEW IF NOT EXISTS latest_decisions AS
            SELECT d.*
            FROM decisions d
            JOIN (
                SELECT lead_id, MAX(version) AS max_version
                FROM decisions
                GROUP BY lead_id
            ) latest
            ON d.lead_id = latest.lead_id AND d.version = latest.max_version
        """)

        for idx, tbl, col in [
            ("idx_decisions_lead",     "decisions", "lead_id"),
            ("idx_decisions_decision", "decisions", "final_decision"),
            ("idx_outcomes_lead",      "outcomes",  "lead_id"),
            ("idx_outcomes_outcome",   "outcomes",  "outcome"),
        ]:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl}({col})")

    logger.debug("Database initialised")


def save_decision(
    result: dict, run_id: str,
    validation_passed: bool, validation_errors: str | None = None
) -> int:
    """
    Append a new decision version for this lead. Never overwrites: the
    next version number is one past the current max for this lead_id
    (1 for a lead seen for the first time). Returns the version written.

    validation_passed/validation_errors are supplied explicitly by the
    caller and capture the ORIGINAL validation outcome for this record -
    before any safe-default fallback was applied. This is set once, at
    insert time, and never revised: a fallback record is persisted with
    validation_passed=0 and the real failure reason, never with a
    passing state (mirrors the Reliability audit's M1 fix).
    """
    ai  = result.get("ai_output") or {}
    inp = result.get("input") or {}
    lead_id = inp.get("id")

    with _connect() as conn:
        prev_version = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM decisions WHERE lead_id = ?", (lead_id,)
        ).fetchone()[0]
        version = prev_version + 1

        conn.execute("""
            INSERT INTO decisions
              (lead_id, version, run_id, raw_text, received_at, category, confidence, reason,
               final_decision, fallback_action, processing_ms, validation_passed,
               validation_errors, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            lead_id,
            version,
            run_id,
            inp.get("raw_text"),
            inp.get("received_at"),
            ai.get("category"),
            ai.get("confidence"),
            ai.get("reason"),
            result.get("final_decision"),
            result.get("fallback_action"),
            result.get("processing_ms"),
            1 if validation_passed else 0,
            validation_errors,
            datetime.now(timezone.utc).isoformat()
        ))

    if version > 1:
        logger.warning(
            f"[{lead_id}] Re-qualified - this is version {version}. "
            f"Previous version(s) are not overwritten and remain queryable via GET /audit/{lead_id}."
        )

    return version


def lead_exists(lead_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM decisions WHERE lead_id = ?", (lead_id,)
        ).fetchone()
    return row is not None


def get_lead_decision(lead_id: str) -> dict | None:
    """The current (latest-version) decision for this lead."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM latest_decisions WHERE lead_id = ?", (lead_id,)
        ).fetchone()
    return dict(row) if row else None


def get_decision_history(lead_id: str) -> list[dict]:
    """Every decision version for this lead, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE lead_id = ? ORDER BY version ASC", (lead_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_outcome(lead_id: str) -> dict | None:
    """The single recorded outcome for this lead, if any (outcomes are
    one-per-lead — UNIQUE(lead_id))."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM outcomes WHERE lead_id = ?", (lead_id,)
        ).fetchone()
    return dict(row) if row else None


def get_lead_outcomes(lead_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM outcomes WHERE lead_id = ? ORDER BY timestamp ASC", (lead_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_outcome(lead_id: str, decision: str, outcome: str, timestamp: str):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO outcomes (lead_id, decision, outcome, timestamp)
            VALUES (?, ?, ?, ?)
        """, (lead_id, decision, outcome, timestamp))


def get_evaluation_data() -> dict:
    """
    Pull raw data needed for the evaluation engine.
    Returns decision counts and outcome breakdowns — always computed
    against the latest decision version per lead, never raw history.
    """
    with _connect() as conn:
        total_decisions = conn.execute(
            "SELECT COUNT(*) FROM latest_decisions"
        ).fetchone()[0]

        by_decision = conn.execute("""
            SELECT final_decision, COUNT(*) as count
            FROM latest_decisions GROUP BY final_decision
        """).fetchall()

        total_outcomes = conn.execute(
            "SELECT COUNT(*) FROM outcomes"
        ).fetchone()[0]

        outcome_by_decision = conn.execute("""
            SELECT d.final_decision, o.outcome, COUNT(*) as count
            FROM outcomes o
            JOIN latest_decisions d ON o.lead_id = d.lead_id
            GROUP BY d.final_decision, o.outcome
        """).fetchall()

        # False positives: sent to sales but NOT converted
        false_positives = conn.execute("""
            SELECT COUNT(*) FROM outcomes o
            JOIN latest_decisions d ON o.lead_id = d.lead_id
            WHERE d.final_decision = 'send_to_sales'
            AND o.outcome != 'converted'
        """).fetchone()[0]

        # Missed opportunities: manual_review or archive but WAS converted
        missed = conn.execute("""
            SELECT COUNT(*) FROM outcomes o
            JOIN latest_decisions d ON o.lead_id = d.lead_id
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
    """Latest decision version per lead, with outcome status if any."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT d.lead_id, d.version, d.final_decision, d.category, d.confidence,
                   d.fallback_action, d.validation_passed, d.validation_errors, d.created_at,
                   o.outcome, o.timestamp as outcome_timestamp
            FROM latest_decisions d
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
