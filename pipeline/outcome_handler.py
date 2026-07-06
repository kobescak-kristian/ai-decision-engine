"""
Outcome Handler — NEW in P2.
Receives outcome feedback and stores it against existing decisions.
This is the feedback loop: decisions are made first, outcomes come later.

Outcomes are one-per-lead (audit finding M3: a second POST for the same
lead was silently accepted, letting one lead count as both a conversion
and a false positive with rates summing over 100%). record_outcome()
now rejects a second outcome for an already-recorded lead rather than
adding a second row. There is no correction path in v1 — see README
Known Limitations.
"""

from datetime import datetime, timezone
from models.schemas import OutcomeType
from database.db import lead_exists, get_lead_decision, get_outcome, save_outcome
from utils.logger import logger


class OutcomeError(Exception):
    """Base for outcome-recording failures."""


class OutcomeConflictError(OutcomeError):
    """An outcome is already recorded for this lead."""


def record_outcome(lead_id: str, outcome: OutcomeType) -> dict:
    """
    Store an outcome for an existing lead decision.

    Raises OutcomeError if the lead_id does not exist in the decisions
    table, or OutcomeConflictError if this lead already has a recorded
    outcome (one outcome per lead; no correction path in v1).
    Returns a confirmation dict on success.
    """
    if not lead_exists(lead_id):
        logger.warning(f"Outcome rejected - unknown lead: {lead_id}")
        raise OutcomeError(f"Lead '{lead_id}' not found. Process it through /qualify first.")

    existing = get_outcome(lead_id)
    if existing:
        logger.warning(
            f"Outcome rejected - lead '{lead_id}' already has an outcome: "
            f"'{existing['outcome']}' recorded at {existing['timestamp']}"
        )
        raise OutcomeConflictError(
            f"Lead '{lead_id}' already has an outcome: '{existing['outcome']}' "
            f"recorded at {existing['timestamp']}. Outcome correction is not supported in v1."
        )

    decision_record = get_lead_decision(lead_id)
    decision = decision_record.get("final_decision", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()

    save_outcome(
        lead_id=lead_id,
        decision=decision,
        outcome=outcome.value,
        timestamp=timestamp
    )

    logger.success(f"Outcome recorded: {lead_id} -> {outcome.value} (decision was: {decision})")

    return {
        "lead_id":   lead_id,
        "decision":  decision,
        "outcome":   outcome.value,
        "timestamp": timestamp
    }
