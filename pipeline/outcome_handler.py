"""
Outcome Handler — NEW in P2.
Receives outcome feedback and stores it against existing decisions.
This is the feedback loop: decisions are made first, outcomes come later.
"""

from datetime import datetime, timezone
from models.schemas import OutcomeType
from database.db import lead_exists, get_lead_decision, save_outcome
from utils.logger import logger


class OutcomeError(Exception):
    pass


def record_outcome(lead_id: str, outcome: OutcomeType) -> dict:
    """
    Store an outcome for an existing lead decision.

    Raises OutcomeError if the lead_id does not exist in the decisions table.
    Returns a confirmation dict on success.
    """
    if not lead_exists(lead_id):
        logger.warning(f"Outcome rejected — unknown lead: {lead_id}")
        raise OutcomeError(f"Lead '{lead_id}' not found. Process it through /qualify first.")

    decision_record = get_lead_decision(lead_id)
    decision = decision_record.get("final_decision", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()

    save_outcome(
        lead_id=lead_id,
        decision=decision,
        outcome=outcome.value,
        timestamp=timestamp
    )

    logger.success(f"Outcome recorded: {lead_id} → {outcome.value} (decision was: {decision})")

    return {
        "lead_id":   lead_id,
        "decision":  decision,
        "outcome":   outcome.value,
        "timestamp": timestamp
    }
