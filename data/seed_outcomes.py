"""
Seed outcome feedback for the 50-lead sample dataset.
Run after python main.py has populated data/decisions.db.

Usage:
    python data/seed_outcomes.py

Idempotent: outcomes are one-per-lead (UNIQUE(lead_id) on the outcomes
table, audit finding M3). Re-running this script against an already
seeded DB skips leads that already have a recorded outcome instead of
erroring out or replacing it — there is no correction path in v1 (see
README Known Limitations).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import init_db
from pipeline.outcome_handler import record_outcome, OutcomeError, OutcomeConflictError
from models.schemas import OutcomeType

OUTCOMES = {
    # send_to_sales -> converted (clear high-value leads)
    "p2_001": "converted", "p2_002": "converted", "p2_003": "converted",
    "p2_004": "converted", "p2_005": "converted", "p2_006": "converted",
    "p2_007": "converted", "p2_008": "converted",
    # send_to_sales -> not_converted (false positives: AI over-classified)
    "p2_009": "not_converted", "p2_010": "not_converted",
    "p2_011": "not_converted", "p2_012": "not_converted",
    # send_to_sales -> delayed (regulatory/procurement holds)
    "p2_013": "delayed", "p2_014": "delayed",
    # send_to_sales -> converted (borderline confidence, real buyer)
    "p2_015": "converted", "p2_016": "converted",
    # send_to_sales -> wrong_segment (product fit mismatch)
    "p2_035": "wrong_segment", "p2_036": "wrong_segment",
    # send_to_sales -> converted (additional high-value, varied industries)
    "p2_039": "converted", "p2_040": "converted",
    "p2_041": "converted", "p2_042": "converted",
    "p2_046": "converted", "p2_047": "converted",
    "p2_050": "converted",
    # archive -> converted (missed opportunities: AI under-classified real buyers)
    "p2_028": "converted", "p2_029": "converted",
    # manual_review -> converted (missed opportunities: low-confidence leads that converted)
    "p2_037": "converted", "p2_038": "converted",
    # manual_review -> no purchase (correct decisions, adds outcome coverage)
    "p2_030": "ignored", "p2_031": "ignored", "p2_049": "not_converted",
}


def seed():
    init_db()

    inserted         = 0
    already_recorded = 0
    skipped          = 0

    for lead_id, outcome_str in OUTCOMES.items():
        try:
            record_outcome(lead_id, OutcomeType(outcome_str))
            inserted += 1
        except OutcomeConflictError:
            already_recorded += 1
        except OutcomeError as e:
            print(f"  SKIP {lead_id}: {e}")
            skipped += 1

    covered = inserted + already_recorded
    print(f"\nOutcomes seeded: {inserted} inserted, {already_recorded} already recorded, {skipped} skipped")
    print(f"Coverage: {covered}/50 = {round(covered / 50 * 100)}%")
    print("\nNext: GET /stats to evaluate decision quality")


if __name__ == "__main__":
    seed()
