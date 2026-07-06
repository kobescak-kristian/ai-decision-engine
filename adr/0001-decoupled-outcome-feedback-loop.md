# ADR 0001: Decouple Decisions and Outcomes, Join Async for Evaluation

## Status
Accepted (implemented)

## Date: 2026-07-04

## Context
The pipeline makes a routing decision (`send_to_sales` / `archive` / `manual_review`)
at the moment a lead is processed (`main.py`), based only on information available
at that time. Whether the decision was actually correct — did the lead convert, was
a real buyer missed — is only known later, at a delay measured in days or weeks, and
from a different source (sales/CRM feedback via `/outcome`, not the AI).

If outcome data were written onto the decision record at pipeline time, "decision
quality at the time it was made" and "later-observed result" would be conflated, and
there would be no way to represent "outcome not known yet" versus "no outcome exists."

## Decision
Decisions and outcomes are stored in two separate SQLite tables (`database/db.py`),
linked only by `lead_id`:
- `decisions` — one row per lead, written once by `save_decision()` at pipeline time.
- `outcomes` — appended later by `record_outcome()` (`pipeline/outcome_handler.py`)
  via `POST /outcome`, keyed to an existing `lead_id` (raises `OutcomeError` if the
  lead was never processed through `/qualify`).

The evaluator (`pipeline/evaluator.py::compute_metrics()`) reads both tables via
`get_evaluation_data()`, which joins them (`outcomes o JOIN decisions d ON
o.lead_id = d.lead_id`) to compute conversion rate, false-positive rate, and
missed-opportunity count — never from the decision record alone.

A coverage gate protects against evaluating on too little data: if
`outcome_coverage < 0.3` (fewer than 30% of decisions have a recorded outcome),
`_generate_insights()` returns `status: "insufficient_data"` instead of a quality
verdict.

## Consequences
- Decision quality is measured from real-world outcomes as they accumulate — never
  inferred from AI confidence alone.
- Outcomes can arrive on their own timeline, independent of pipeline runs.
- Recording an outcome for an unknown `lead_id` fails loudly (`OutcomeError`) rather
  than silently creating an orphan record.
- Trade-off: below 30% coverage, `/stats` reports `insufficient_data` rather than a
  number — by design, but it means early in rollout the system cannot yet say
  whether it's working.
- The 30% gate and the three evaluation thresholds (`MIN_CONVERSION_RATE`,
  `MAX_FALSE_POSITIVE_RATE`, `MAX_MANUAL_REVIEW_RATE`, `config/settings.py`) are
  static config values, not learned — tuning them is a manual step, not automated.

## Clarification (2026-07-06)
"Written once" above described intent, not the original implementation: `save_decision()`
used `INSERT OR REPLACE` keyed on `lead_id`, so re-qualifying a lead silently destroyed
the prior decision row — including any outcome already evaluated against it. Decisions
are now append-only and keyed by `(lead_id, version)`; re-qualification inserts a new
version rather than overwriting. "One row per lead" is superseded by "one row per
lead per version, and every version is retained." `get_evaluation_data()` and the
default `/audit` list read only the latest version per lead (via the `latest_decisions`
view); `GET /audit/{lead_id}` exposes the full version history. This is the mechanism
that makes "written once" true: each version, once written, is never altered or removed.
