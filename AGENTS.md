# AGENTS.md — ai-decision-engine

Tool-neutral context router for coding agents working in this repository.
It points to where repository truth lives. It is not evidence of
implementation or project state: cite the source it routes to, never this
file.

## Repository purpose

AI-assisted lead-routing pipeline with a delayed feedback loop. It records
each routing decision, accepts the real-world outcome later, and evaluates
decision quality from those outcomes instead of stopping at classification.
Human overview: `README.md`.

## Authority and conflict handling

| Question | Canonical source |
|---|---|
| What the system is and claims publicly | `README.md` |
| Material design decisions and their rationale | `adr/` |
| Documented version history | `README.md` Version Log |
| What the implementation actually does | affected source, tests and configuration inspected together |

When sources disagree:

- An adopted ADR, including its dated clarifications, governs the decision
  it records.
- Documents state intent; implementation, tests and configuration state
  behaviour. A gap between them is a finding to surface, not something to
  reconcile silently.
- If the requested task materially depends on an unresolved conflict, stop
  and ask the owner.

## Task routing

These are starting points, not exhaustive reading lists.

| Task class | Start here |
|---|---|
| Understand or explain the system | `README.md` |
| Decision pipeline: classification, validation, fallback, routing | `pipeline/`, `models/schemas.py`, `main.py`, `config/settings.py`, `tests/test_ci.py` |
| Decision and outcome storage, evaluation | `database/db.py`, `pipeline/outcome_handler.py`, `pipeline/evaluator.py`, `adr/0001` |
| HTTP API | `api.py`, then the pipeline and database routes above |
| Material design decision | `adr/` |
| Documentation, README or ADR work | the affected artifact; `.githooks/validate_artifacts.py` for enforced structure |
| CI, hooks, publishing | `.github/workflows/ci.yml`, `.githooks/`, `.publicgate-allow` |

## Always-on constraints

- Decisions are append-only and versioned per lead: never add a path that
  updates or deletes a stored decision version. Outcomes are one per lead:
  a duplicate is rejected, never overwritten or double-counted, and outcome
  seeding stays idempotent. A fallback decision keeps its original
  validation failure. See `adr/0001` and `database/db.py`.
- Without a valid model credential the system runs in explicit simulation
  mode. Do not silently restore network or model calls when credentials are
  absent, and do not turn authentication, network or repeated AI-layer
  failures into silent degradation.
- Development and verification use the keyless simulation path. Do not run
  paid or real-model execution without explicit owner authorization.
- Committed gates, frozen test assertions and published results are final
  records. Do not adjust thresholds or frozen expected values after a run,
  or rewrite them to make a check green.
- No secrets or machine-local absolute paths in tracked files.
  Machine-specific values go to the gitignored `.env`.
- Git history is evidence and commits are hash-pinned externally. Never
  rewrite repository history.
- Trigger-gated artifacts such as `CHANGELOG.md`, `RUNBOOK.md`, and
  `SYSTEM_WALKTHROUGH.md` are not created without a decision record
  citing the trigger. ADRs are created only for genuine material
  decisions; there is no hard maximum. The version log lives in
  `README.md`.
- `AGENTS.md` is an instruction surface, not a security boundary. Hooks, CI
  and tests remain the enforcement.

## Verification

Keyless and bounded; no model or API calls. Requires
`pip install -r requirements.txt` and `pytest`.

```bash
python .githooks/validate_artifacts.py .
pytest tests/ -v
```

`tests/test_ci.py` forces an empty model key (simulation mode). Generated
files under `data/` are gitignored.
