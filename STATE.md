# STATE — ai-decision-engine

**Classification:** PROJECT · T0 (AI-assisted lead-routing pipeline with a delayed feedback loop; `domains/github-ops/CONVENTIONS.md` PROJECT/SYSTEM/EXPERIMENT taxonomy).

**RECONSTRUCTED** (GOVERNANCE.md Build-repo STATE rule, clause 7): derived from git history and the README Version Log at scaffold time (2026-09-19, Q-72(f)), not written contemporaneously. Reconstructed entries are retrospective evidence, not contemporaneous record — the commit that adds this file begins the contemporaneous record going forward.

## Current state

No `## Status` heading exists in this README (unlike ai-reliability-engine/ai-impact-scoring-engine/ai-execution-engine) — the closest analog is a "**Type:** System design / AI evaluation pipeline" line and the title's own version tag, currently `v1.4`. One ADR: `adr/0001-decoupled-outcome-feedback-loop.md` (decisions and outcomes join asynchronously for evaluation).

## Version Log (from README, verbatim)

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-04-23 | Initial release — decision pipeline, feedback loop, API |
| v1.0 | 2026-04-24 | Added System Context section; clarified project title |
| v1.0 | 2026-06-15 – 2026-06-18 | Fixed README casing/formatting; corrected System Context references across all five engines |
| v1.0 | 2026-06-20 | Added outcome seeding script and demo sequence; fixed decision routing count; cleaned up low-severity audit items |
| v1.0 | 2026-07-04 | Adopted ARTIFACT_STANDARD Tier 0 — CLAUDE.md, pre-push validation, README restructure, first ADR |
| v1.1 | 2026-07-05 | Audit remediation (B1, silent degradation): placeholder key detection, simulation-mode banner, loud abort on auth failure, fallback count and reasons in run summary |
| v1.1 | 2026-07-06 | Audit remediation (silent degradation, follow-up): loud abort on unreachable API and on repeated identical AI-layer failures |
| v1.2 | 2026-07-06 | Audit remediation (M2, mutable decisions): decisions now append-only and versioned per lead_id |
| v1.3 | 2026-07-06 | Audit remediation (M3, duplicate outcomes): outcomes one-per-lead, 409 on repeat, idempotent seeding |
| v1.4 | 2026-07-06 | Audit remediation (validation-overwrite, mirrors Reliability's M1): persisted decisions now reflect the ORIGINAL validation outcome |

## Build history since the Version Log's last entry (from `git log --reverse`)

- **2026-07-11** (`fb8ad35`) — CLAUDE.md: session boot + governance pointer.
- **2026-07-24** (`d810583`) — Canonical pre-commit local-path guard added (Q-48 wave 1).
- **2026-07-27** (`701282c`, `68ecf1b`) — Keyless deterministic CI added (3-OS workflow); CI badge + coverage note.
- **2026-08-03** (`bf659bb`) — Publish-gate coverage canary added.
- **2026-08-04** (`3ac68bd`, `ff7c590`, `f3f1d0f`) — Allowlist migrated to entry-exact form; Apache-2.0 license added; Q-35 hook rollout.
- **2026-09-15** (`c83dc28`) — Canonical AGENTS.md router adopted (Q-93).
- **2026-09-19** (this commit) — Q-72(f): STATE.md added (this file); validator gains a STATE.md-existence check, the obsolete 5-record decision cap is removed, and the six-name BANNED_WITHOUT_TRIGGER list is propagated (live-file precondition checked, clear).

## Open loops

None on disk. This repo's README has no "## Status" heading — flagged here rather than papered over; ai-reliability-engine, ai-impact-scoring-engine, and ai-execution-engine all have one.
