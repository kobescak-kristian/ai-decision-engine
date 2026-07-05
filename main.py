"""
AI Decision Engine — main pipeline runner.
Processes a batch of leads from a JSON file.

Run:  python main.py
      python main.py --input data/sample_input.json --output data/results.json
"""

import sys
import json
import time
import argparse
from pathlib import Path

from pipeline.input_handler import load_inputs
from pipeline import ai_processor, validator, router
from pipeline.ai_processor import AIAuthError, AIConnectionError
from models.schemas import DecisionResult, FallbackAction, FinalDecision, AIOutput
from config.settings import config
from utils.logger import logger
from database.db import init_db, save_decision, generate_run_id

# Defense in depth: some AI-layer failures (rate limits, 5xx, provider
# outages) don't raise AIAuthError/AIConnectionError but still recur
# identically across many leads — that pattern is systemic, not per-lead.
SYSTEMIC_FAILURE_THRESHOLD = 3


class SystemicFailureError(RuntimeError):
    """N consecutive leads failed with the same AI-layer reason."""


def _abort(exc: Exception, records_committed: int):
    logger.section("RUN ABORTED")
    logger.error(f"FATAL: {type(exc).__name__}: {exc}")
    logger.error(f"{records_committed} record(s) were processed and persisted before the abort.")
    sys.exit(1)


def _print_mode_banner():
    if config.simulation_mode():
        logger.section("MODE: SIMULATION")
        logger.warning(f"Reason: {config.simulation_reason()}")
        logger.warning("AI outputs come from the built-in simulated dataset - no API calls are made.")
    else:
        logger.section(f"MODE: LIVE ({config.OPENAI_MODEL})")
        logger.info("A real OpenAI key is configured - leads are classified by the live model.")


def run_pipeline(input_path: str, output_path: str | None = None) -> list[dict]:
    logger.section("AI DECISION ENGINE - START")
    _print_mode_banner()

    for k, v in config.summary().items():
        logger.info(f"  {k}: {v}")

    init_db()
    run_id = generate_run_id()
    logger.info(f"Run ID: {run_id}")

    records = load_inputs(input_path)
    results = []
    fallbacks = []   # (record_id, fallback reason) — reported in summary
    consecutive_ai_failures = 0
    last_ai_failure_reason  = None

    for record in records:
        logger.section(f"Processing: {record.id}")
        t_start = time.time()

        try:
            ai_output, ai_failure_reason = ai_processor.process_record(record)
        except (AIAuthError, AIConnectionError) as e:
            _abort(e, len(results))

        validation_result = validator.validate(ai_output, record.id)

        fallback_action = FallbackAction.NONE

        # Minimal fallback — assign safe default on validation failure
        # (no retry; emphasis is on decision tracking, not failure handling)
        if not validation_result.valid:
            fallback_reason = ai_failure_reason or "; ".join(validation_result.errors)
            logger.warning(f"[{record.id}] Validation failed - assigning safe default ({fallback_reason})")
            fallbacks.append((record.id, fallback_reason))

            if ai_failure_reason:
                if ai_failure_reason == last_ai_failure_reason:
                    consecutive_ai_failures += 1
                else:
                    last_ai_failure_reason  = ai_failure_reason
                    consecutive_ai_failures = 1

                if consecutive_ai_failures >= SYSTEMIC_FAILURE_THRESHOLD:
                    _abort(
                        SystemicFailureError(
                            f"{consecutive_ai_failures} consecutive leads failed with the same "
                            f"AI-layer reason (not a typed auth/connection error, but the pattern "
                            f"is systemic): {ai_failure_reason}"
                        ),
                        len(results)
                    )
            else:
                consecutive_ai_failures = 0
                last_ai_failure_reason  = None

            ai_output = AIOutput(
                category="unknown",
                confidence=0.0,
                reason=(
                    f"Validation failed — safe default assigned. Cause: {ai_failure_reason}"
                    if ai_failure_reason else
                    "Validation failed — safe default assigned."
                )
            )
            fallback_action = FallbackAction.MANUAL_REVIEW_FLAGGED
            validation_result = validator.validate(ai_output, record.id)
        else:
            consecutive_ai_failures = 0
            last_ai_failure_reason  = None

        final_decision = router.route(ai_output, fallback_action, record.id)
        processing_ms  = round((time.time() - t_start) * 1000, 2)

        result = DecisionResult(
            input=record,
            ai_output=ai_output,
            validation=validation_result,
            fallback_action=fallback_action,
            final_decision=final_decision,
            processing_ms=processing_ms
        )

        result_dict = result.model_dump()
        results.append(result_dict)
        save_decision(result_dict, run_id)

        logger.info(
            f"[{record.id}] -> {final_decision.value} | "
            f"category={ai_output.category} | conf={ai_output.confidence:.2f} | {processing_ms}ms"
        )

    _print_summary(results, run_id, fallbacks)

    if output_path:
        _write_output(results, output_path)

    return results


def _print_summary(results: list[dict], run_id: str, fallbacks: list[tuple[str, str]]):
    logger.section("PIPELINE SUMMARY")
    decisions = {}
    total_ms  = 0

    for r in results:
        d = r["final_decision"]
        decisions[d] = decisions.get(d, 0) + 1
        if r.get("processing_ms"):
            total_ms += r["processing_ms"]

    avg_ms = round(total_ms / len(results), 2) if results else 0

    logger.info(f"Run ID        : {run_id}")
    logger.info(f"Total records : {len(results)}")
    for decision, count in sorted(decisions.items()):
        logger.info(f"  {decision}: {count}")
    logger.info(f"Avg time      : {avg_ms}ms per record")

    if fallbacks:
        logger.warning(f"Fallbacks     : {len(fallbacks)} record(s) degraded to safe default (unknown/manual_review)")
        for record_id, errors in fallbacks:
            logger.warning(f"  {record_id}: {errors}")
    else:
        logger.info("Fallbacks     : 0")

    logger.success(f"Persisted -> {config.DB_PATH}")
    logger.info("Next step: POST outcomes to /outcome - then GET /stats to evaluate decision quality")


def _write_output(results: list[dict], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Results written -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Decision Engine")
    parser.add_argument("--input",  default="data/sample_input.json")
    parser.add_argument("--output", default="data/results.json")
    args = parser.parse_args()
    # Abort (non-zero exit) is handled inline in run_pipeline via _abort(),
    # so that the FATAL message can report how many records were already
    # committed before the failure was detected.
    run_pipeline(args.input, args.output)
