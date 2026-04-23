"""
AI Decision Engine — main pipeline runner.
Processes a batch of leads from a JSON file.

Run:  python main.py
      python main.py --input data/sample_input.json --output data/results.json
"""

import json
import time
import argparse
from pathlib import Path

from pipeline.input_handler import load_inputs
from pipeline import ai_processor, validator, router
from models.schemas import DecisionResult, FallbackAction, FinalDecision, AIOutput
from config.settings import config
from utils.logger import logger
from database.db import init_db, save_decision, generate_run_id


def run_pipeline(input_path: str, output_path: str | None = None) -> list[dict]:
    logger.section("AI DECISION ENGINE — START")

    for k, v in config.summary().items():
        logger.info(f"  {k}: {v}")

    init_db()
    run_id = generate_run_id()
    logger.info(f"Run ID: {run_id}")

    records = load_inputs(input_path)
    results = []

    for record in records:
        logger.section(f"Processing: {record.id}")
        t_start = time.time()

        ai_output         = ai_processor.process_record(record)
        validation_result = validator.validate(ai_output, record.id)

        fallback_action = FallbackAction.NONE

        # Minimal fallback — assign safe default on validation failure
        # (P2 does not retry; emphasis is on decision tracking, not failure handling)
        if not validation_result.valid:
            logger.warning(f"[{record.id}] Validation failed — assigning safe default")
            ai_output = AIOutput(
                category="unknown",
                confidence=0.0,
                reason="Validation failed — safe default assigned."
            )
            fallback_action = FallbackAction.MANUAL_REVIEW_FLAGGED
            validation_result = validator.validate(ai_output, record.id)

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
            f"[{record.id}] → {final_decision.value} | "
            f"category={ai_output.category} | conf={ai_output.confidence:.2f} | {processing_ms}ms"
        )

    _print_summary(results, run_id)

    if output_path:
        _write_output(results, output_path)

    return results


def _print_summary(results: list[dict], run_id: str):
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
    logger.success(f"Persisted → {config.DB_PATH}")
    logger.info("Next step: POST outcomes to /outcome — then GET /stats to evaluate decision quality")


def _write_output(results: list[dict], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Results written → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Decision Engine")
    parser.add_argument("--input",  default="data/sample_input.json")
    parser.add_argument("--output", default="data/results.json")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
