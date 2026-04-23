"""
Decision Evaluation Engine — NEW in P2.
Computes performance metrics from decisions + outcomes.
Flags issues when the system is performing below acceptable thresholds.
"""

from database.db import get_evaluation_data
from config.settings import config
from utils.logger import logger


def compute_metrics() -> dict:
    """
    Compute decision quality metrics from the database.
    Returns metrics, insights, and decision distribution.
    """
    data = get_evaluation_data()

    total_decisions = data["total_decisions"]
    total_outcomes  = data["total_outcomes"]
    by_decision     = data["by_decision"]
    false_positives = data["false_positives"]
    missed          = data["missed_opportunities"]

    # ── Core counts ───────────────────────────────────────────────────────
    sent_to_sales  = by_decision.get("send_to_sales", 0)
    archived       = by_decision.get("archive", 0)
    manual_review  = by_decision.get("manual_review", 0)

    # ── Conversion rate ───────────────────────────────────────────────────
    # Of leads sent to sales, how many actually converted?
    converted_from_sales = _count_outcome_for_decision(
        data["outcome_by_decision"], "send_to_sales", "converted"
    )
    conversion_rate = (
        round(converted_from_sales / sent_to_sales, 3)
        if sent_to_sales > 0 else None
    )

    # ── False positive rate ───────────────────────────────────────────────
    # Of leads sent to sales, how many did NOT convert?
    false_positive_rate = (
        round(false_positives / sent_to_sales, 3)
        if sent_to_sales > 0 else None
    )

    # ── Manual review rate ────────────────────────────────────────────────
    manual_review_rate = (
        round(manual_review / total_decisions, 3)
        if total_decisions > 0 else None
    )

    # ── Outcome coverage ──────────────────────────────────────────────────
    outcome_coverage = (
        round(total_outcomes / total_decisions, 3)
        if total_decisions > 0 else 0
    )

    metrics = {
        "total_leads":          total_decisions,
        "total_outcomes":       total_outcomes,
        "outcome_coverage":     outcome_coverage,
        "sent_to_sales":        sent_to_sales,
        "archived":             archived,
        "manual_review":        manual_review,
        "converted":            converted_from_sales,
        "conversion_rate":      conversion_rate,
        "false_positives":      false_positives,
        "false_positive_rate":  false_positive_rate,
        "missed_opportunities": missed,
        "manual_review_rate":   manual_review_rate,
    }

    insights = _generate_insights(metrics)

    return {
        "metrics":              metrics,
        "insights":             insights,
        "decision_distribution": by_decision,
        "outcome_breakdown":    data["outcome_by_decision"],
    }


def _count_outcome_for_decision(outcome_by_decision: list, decision: str, outcome: str) -> int:
    for row in outcome_by_decision:
        if row["final_decision"] == decision and row["outcome"] == outcome:
            return row["count"]
    return 0


def _generate_insights(metrics: dict) -> dict:
    """
    Apply simple rules to flag issues with decision quality.
    Returns a status ('ok' | 'warning' | 'critical') and a list of issues.
    """
    issues = []
    status = "ok"

    conversion_rate    = metrics.get("conversion_rate")
    false_positive_rate = metrics.get("false_positive_rate")
    manual_review_rate  = metrics.get("manual_review_rate")
    missed             = metrics.get("missed_opportunities", 0)
    coverage           = metrics.get("outcome_coverage", 0)

    # Not enough outcome data to evaluate yet
    if coverage < 0.3:
        return {
            "status": "insufficient_data",
            "issues": [f"Only {round(coverage * 100, 1)}% of decisions have outcomes. Need at least 30% for reliable evaluation."]
        }

    # Conversion rate check
    if conversion_rate is not None:
        if conversion_rate < config.MIN_CONVERSION_RATE:
            issues.append(
                f"Low conversion rate: {round(conversion_rate * 100, 1)}% "
                f"(threshold: {round(config.MIN_CONVERSION_RATE * 100, 1)}%). "
                f"Too many low-quality leads being sent to sales."
            )
            status = "warning"

    # False positive check
    if false_positive_rate is not None:
        if false_positive_rate > config.MAX_FALSE_POSITIVE_RATE:
            issues.append(
                f"High false positive rate: {round(false_positive_rate * 100, 1)}% "
                f"(threshold: {round(config.MAX_FALSE_POSITIVE_RATE * 100, 1)}%). "
                f"AI is over-classifying leads as high value."
            )
            status = "warning" if status == "ok" else "critical"

    # Manual review rate check
    if manual_review_rate is not None:
        if manual_review_rate > config.MAX_MANUAL_REVIEW_RATE:
            issues.append(
                f"High manual review rate: {round(manual_review_rate * 100, 1)}% "
                f"(threshold: {round(config.MAX_MANUAL_REVIEW_RATE * 100, 1)}%). "
                f"Too many leads requiring human intervention — system inefficiency."
            )
            status = "warning" if status == "ok" else status

    # Missed opportunities check
    if missed > 0:
        issues.append(
            f"{missed} missed opportunity/ies detected: leads archived or sent to manual review that later converted. "
            f"Consider lowering the confidence threshold."
        )
        status = "warning" if status == "ok" else status

    if not issues:
        issues.append("All metrics within acceptable thresholds.")

    logger.info(f"Evaluation complete — status: {status}, issues: {len(issues)}")

    return {
        "status": status,
        "issues": issues
    }
