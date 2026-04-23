from models.schemas import AIOutput, FinalDecision, FallbackAction
from config.settings import config
from utils.logger import logger


def route(ai_output: AIOutput, fallback_action: FallbackAction, record_id: str) -> FinalDecision:
    """
    Route validated AI output to a final business decision.

    Logic:
    - Fallback flagged              → manual_review (always)
    - high_value + conf >= threshold → send_to_sales
    - high_value + conf < threshold  → manual_review
    - low_value                      → archive
    - unknown                        → manual_review
    """
    threshold = config.CONFIDENCE_THRESHOLD

    if fallback_action == FallbackAction.MANUAL_REVIEW_FLAGGED:
        logger.info(f"[{record_id}] Route: MANUAL_REVIEW (fallback flagged)")
        return FinalDecision.MANUAL_REVIEW

    cat  = ai_output.category
    conf = ai_output.confidence

    if cat == "high_value":
        if conf >= threshold:
            logger.success(f"[{record_id}] Route: SEND_TO_SALES (conf={conf:.2f})")
            return FinalDecision.SEND_TO_SALES
        else:
            logger.warning(f"[{record_id}] Route: MANUAL_REVIEW — high_value conf={conf:.2f} below threshold")
            return FinalDecision.MANUAL_REVIEW

    elif cat == "low_value":
        logger.info(f"[{record_id}] Route: ARCHIVE")
        return FinalDecision.ARCHIVE

    else:
        logger.info(f"[{record_id}] Route: MANUAL_REVIEW (category={cat})")
        return FinalDecision.MANUAL_REVIEW
