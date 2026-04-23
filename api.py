"""
FastAPI wrapper — AI Decision Engine

Endpoints:
  POST /qualify          — Process a lead, make a decision
  POST /outcome          — Record what actually happened after the decision
  GET  /stats            — Decision quality metrics + insights
  GET  /audit            — Recent decisions with outcome status
  GET  /health           — Health check

Run: uvicorn api:app --reload --port 8000
"""

import time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from models.schemas import (
    InputRecord, FallbackAction, FinalDecision,
    AIOutput, OutcomeRequest, OutcomeType
)
from pipeline import ai_processor, validator, router
from pipeline.outcome_handler import record_outcome, OutcomeError
from pipeline.evaluator import compute_metrics
from config.settings import config
from utils.logger import logger
from database.db import (
    init_db, save_decision, get_recent_decisions,
    test_connection, generate_run_id
)

init_db()

app = FastAPI(
    title="AI Decision Engine",
    description=(
        "Evaluates whether AI-driven lead decisions are actually effective. "
        "Decisions are made, outcomes are tracked, and performance is measured over time."
    ),
    version="1.0.0"
)


# ── Request / Response models ─────────────────────────────────────────────

class LeadRequest(BaseModel):
    id: str
    raw_text: str
    metadata: Optional[dict] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "p2_demo_01",
                "raw_text": "CFO confirmed 40k EUR budget. CTO and procurement involved. Go-live in 8 weeks.",
                "metadata": {"source": "web_form", "region": "EU"}
            }
        }
    }


class LeadResponse(BaseModel):
    id: str
    category: str
    confidence: float
    reason: str
    final_decision: str
    fallback_action: str
    processing_ms: float


class OutcomeResponse(BaseModel):
    lead_id: str
    decision: str
    outcome: str
    timestamp: str


# ── Core processing ───────────────────────────────────────────────────────

def process_lead(lead: LeadRequest, run_id: str) -> LeadResponse:
    t_start = time.time()
    record  = InputRecord(id=lead.id, raw_text=lead.raw_text, metadata=lead.metadata)

    ai_output         = ai_processor.process_record(record)
    validation_result = validator.validate(ai_output, record.id)

    fallback_action = FallbackAction.NONE

    if not validation_result.valid:
        ai_output = AIOutput(
            category="unknown",
            confidence=0.0,
            reason="Validation failed — safe default assigned."
        )
        fallback_action = FallbackAction.MANUAL_REVIEW_FLAGGED
        validation_result = validator.validate(ai_output, record.id)

    final_decision = router.route(ai_output, fallback_action, record.id)
    processing_ms  = round((time.time() - t_start) * 1000, 2)

    result_dict = {
        "input": {
            "id": record.id, "raw_text": record.raw_text,
            "metadata": record.metadata, "received_at": record.received_at
        },
        "ai_output": {
            "category":   ai_output.category,
            "confidence": ai_output.confidence,
            "reason":     ai_output.reason
        },
        "validation":     {"valid": validation_result.valid, "errors": validation_result.errors},
        "fallback_action": fallback_action.value,
        "final_decision":  final_decision.value,
        "processing_ms":   processing_ms
    }

    save_decision(result_dict, run_id)

    return LeadResponse(
        id=record.id,
        category=ai_output.category,
        confidence=ai_output.confidence,
        reason=ai_output.reason,
        final_decision=final_decision.value,
        fallback_action=fallback_action.value,
        processing_ms=processing_ms
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """System health check."""
    db_ok = test_connection()
    return {
        "status":              "ok" if db_ok else "degraded",
        "database":            "connected" if db_ok else "unreachable",
        "simulation_mode":     config.simulation_mode(),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "version":             "1.0.0"
    }


@app.post("/qualify", response_model=LeadResponse)
def qualify_lead(lead: LeadRequest):
    """
    Process a lead through the decision pipeline.
    Returns the AI classification and final routing decision.
    Feed the outcome back later via POST /outcome.
    """
    try:
        run_id = generate_run_id()
        logger.info(f"POST /qualify — {lead.id}")
        result = process_lead(lead, run_id)
        logger.success(f"POST /qualify — {lead.id} → {result.final_decision}")
        return result
    except Exception as e:
        logger.error(f"POST /qualify error for {lead.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/outcome", response_model=OutcomeResponse)
def submit_outcome(request: OutcomeRequest):
    """
    Record what actually happened after a lead decision was made.
    Outcomes: converted | not_converted | ignored | wrong_segment | delayed

    The lead must have been processed through /qualify first.
    This is how the feedback loop works — outcomes are stored against
    the original decision and feed the evaluation engine.
    """
    try:
        result = record_outcome(request.lead_id, request.outcome)
        return OutcomeResponse(**result)
    except OutcomeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"POST /outcome error for {request.lead_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """
    Decision quality metrics and insights.

    Returns:
    - Conversion rate (of leads sent to sales, how many converted)
    - False positive rate (sent to sales but did not convert)
    - Manual review rate (operational efficiency indicator)
    - Missed opportunities (archived/manual review leads that converted)
    - Insight flags when metrics fall outside acceptable thresholds
    """
    return compute_metrics()


@app.get("/audit")
def audit(limit: int = Query(default=20, le=100)):
    """
    Recent decisions with their outcome status.
    Shows which decisions have received outcome feedback.
    """
    records = get_recent_decisions(limit=limit)
    return {"count": len(records), "records": records}
