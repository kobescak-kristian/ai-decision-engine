"""
AI Processor — AI Decision Engine.
Classifies leads using OpenAI. Falls back to simulation mode if no API key.

Simulation data uses p2_001–p2_050 IDs.
Dataset is outcome-focused: realistic scenarios that produce interesting
evaluation results when outcomes are fed back in.
"""

import json
from models.schemas import AIOutput, InputRecord
from config.settings import config
from utils.logger import logger

SYSTEM_PROMPT = """You are a lead qualification engine for a B2B SaaS company.

Analyse the lead text and return ONLY a valid JSON object with exactly these fields:
{
  "category": "high_value" | "low_value" | "unknown",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explanation, max 20 words>"
}

Rules:
- high_value: clear business need, budget signals, decision-maker involvement, urgency
- low_value: student, vague, no budget, no urgency, general curiosity, no company
- unknown: ambiguous, insufficient info, or cannot determine intent

Return ONLY the JSON object. No markdown. No explanation."""


# ── Simulation data ───────────────────────────────────────────────────────
# These responses are used when OPENAI_API_KEY is not set.
# Each lead's simulated AI output is designed to produce realistic
# outcomes when outcome data is fed back — including intentional wrong
# decisions to demonstrate the evaluation engine flagging issues.

SIMULATED = {

    # ── Clear high value → will convert ──────────────────────────────────
    "p2_001": {"category": "high_value", "confidence": 0.96, "reason": "CFO confirmed 60k EUR budget, CTO attending, go-live in 8 weeks."},
    "p2_002": {"category": "high_value", "confidence": 0.94, "reason": "500-staff logistics firm, board mandate for automation, CEO contact."},
    "p2_003": {"category": "high_value", "confidence": 0.97, "reason": "Fortune 500 with active RFP, procurement engaged, 120k EUR budget."},
    "p2_004": {"category": "high_value", "confidence": 0.91, "reason": "Repeat client expanding contract, satisfied, clear upgrade scope."},
    "p2_005": {"category": "high_value", "confidence": 0.93, "reason": "Series B startup, 15M EUR raised, Q2 budget approved, CTO-led."},
    "p2_006": {"category": "high_value", "confidence": 0.95, "reason": "PE-backed firm, partner sign-off, 80k EUR confirmed, immediate start."},
    "p2_007": {"category": "high_value", "confidence": 0.88, "reason": "Regional bank, compliance sign-off, 40k EUR pilot budget approved."},
    "p2_008": {"category": "high_value", "confidence": 0.90, "reason": "Healthcare group, 12 clinics, COO decision maker, compliance involved."},

    # ── High value → will NOT convert (false positives for evaluator) ─────
    "p2_009": {"category": "high_value", "confidence": 0.85, "reason": "Insurance company switching from competitor, 30-day decision timeline."},
    "p2_010": {"category": "high_value", "confidence": 0.82, "reason": "200-lawyer firm, managing partner involved, RFP issued, 3-month timeline."},
    "p2_011": {"category": "high_value", "confidence": 0.87, "reason": "E-commerce at 10k orders/day, technical team ready, integration need clear."},
    "p2_012": {"category": "high_value", "confidence": 0.84, "reason": "Global manufacturer shortlisted, VP Operations meeting scheduled."},

    # ── High value → delayed (system will flag as manual review needed) ───
    "p2_013": {"category": "high_value", "confidence": 0.89, "reason": "Pharma company, CIO and compliance on call, six-figure budget approved."},
    "p2_014": {"category": "high_value", "confidence": 0.86, "reason": "Telecoms operator, mapped integration points, 25k EUR pilot confirmed."},

    # ── Borderline high value → will convert despite borderline confidence ─
    "p2_015": {"category": "high_value", "confidence": 0.63, "reason": "Senior VP at large retailer, genuine need but slow follow-up."},
    "p2_016": {"category": "high_value", "confidence": 0.61, "reason": "Operations director at 500-staff company, budget pending approval."},
    "p2_017": {"category": "high_value", "confidence": 0.58, "reason": "Enterprise demo booked but minimal context, unclear contact seniority."},

    # ── Borderline → will NOT convert ─────────────────────────────────────
    "p2_018": {"category": "high_value", "confidence": 0.55, "reason": "Mid-size company manager interested, vague on budget and timeline."},
    "p2_019": {"category": "high_value", "confidence": 0.52, "reason": "IT contact at 500-employee company, unclear purchasing authority."},

    # ── Clear low value → correctly ignored ──────────────────────────────
    "p2_020": {"category": "low_value", "confidence": 0.94, "reason": "Student dissertation research, no commercial intent or business context."},
    "p2_021": {"category": "low_value", "confidence": 0.91, "reason": "Anonymous pricing query, no contact info, no company, no stated need."},
    "p2_022": {"category": "low_value", "confidence": 0.88, "reason": "3-person startup, sub-500 EUR budget, no timeline, no defined use case."},
    "p2_023": {"category": "low_value", "confidence": 0.93, "reason": "Solo freelancer, no company, no team, no budget."},
    "p2_024": {"category": "low_value", "confidence": 0.87, "reason": "High school teacher, not a business inquiry, no commercial context."},
    "p2_025": {"category": "low_value", "confidence": 0.89, "reason": "Blog visitor asking definitional questions, no purchase signal."},
    "p2_026": {"category": "low_value", "confidence": 0.92, "reason": "Intern doing competitive research, no purchasing authority."},
    "p2_027": {"category": "low_value", "confidence": 0.85, "reason": "NGO volunteer, no budget, exploratory only, no decision maker."},

    # ── Low value → will convert (missed opportunity, evaluator flags) ────
    "p2_028": {"category": "low_value", "confidence": 0.72, "reason": "Small company inquiry, limited context, no clear budget signals."},
    "p2_029": {"category": "low_value", "confidence": 0.68, "reason": "Vague inquiry, unclear company size, minimal contact information."},

    # ── Unknown / ambiguous ───────────────────────────────────────────────
    "p2_030": {"category": "unknown", "confidence": 0.42, "reason": "Decision maker present but budget frozen until Q3, unclear timeline."},
    "p2_031": {"category": "unknown", "confidence": 0.38, "reason": "Partner reseller inquiry, indirect revenue, no end-user budget confirmed."},
    "p2_032": {"category": "unknown", "confidence": 0.45, "reason": "Consultant won't disclose client, budget and timeline completely unknown."},
    "p2_033": {"category": "unknown", "confidence": 0.28, "reason": "Technical questions only, no business need stated, possible competitor."},
    "p2_034": {"category": "unknown", "confidence": 0.35, "reason": "Press inquiry for editorial purposes, no purchase intent."},

    # ── Wrong segment (AI routes correctly, but outcome = wrong_segment) ──
    "p2_035": {"category": "high_value", "confidence": 0.81, "reason": "Mid-market company, budget present, but product may not fit their scale."},
    "p2_036": {"category": "high_value", "confidence": 0.79, "reason": "Large enterprise with automation interest, platform fit unclear."},

    # ── Correctly routed → manual review converts later ───────────────────
    "p2_037": {"category": "unknown", "confidence": 0.44, "reason": "Mixed signals: small team but clear ROI understanding and urgency."},
    "p2_038": {"category": "high_value", "confidence": 0.57, "reason": "IT manager with strong use case but budget still needs sign-off."},

    # ── Additional high value (varied industries) ─────────────────────────
    "p2_039": {"category": "high_value", "confidence": 0.92, "reason": "German consultancy, 150 staff, CEO decision maker, budget confirmed."},
    "p2_040": {"category": "high_value", "confidence": 0.90, "reason": "Property management firm, 200 properties, efficiency mandate from board."},
    "p2_041": {"category": "high_value", "confidence": 0.88, "reason": "Legal tech startup, Series A, clear integration need, CTO engaged."},
    "p2_042": {"category": "high_value", "confidence": 0.86, "reason": "Retail chain, 50 locations, centralised ops initiative, CFO sponsoring."},

    # ── Additional low value (varied) ────────────────────────────────────
    "p2_043": {"category": "low_value", "confidence": 0.90, "reason": "Recent graduate exploring options, no company or business context."},
    "p2_044": {"category": "low_value", "confidence": 0.86, "reason": "Asking only about free tier, no stated business purpose or company."},
    "p2_045": {"category": "low_value", "confidence": 0.83, "reason": "Retired professional, personal curiosity, no commercial application."},

    # ── Mixed outcome batch (for complete evaluation demo) ────────────────
    "p2_046": {"category": "high_value", "confidence": 0.94, "reason": "Manufacturing firm, 300 staff, automation roadmap approved, CIO contact."},
    "p2_047": {"category": "high_value", "confidence": 0.91, "reason": "Financial services company, compliance-driven need, 50k EUR allocated."},
    "p2_048": {"category": "low_value",  "confidence": 0.88, "reason": "University researcher, academic context only, no commercial pathway."},
    "p2_049": {"category": "unknown",    "confidence": 0.40, "reason": "Startup pre-seed, idea stage, no product-market fit confirmed yet."},
    "p2_050": {"category": "high_value", "confidence": 0.96, "reason": "Global 25k-staff enterprise, multi-year contract, CIO and CFO sponsors."},
}


def call_openai(record: InputRecord, clean_text: str) -> dict | None:
    if not config.OPENAI_API_KEY:
        logger.debug(f"Simulation mode for {record.id}")
        return SIMULATED.get(record.id)

    try:
        import openai
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this lead:\n\n{clean_text}"}
            ],
            temperature=0.1,
            max_tokens=150
        )
        raw = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response for {record.id}: {raw}")
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"Non-JSON response for {record.id}: {e}")
        return None
    except Exception as e:
        logger.error(f"OpenAI call failed for {record.id}: {e}")
        return None


def process_record(record: InputRecord) -> AIOutput | None:
    raw = call_openai(record, record.raw_text)
    if raw is None:
        logger.warning(f"[{record.id}] No AI output returned")
        return None

    try:
        return AIOutput(**raw)
    except Exception as e:
        logger.error(f"Could not parse AI output for {record.id}: {e}")
        return None
