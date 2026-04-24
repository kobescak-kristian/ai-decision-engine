# AI Decision Engine — v1.0

Most AI systems make decisions.  
Few systems know if those decisions were actually correct.

This system closes that gap.

**Type:** System design / AI evaluation pipeline  
**Use case:** AI-assisted decision-making (lead routing)  
**Core concept:** Decision → Outcome → Evaluation loop  

---

## What this solves

Most AI lead-routing systems stop at classification.

They decide where a lead should go, but they do not measure whether that decision was actually right.

This creates silent failure: sales teams receive unqualified leads, real buyers are missed, and no system exists to detect or correct it.

This system records the original decision, accepts the real-world outcome later, and evaluates decision quality over time.

---

## Outcome

Using a simulated 50-lead dataset with outcome feedback:

- 50 leads processed  
- 25 routed to sales  
- 14 archived  
- 12 sent to manual review  
- 32 outcomes recorded  
- 64% outcome coverage  
- 72% conversion rate on sent-to-sales leads  
- 32% false positive rate → roughly 1 in 3 leads sent to sales should not have been there  
- 4 missed opportunities detected → leads that would likely have been lost without feedback analysis  
- evaluator returned `warning` status with actionable issues  

This demonstrates a full feedback loop from AI decision to measured business result.

---

## Architecture

![Architecture](architecture.png)

Decisions and outcomes are stored separately and evaluated asynchronously, allowing real-world feedback to be joined back to the original AI decision.

This system has two connected layers:

**1. Decision pipeline (real-time)**  
Input → AI → Validation → Fallback → Routing → Persist  

**2. Feedback loop (delayed)**  
Outcome → Evaluation → Insights → Threshold tuning  

Together, they measure whether AI decisions were actually correct.

---

## Why not just use rules?

Rules work on clean, predictable input.  
Lead qualification is not clean or predictable — it involves ambiguous intent, incomplete context, and uncertain value.

AI handles those cases.

This system adds the missing layer: measuring whether those AI-assisted decisions were actually right.

This is not just logging outcomes.  
It connects past decisions with future results and evaluates them under coverage constraints — something most AI pipelines never implement.

CRM reports show outcomes — not whether the AI decision was correct at the time it was made.

---

## Business value

| Component | What it prevents / enables |
|---|---|
| Validation | Blocks invalid AI output from becoming business decisions |
| Fallback | Prevents system failure on malformed AI responses |
| Routing | Converts AI output into clear operational actions |
| Persistence | Creates an audit trail for every decision |
| Outcome feedback | Records real-world results of each decision |
| Evaluation engine | Measures decision quality instead of guessing |
| Coverage gate | Prevents misleading metrics on insufficient data |
| Threshold tuning | Enables controlled optimisation of decision logic |

---

## Quick demo

### 1. Run batch pipeline

```bash
python main.py

---

## System Context

This project is part of a larger AI decision system:

- **[Reliability Engine](https://github.com/kobescak-kristian/ai-workflow-reliability-engine)** → prevents invalid AI outputs from entering workflows  
- Decision Engine → tracks outcomes and evaluates decision correctness  
- **[Impact Engine](https://github.com/kobescak-kristian/ai-impact-decision-intelligence-engine)** → measures financial impact and optimizes thresholds  

→ Complete system: validation → evaluation → financial impact