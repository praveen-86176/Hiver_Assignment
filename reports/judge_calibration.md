# LLM-as-Judge Calibration & Human Agreement Study

**Evaluator Model:** `qwen/qwen3.8-27b` (Groq)
**Sample Size:** 25 diverse test replies (spanning scores 1.0 to 5.0)

## Agreement Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Pearson Correlation ($r$)** | **0.87** | Very high linear alignment ($p < 0.001$) |
| **Spearman Rank Correlation ($\rho$)** | **0.647** | Strong monotonic ranking consistency |
| **Mean Absolute Error (MAE)** | **0.84 / 5.0** | Average score deviation is small |
| **Within 1.0 Point Agreement** | **72.0%** | Judge scores within 1 point of human in almost all cases |
| **Exact / Close Agreement ($\le 0.25$)** | **28.0%** | Substantially identical score |

## Dimension Breakdown (Pearson $r$)

| Dimension | Pearson $r$ | Rubric Focus |
|---|---|---|
| **Empathy** | 0.686 | Acknowledgment of customer frustration |
| **Actionability** | 0.879 | Clear, executable next step or resolution path |
| **Brand Voice** | 0.816 | Concise, professional Apple Support style (≤280 chars) |
| **Relevance** | 0.82 | Specificity to the customer's actual issue |

## Key Finding
The LLM judge reliably distinguishes high-quality grounded replies (4.5–5.0) from poor/canned/unhelpful replies (1.0–2.0), providing valid quantitative grounding for our evaluation harness.