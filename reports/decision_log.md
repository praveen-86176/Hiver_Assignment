# Architecture Decision Log

All significant technical decisions are recorded here with rationale, alternatives considered, and the date.

---

## Decision 001 — Brand Selection: @AppleSupport

**Date**: Phase 1  
**Decision**: Use @AppleSupport as the target brand.  
**Rationale**:
- Highest resolved pair count: 74,613 (vs 71,208 for AmazonHelp)
- Highest English purity: 93.8% (reduces language filtering noise)
- Technical diversity: iOS bugs, hardware, account security, billing — non-trivial intent taxonomy
- Real-world signal: AppleSupport is a known gold-standard for Twitter customer support tone

**Alternatives rejected**: AmazonHelp (shipping/orders too narrow), Spotify (lower volume), Delta (too domain-specific).

---

## Decision 002 — Model: qwen/qwen3.8-27b via Groq

**Date**: Phase 0  
**Decision**: Use `qwen/qwen3.8-27b` for both classifier and reply drafter via Groq API.  
**Rationale**:
- Free-tier Groq access available during development; fast inference (~0.8s p50)
- Strong structured JSON output reliability for classification
- Equivalent capability to Claude Haiku at zero incremental cost for prototype
- Groq's LPU inference makes it faster than self-hosted alternatives

**Alternatives rejected**: Claude Haiku 4.5 (paid, no free tier), OpenAI GPT-3.5 (cost), local llama (no GPU available).

---

## Decision 003 — TF-IDF Retrieval (not Dense Embeddings)

**Date**: Phase 3  
**Decision**: Use TF-IDF + cosine similarity for retrieval, not sentence-transformers.  
**Rationale**:
- Builds in <5 seconds on 74k pairs; no GPU or model download required
- Deterministic and fully reproducible
- Technical support queries are highly keyword-driven (model names, error messages, iOS versions) — TF-IDF excels in this regime
- Extension path to dense embeddings is documented inline in `retriever.py`

**Alternatives rejected**: sentence-transformers (requires model download, slower on CPU), OpenAI embeddings (API cost, network dependency).

---

## Decision 004 — Deterministic Policy Routing (No LLM)

**Date**: Phase 4  
**Decision**: Routing is purely deterministic (intent-level policy + regex signal patterns). No LLM involved in routing.  
**Rationale**:
- Eliminates ~1 full LLM call per message for the majority of traffic
- Policy rules are auditable, interpretable, and debuggable — critical for a production support system where legal/compliance requirements exist
- LLM judgment for routing introduces non-determinism: the same message might be routed differently on re-run
- Escalation errors (false negatives) are expensive; hard rules prevent hallucinated escalation confidence

**Tradeoff**: Subtle signals (passive aggression, implicit frustration) that don't match regex patterns will not trigger escalation. Future work: add an optional LLM escalation check for borderline-confidence classifications.

---

## Decision 005 — Golden Set: Heuristic Labels (not Manual)

**Date**: Phase 2  
**Decision**: Labels in `golden_set.jsonl` are from keyword-heuristic classification, not hand-labelled by a human.  
**Rationale**:
- Full manual labelling of 200 examples takes ~3 hours; scope of this assignment prioritises building the pipeline
- Labels are clearly marked as `human_verified: false`
- The heuristic labeller is deterministic and reproducible — evaluator can inspect and override any label
- All labelling logic is documented in `data/eval/golden_set_sampling_method.md`

**Known limitation**: The agent LLM classifier may legitimately disagree with the heuristic labeller in ambiguous cases — this inflates "intent accuracy" errors. For final production evaluation, 20–30% of labels should be human-verified.

---

## Decision 006 — ROUGE-1 as Proxy, Not Primary Metric

**Date**: Phase 6  
**Decision**: ROUGE-1 recall is computed but treated as a secondary/sanity metric.  
**Rationale**:
- The task is to draft an original grounded reply, not reproduce the reference reply verbatim
- A high-quality novel reply would score poorly on ROUGE against the specific reference tweet
- ROUGE is kept as a signal for vocabulary alignment / topical relevance, not quality
- Primary quality metric: LLM-as-judge (4-dimensional rubric, 1–5 scale)
