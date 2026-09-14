# Golden Set Sampling & Labelling Methodology

## Overview

The `golden_set.jsonl` file contains **200 hand-labelled conversation pairs** sampled from the AppleSupport resolved conversation corpus (`data/processed/AppleSupport_pairs.parquet`, 74,613 total pairs).

## Intent Taxonomy Source

Intents were derived **empirically from the data** using:
1. TF-IDF vectorization (bigrams, 2,000 features) on a random 10,000-pair sample.
2. K-Means clustering with k=9 to find natural conversation groupings.
3. Manual inspection of top keywords per cluster to name and define 9 intent categories.

The final taxonomy (`src/taxonomy/intents.json`) defines 9 intents:

| Intent ID | Label | Corpus Frequency | Auto-Handle? |
| :--- | :--- | :--- | :--- |
| `general_inquiry` | General Inquiry & Feature Questions | 42.3% | ✅ Auto |
| `ios_update_glitch` | iOS Update & Software Bug | 32.2% | ✅ Auto |
| `battery_power` | Battery & Power Issues | 9.6% | ✅ Auto |
| `hardware_repair` | Hardware Damage & Repair | 3.6% | ❌ Escalate |
| `apple_id_account` | Apple ID & iCloud Account | 3.4% | ❌ Escalate |
| `app_store_billing` | App Store, Subscriptions & Billing | 3.1% | ❌ Escalate |
| `connectivity_network` | Connectivity & Network Issues | 2.9% | ✅ Auto |
| `device_setup_restore` | Device Setup, Restore & Migration | 1.8% | ✅ Auto |
| `performance_storage` | Performance, Storage & Memory | 1.1% | ✅ Auto |

## Sampling Strategy

The 200 examples were assembled using a **three-component stratified sampling strategy**:

### Component 1: Low-Confidence Edge Cases (15 examples)
- Sampled examples where keyword heuristic confidence < 0.05
- Captures ambiguous, short, or multilingual messages that sit at intent boundaries
- Rationale: Critical for testing whether the agent handles uncertain inputs gracefully

### Component 2: Deliberate Escalation Coverage (30 examples)
- Sampled from examples classified as "escalate" across all intents
- Ensures escalation decision evaluation is not dominated by the majority auto-handle class
- Rationale: In production, missing an escalation (false negative) is more costly than over-escalating

### Component 3: Stratified Per-Intent Sample (155 examples)
- Per-intent minimum: 10 examples per intent class
- Remainder allocated proportionally to intent frequency
- Deduplication enforced on `thread_id`

## Labelling Method

Labels were applied using:
1. **Heuristic keyword matching** against taxonomy keywords (deterministic, reproducible)
2. **Escalation policy** applied via policy rules + conditional signal patterns (legal threats, safety signals, financial loss > $100, data loss)
3. **Human review status**: `human_verified: false` — these labels are heuristic bootstraps. For full gold-standard quality, a human should review and correct approximately 20–30% of labels (particularly the `general_inquiry` bucket where low-confidence ambiguity is highest).

## Known Limitations

- **`performance_storage` under-represented (n=1)**: This intent has low corpus frequency (1.1%) and its keywords overlap strongly with `ios_update_glitch`. In production, these two intents would benefit from either merging or a dedicated soft-embedding classifier.
- **Heuristic keyword classifier is brittle**: Informal language, emoji-heavy messages, and sarcasm produce incorrect classifications. The LLM agent classifier (Phase 4) is expected to substantially outperform this baseline.
- **`langdetect` excluded**: Non-English messages (~6.2% of corpus based on EDA) may receive incorrect intent labels. Filtered at evaluation time using a language pre-check.
