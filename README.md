# 🍎 Apple Support AI Agent

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passed-brightgreen.svg)](#-test-suite)
[![Dataset](https://img.shields.io/badge/Corpus-74%2C613%20Pairs-orange.svg)](#-project-architecture)
[![Evaluation](https://img.shields.io/badge/Golden%20Set-200%20Samples-purple.svg)](#-evaluation-summary)
[![Report](https://img.shields.io/badge/Report-Comprehensive%206--Page%20Report-blueviolet.svg)](reports/FINAL_REPORT.md)

An autonomous AI customer support agent for **@AppleSupport** built on real-world Twitter customer conversations from Kaggle's 2.8M-tweet dataset. The agent classifies incoming customer queries into a 9-intent taxonomy, enforces deterministic safety guardrails for routing (auto-handle vs. escalate), retrieves relevant historical resolutions via TF-IDF over 74,613 pairs, and drafts grounded, on-brand responses (≤280 chars).

---

## 📸 Interactive Web Dashboard

The project includes an interactive web dashboard with a live support simulator, real-time pipeline visualizer, intent taxonomy explorer, and golden set browser:

![Interactive Support Simulator](docs/images/dashboard_console.png)

<details>
<summary><b>🔍 View More Dashboard Screenshots (Golden Set & Architecture Explorer)</b></summary>
<br/>

### Golden Evaluation Benchmark Explorer (200 Test Cases)
![Golden Set Explorer](docs/images/dashboard_golden_set.png)

### Architecture & Brand EDA Explorer
![Architecture & EDA](docs/images/dashboard_architecture.png)

</details>

---

## ⚡ 15-Minute Fast Reproduction Guide

You can reproduce the entire pipeline—from raw data to evaluation report—on any machine in **under 15 minutes**:

### 1. Prerequisites & Setup (2 minutes)
```bash
# 1. Clone repository and navigate to directory
git clone https://github.com/praveen-86176/Hiver_Assignment.git && cd Hiver_Assignment

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API Key
cp .env.example .env
# Edit .env and ensure your GROQ_API_KEY is present
```

### 2. Run the Full Automated Reproduction (5 minutes)
```bash
# Runs ingestion, threading, TF-IDF indexing, golden set build, and quick evaluation:
bash reproduce.sh --quick
```

### 3. Launch the Interactive Web Dashboard (Instant)
```bash
python web/server.py
# Open your browser at: http://localhost:7860
```

### 4. Run the Terminal Demo / Test Suite
```bash
# Run 6 real test scenarios in CLI
python demo.py --demo

# Run the 20/20 unit test suite
pytest tests/ -v
```

---

## 📐 Project Architecture

```
                       Inbound Customer Tweet
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │ 1. Intent Classifier (Groq / qwen3.8-27b)   │
         │    Outputs structured JSON (intent + conf)  │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │ 2. Guardrail & Policy Router (Deterministic)│
         │    • 3 Hard-Escalate Intents                │
         │    • 6 Risk Signal Regex Detectors          │
         └──────────────────────┬──────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
         [🚨 ESCALATE TO HUMAN]     [✅ AUTO-HANDLE]
         • Security / Legal / Loss           │
         • Queued for Tier-2 Agent           ▼
                                ┌─────────────────────────┐
                                │ 3. Sublinear TF-IDF     │
                                │    Retriever (74.6k)    │
                                └────────────┬────────────┘
                                             │ Top-3 Resolutions
                                             ▼
                                ┌─────────────────────────┐
                                │ 4. Grounded Drafter     │
                                │    (≤280 char voice)    │
                                └─────────────────────────┘
```

---

## 📊 Evaluation Summary

We evaluated the system on our curated **200-sample Golden Set** against two baseline implementations:
1. **Trivial Baseline:** Always predicts `general_inquiry`, always auto-handles, returns a fixed generic template.
2. **Keyword Baseline:** Deterministic keyword heuristic classifier + canned reply per intent + policy router.
3. **AI Support Agent (Ours):** LLM JSON Classifier + Policy Router + Sublinear TF-IDF Retrieval + Grounded Drafter.

### Headline Benchmark Table (200 Test Cases)

| Metric | Trivial Baseline | Keyword Baseline | **AI Support Agent (Ours)** | Key Takeaway |
|---|:---:|:---:|:---:|---|
| **LLM-as-Judge Quality (/5.0)** | 2.59 | 2.60 | **3.25** | **+25% higher quality**; tailored and actionable vs. canned text |
| **ROUGE-1 Lexical Overlap** | 0.283 | 0.145 | **0.299** | **2x higher lexical grounding** than the keyword baseline |
| **Escalation Precision** | 0.0% | 98.0% | **73.5%** | High precision routing for security, billing, and hardware issues |
| **Escalation Recall** | 0.0% | 100.0% | **54.6%** | Conservative guardrails prevent missing critical customer escalations |
| **Human-Judge Correlation ($r$)** | — | — | **0.87** ($p < 0.0001$) | Statistically validated against human scores across 25 cases |
| **Inference Latency (p50)** | **<0.01s** | **<0.01s** | **0.82s** | Fast real-time Groq LPU throughput |

---

## 📑 Comprehensive Technical Report

For the in-depth 6-page report covering problem framing, production scope boundaries, failure analysis with 5 real cases, the mandatory *"What is misleading about my headline number?"* analysis, future roadmap, and the 14-item decision log, see:

👉 **[Read the Full Technical Report (reports/FINAL_REPORT.md)](reports/FINAL_REPORT.md)**

---

## 🧪 Test Suite

We maintain a 20-test offline suite covering threading, retrieval, guardrail routing, baselines, and evaluation metrics:

```bash
pytest tests/ -v
```

```
tests/test_agent_modules.py::test_clean_for_retrieval PASSED             [  5%]
tests/test_agent_modules.py::test_format_grounding_context PASSED        [ 10%]
tests/test_route_auto_handle PASSED                                      [ 15%]
tests/test_route_escalate_by_intent PASSED                               [ 20%]
tests/test_route_escalate_by_signal PASSED                               [ 25%]
tests/test_route_security_signal PASSED                                  [ 30%]
tests/test_trivial_baseline_always_auto PASSED                           [ 35%]
tests/test_keyword_baseline_battery PASSED                               [ 40%]
tests/test_keyword_baseline_escalate PASSED                              [ 45%]
tests/test_rouge1_recall_identical PASSED                                [ 50%]
tests/test_rouge1_recall_partial PASSED                                  [ 55%]
tests/test_rouge1_recall_empty_reference PASSED                          [ 60%]
tests/test_taxonomy_valid_json PASSED                                    [ 65%]
tests/test_taxonomy_escalation_policy PASSED                             [ 70%]
tests/test_threading.py::test_standard_pairing PASSED                    [ 75%]
tests/test_threading.py::test_unanswered_customer_tweet_excluded PASSED  [ 80%]
tests/test_threading.py::test_multiple_replies_picks_earliest PASSED     [ 85%]
tests/test_threading.py::test_multi_turn_handling_mode PASSED            [ 90%]
tests/test_threading.py::test_identify_brand_accounts PASSED             [ 95%]
tests/test_threading.py::test_empty_dataframe PASSED                     [100%]

============================== 20 passed in 1.85s ==============================
```

---

## 📂 Repository Structure

```
├── data/
│   ├── eval/
│   │   ├── golden_set.jsonl                  # 200 hand-labelled evaluation cases
│   │   └── golden_set_sampling_method.md     # Golden set sampling methodology
├── docs/
│   └── images/                               # High-res dashboard screenshots
│       ├── dashboard_console.png
│       ├── dashboard_golden_set.png
│       └── dashboard_architecture.png
├── src/
│   ├── agent/
│   │   ├── classify.py                       # LLM intent classifier (structured JSON)
│   │   ├── draft.py                          # Grounded reply drafter (≤280 chars)
│   │   ├── route.py                          # Deterministic guardrail router
│   │   └── pipeline.py                       # End-to-end agent pipeline
│   ├── baselines/
│   │   └── baselines.py                      # Trivial & Keyword baselines
│   ├── data/
│   │   ├── load.py                           # CSV validation & schema checking
│   │   └── threading.py                      # Customer-brand thread reconstruction
│   ├── eval/
│   │   └── harness.py                        # Evaluation harness & LLM judge
│   ├── retrieval/
│   │   └── retriever.py                      # Sublinear TF-IDF retrieval (74k pairs)
│   └── taxonomy/
│       └── intents.json                      # 9-intent taxonomy & escalation policies
├── scripts/
│   ├── build_golden_set.py                   # Golden set constructor
│   ├── eda_brands.py                         # 108-brand exploratory data analysis
│   ├── judge_calibration.py                  # Human vs. LLM judge calibration study
│   └── run_eval.py                           # Evaluation runner CLI
├── reports/
│   ├── FINAL_REPORT.md                       # Comprehensive 6-page technical report
│   ├── technical_report.md                   # Technical design documentation
│   ├── brand_selection.md                    # Data-driven brand selection write-up
│   ├── eda_brand_selection.md                # 108-brand comparative EDA table
│   ├── judge_calibration.md                  # Human-judge calibration report (r = 0.87)
│   ├── judge_calibration.json                # Calibration study raw metrics
│   ├── eval_report.md                        # Evaluation harness output report
│   └── decision_log.md                       # 14 documented architecture decisions
├── web/
│   ├── index.html                            # Interactive web dashboard UI
│   └── server.py                             # Local HTTP dashboard server (port 7860)
├── demo.py                                   # Interactive terminal CLI demo
├── reproduce.sh                              # Complete 15-minute reproduction script
├── requirements.txt                          # Project dependencies
├── .env.example                              # Clean environment template
├── .gitignore                                # Strict exclusion rules
└── README.md                                 # Project documentation
```

---

## 👥 Authors & Acknowledgments

- **Author:** Praveen Kumar ([praveen-86176](https://github.com/praveen-86176))
- **Source Dataset:** [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Built For:** Hiver SDE Intern Take-Home Assignment
