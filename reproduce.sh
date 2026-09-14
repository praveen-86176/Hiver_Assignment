#!/usr/bin/env bash
# reproduce.sh — Full pipeline reproduction from scratch.
#
# Runs the complete pipeline in under 15 minutes on any machine with the raw data.
# Requirements: Python 3.11+, .venv/, .env with GROQ_API_KEY
#
# Usage:
#   bash reproduce.sh            # Full pipeline
#   bash reproduce.sh --quick    # Quick mode (20 golden set samples)

set -euo pipefail

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    echo "⚡ Running in QUICK mode (20 samples)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Apple Support AI Agent — Full Pipeline Reproduction   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 0. Environment check
echo "▶ Step 0: Checking environment..."
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Copy .env.example and set GROQ_API_KEY."
    exit 1
fi
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run: python -m venv .venv && pip install -r requirements.txt"
    exit 1
fi
source .venv/bin/activate
echo "   Python: $(python --version)"
echo "   Environment: OK"

# 1. Data ingestion + threading
echo ""
echo "▶ Step 1: Thread reconstruction..."
python -c "
from src.data.load import load_and_validate
from src.data.threading import reconstruct_threads, extract_brand_pairs
import pandas as pd
print('Loading raw data...')
df = load_and_validate('data/raw/twcs.csv')
threads = reconstruct_threads(df)
pairs = extract_brand_pairs(threads, brand='AppleSupport')
pairs.to_parquet('data/processed/AppleSupport_pairs.parquet', index=False)
print(f'  Extracted {len(pairs):,} AppleSupport conversation pairs.')
"

# 2. Build retrieval index
echo ""
echo "▶ Step 2: Building retrieval index..."
python -c "from src.retrieval.retriever import build_index; build_index(force_rebuild=True)"

# 3. Build golden evaluation set
echo ""
echo "▶ Step 3: Building golden evaluation set..."
python scripts/build_golden_set.py

# 4. Run demo (smoke test)
echo ""
echo "▶ Step 4: Agent smoke test..."
python demo.py --demo 2>&1 | head -80

# 5. Full evaluation
echo ""
echo "▶ Step 5: Running evaluation harness..."
python scripts/run_eval.py $QUICK --judge-n 5

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Reproduction complete! Check reports/ for results.    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Key output files:"
echo "  reports/eval_report.md        — Evaluation results and analysis"
echo "  reports/eval_results.json     — Raw metrics (all systems)"
echo "  data/eval/golden_set.jsonl    — 200-example golden set"
echo "  data/processed/AppleSupport_pairs.parquet — Cleaned conversation pairs"
