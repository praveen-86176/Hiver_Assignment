"""
Phase 3: Retrieval / Grounding Module

Builds a TF-IDF + cosine similarity retriever over AppleSupport resolved conversation pairs.
Given a new customer message, returns top-k historically similar (customer_message, brand_reply) pairs
to use as grounding context for reply generation.

Design choice: TF-IDF (not dense embeddings) for Phase 3 because:
- Runs fully offline, zero API cost, < 2s index build time on 74k pairs
- Deterministic and reproducible (no model download required)
- Competitive with dense embeddings for keyword-heavy technical support queries
- Dense embeddings (sentence-transformers) can be swapped in if needed (see retriever.py comment)
"""

import json
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("data/processed")
PAIRS_PATH = INDEX_DIR / "AppleSupport_pairs.parquet"
INDEX_PATH = INDEX_DIR / "retrieval_index.pkl"


def clean_for_retrieval(text: str) -> str:
    """Normalize text for TF-IDF retrieval."""
    t = re.sub(r"@[A-Za-z0-9_]+", "", str(text))
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return t.lower().strip()


def build_index(force_rebuild: bool = False) -> dict[str, Any]:
    """
    Build and cache the TF-IDF retrieval index over AppleSupport pairs.

    # NOTE(phase3-extension): To upgrade to dense embeddings (sentence-transformers),
    # replace TfidfVectorizer with SentenceTransformer('all-MiniLM-L6-v2').encode()
    # and cosine_similarity with sklearn.metrics.pairwise.cosine_similarity on
    # numpy arrays. The rest of the retriever interface stays the same.
    """
    if INDEX_PATH.exists() and not force_rebuild:
        print(f"Loading cached retrieval index from {INDEX_PATH}...")
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)

    print("Building retrieval index from AppleSupport pairs...")
    df = pd.read_parquet(PAIRS_PATH)
    df = df.dropna(subset=["customer_message", "brand_reply"]).reset_index(drop=True)

    df["clean_msg"] = df["customer_message"].apply(clean_for_retrieval)

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["clean_msg"])

    index = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "customer_messages": df["customer_message"].tolist(),
        "brand_replies": df["brand_reply"].tolist(),
        "thread_ids": df["thread_id"].tolist(),
        "n_pairs": len(df),
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"Built index: {len(df):,} pairs, vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"Saved index to {INDEX_PATH}")
    return index


def retrieve(
    query: str,
    index: Optional[dict] = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve top-k most similar historical (customer_message, brand_reply) pairs.

    Args:
        query: Incoming customer message text.
        index: Pre-built retrieval index dict (builds+caches if None).
        top_k: Number of similar examples to return.

    Returns:
        List of dicts with keys: rank, similarity, customer_message, brand_reply, thread_id
    """
    if index is None:
        index = build_index()

    clean_query = clean_for_retrieval(query)
    query_vec = index["vectorizer"].transform([clean_query])
    similarities = cosine_similarity(query_vec, index["tfidf_matrix"]).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "similarity": round(float(similarities[idx]), 4),
            "customer_message": index["customer_messages"][idx],
            "brand_reply": index["brand_replies"][idx],
            "thread_id": int(index["thread_ids"][idx]),
        })
    return results


def format_grounding_context(retrieved: list[dict], max_chars: int = 800) -> str:
    """
    Format retrieved pairs into a grounding context string for LLM prompting.
    Truncates to avoid bloating prompt context.
    """
    lines = ["Historical Apple Support resolutions similar to this query:\n"]
    total_chars = 0
    for item in retrieved:
        pair_text = (
            f"[Example {item['rank']} | Similarity: {item['similarity']}]\n"
            f"  Customer: {item['customer_message'][:120]}\n"
            f"  AppleSupport: {item['brand_reply'][:200]}\n"
        )
        if total_chars + len(pair_text) > max_chars:
            break
        lines.append(pair_text)
        total_chars += len(pair_text)
    return "\n".join(lines)


if __name__ == "__main__":
    idx = build_index()
    test_query = "@AppleSupport my battery is dying super fast since I updated to iOS 11. Help!"
    results = retrieve(test_query, idx, top_k=3)
    print(f"\nQuery: {test_query}\n")
    for r in results:
        print(f"Rank {r['rank']} (sim={r['similarity']}):")
        print(f"  Customer: {r['customer_message'][:100]}")
        print(f"  Reply:    {r['brand_reply'][:120]}\n")
