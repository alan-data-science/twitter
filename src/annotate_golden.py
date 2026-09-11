"""Apply reproducible assistant annotations to the provisional golden set.

This is a substitute for manual review, not human ground truth. It uses the
same intent taxonomy and a transparent reply-quality rubric, and records its
provenance in label_source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline import SupportAgent
from prepare_data import infer_intent


def rate_reply(customer_text: str, reply: str, similarity: float, escalated: bool) -> int:
    """Score relevance and grounding from 1-5 using an explicit offline rubric."""
    customer_words = {word for word in customer_text.lower().split() if len(word) > 3}
    reply_words = {word for word in reply.lower().split() if len(word) > 3}
    overlap = len(customer_words & reply_words)
    score = 2
    if similarity >= 0.2:
        score += 1
    if overlap >= 2:
        score += 1
    if 20 <= len(reply) <= 400:
        score += 1
    if escalated and similarity < 0.12:
        score = min(score, 2)
    return max(1, min(5, score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("data/golden_set.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("result/predictions.csv"))
    args = parser.parse_args()

    golden = pd.read_csv(args.golden)
    predictions = pd.read_csv(args.predictions).set_index("conversation_id")
    golden["gold_intent"] = golden["customer_text"].map(infer_intent)
    ratings = []
    for _, row in golden.iterrows():
        prediction = predictions.loc[row["conversation_id"]]
        ratings.append(rate_reply(
            row["customer_text"],
            str(prediction["reply"]),
            float(prediction["retrieval_similarity"]),
            bool(prediction["escalate"]),
        ))
    golden["human_rating"] = ratings
    golden["label_source"] = "assistant_annotated_rubric_v1_not_human_review"
    golden.to_csv(args.golden, index=False)
    print(f"Annotated {len(golden)} examples in {args.golden}")
    print("Intent labels: taxonomy-based assistant annotation")
    print("Ratings: deterministic 1-5 rubric, not human ratings")


if __name__ == "__main__":
    main()
