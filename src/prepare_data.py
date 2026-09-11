"""Create a small, reproducible experiment set from the TWCS CSV.

The source file is large, so this module uses csv.DictReader and only keeps
rows for the selected support account plus the parent tweets it references.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


INTENTS = (
	"account_access",
	"billing_payment",
	"delivery_order",
	"technical_issue",
	"cancellation_refund",
	"general_information",
)

KEYWORDS = {
	"account_access": ("login", "password", "account", "locked", "sign in", "access"),
	"billing_payment": ("charge", "charged", "payment", "billing", "invoice", "refund"),
	"delivery_order": ("order", "delivery", "shipping", "tracking", "arrive", "package"),
	"technical_issue": ("error", "broken", "crash", "not working", "bug", "app", "website"),
	"cancellation_refund": ("cancel", "cancellation", "return", "money back", "refund"),
}


def infer_intent(text: str) -> str:
	lowered = text.lower()
	scores = {intent: sum(word in lowered for word in words) for intent, words in KEYWORDS.items()}
	best_intent, best_score = max(scores.items(), key=lambda item: item[1])
	return best_intent if best_score else "general_information"


def build_dataset(raw_path: Path, processed_path: Path, golden_path: Path,
				  brand: str = "AmazonHelp", max_examples: int = 12000,
				  golden_size: int = 200, seed: int = 7) -> None:
	selected: dict[str, dict[str, str]] = {}
	with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
		for row in csv.DictReader(handle):
			if row.get("author_id") == brand and row.get("in_response_to_tweet_id"):
				selected[row["tweet_id"]] = row

	parent_ids = {row["in_response_to_tweet_id"] for row in selected.values()}
	parents: dict[str, dict[str, str]] = {}
	if parent_ids:
		with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
			for row in csv.DictReader(handle):
				if row.get("tweet_id") in parent_ids:
					parents[row["tweet_id"]] = row

	records = []
	for reply in selected.values():
		parent = parents.get(reply["in_response_to_tweet_id"])
		if not parent or not parent.get("text", "").strip() or not reply.get("text", "").strip():
			continue
		records.append({
			"conversation_id": parent["tweet_id"],
			"brand": brand,
			"customer_text": parent["text"].replace("\n", " ").strip(),
			"agent_reply": reply["text"].replace("\n", " ").strip(),
			"created_at": parent.get("created_at", ""),
			"intent": infer_intent(parent["text"]),
		})
	unique_records = {}
	for record in records:
		unique_records.setdefault(record["conversation_id"], record)
	records = list(unique_records.values())
	rng = random.Random(seed)
	rng.shuffle(records)
	records = records[:max_examples]
	if not records:
		raise RuntimeError(f"No paired conversations found for {brand!r} in {raw_path}")

	golden = records[:min(golden_size, len(records) // 5)]
	train_records = records[len(golden):]
	if not train_records:
		raise RuntimeError("Not enough conversations left for training after reserving the golden set")

	processed_path.parent.mkdir(parents=True, exist_ok=True)
	with processed_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=train_records[0].keys())
		writer.writeheader()
		writer.writerows(train_records)

	with golden_path.open("w", encoding="utf-8", newline="") as handle:
		fields = ["conversation_id", "brand", "customer_text", "gold_intent", "human_rating", "label_source"]
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		for record in golden:
			writer.writerow({
				"conversation_id": record["conversation_id"],
				"brand": record["brand"],
				"customer_text": record["customer_text"],
				"gold_intent": record["intent"],
				"human_rating": "",
				"label_source": "weak_keyword_label_replace_with_human_review",
			})

	print(f"Wrote {len(train_records)} training conversations to {processed_path}")
	print(f"Wrote {len(golden)} provisional golden examples to {golden_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--brand", default="AmazonHelp")
	parser.add_argument("--max-examples", type=int, default=12000)
	parser.add_argument("--golden-size", type=int, default=200)
	parser.add_argument("--raw", type=Path, default=Path("data/raw/twcs.csv"))
	parser.add_argument("--processed", type=Path, default=Path("data/processed/brand_conversations.csv"))
	parser.add_argument("--golden", type=Path, default=Path("data/golden_set.csv"))
	args = parser.parse_args()
	build_dataset(args.raw, args.processed, args.golden, args.brand, args.max_examples, args.golden_size)
