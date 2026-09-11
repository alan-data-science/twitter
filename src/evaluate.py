"""Run the complete experiment and write result artifacts."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

from pipeline import SupportAgent, evaluate_intents, majority_baseline, ml_baseline


def judge_reply(row: pd.Series) -> tuple[float, str]:
	"""Use an optional OpenAI-compatible judge, with a deterministic offline fallback."""
	api_key = os.getenv("OPENAI_API_KEY")
	if api_key:
		payload = json.dumps({
			"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
			"temperature": 0,
			"response_format": {"type": "json_object"},
			"messages": [{"role": "user", "content": (
				"Score this support reply from 1 to 5 for relevance, groundedness, and helpfulness. "
				"Return JSON exactly as {\"score\": number}.\n"
				f"Customer: {row['customer_text']}\nReply: {row['reply']}"
			)}],
		}).encode("utf-8")
		request = urllib.request.Request(
			os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
			data=payload,
			headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urllib.request.urlopen(request, timeout=30) as response:
				content = json.loads(response.read())
			score = float(json.loads(content["choices"][0]["message"]["content"])["score"])
			return max(1.0, min(5.0, score)), "llm_judge"
		except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
			pass

	reply = str(row["reply"]).lower()
	source = str(row["customer_text"]).lower()
	overlap = len(set(source.split()) & set(reply.split()))
	score = min(5.0, 2.5 + min(overlap / 8.0, 1.5) + (0.5 if len(reply) > 20 else 0))
	return round(score, 2), "offline_groundedness_proxy"


def main() -> None:
	root = Path(__file__).resolve().parents[1]
	load_dotenv(root / ".env")
	processed = root / "data/processed/brand_conversations.csv"
	golden_path = root / "data/golden_set.csv"
	result_dir = root / "result"
	result_dir.mkdir(exist_ok=True)
	if not processed.exists() or not golden_path.exists():
		raise FileNotFoundError("Run src/prepare_data.py before src/evaluate.py")

	train = pd.read_csv(processed).dropna(subset=["customer_text", "agent_reply", "intent"])
	golden = pd.read_csv(golden_path).dropna(subset=["customer_text", "gold_intent"])
	agent = SupportAgent.fit(train)
	predictions, agent_metrics = evaluate_intents(agent, golden)
	judged = predictions.apply(judge_reply, axis=1, result_type="expand")
	predictions["reply_score"] = judged[0]
	predictions["judge_type"] = judged[1]
	predictions.to_csv(result_dir / "predictions.csv", index=False)
	metrics = {
		**agent_metrics,
		"majority_baseline": majority_baseline(train, golden),
		"ml_baseline": ml_baseline(train, golden),
		"reply_quality_mean": float(predictions["reply_score"].mean()),
		"golden_examples": int(len(golden)),
		"label_source": str(golden.get("label_source", pd.Series(["unspecified"])).iloc[0]),
		"human_judge_agreement": None,
	}
	is_human_reviewed = str(metrics["label_source"]).startswith("human_reviewed")
	if is_human_reviewed and "human_rating" in golden and golden["human_rating"].notna().any():
		human = pd.to_numeric(golden["human_rating"], errors="coerce")
		mask = human.notna()
		metrics["human_judge_agreement"] = float(cohen_kappa_score(human[mask] >= 3, predictions.loc[mask, "reply_score"] >= 3))

	errors = predictions[predictions["gold_intent"] != predictions["predicted_intent"]].copy()
	grouped = errors.groupby(["gold_intent", "predicted_intent"]).size().sort_values(ascending=False).head(5)
	lines = ["# Top failure modes", "", f"Misclassified examples: {len(errors)}", ""]
	if grouped.empty:
		lines.append("No intent errors in this run.")
	else:
		for (gold, predicted), count in grouped.items():
			lines.append(f"- **{gold} -> {predicted}**: {count} examples")
	lines += ["", "## Limitations", "", "- The golden set uses assistant-generated intent labels and deterministic rubric scores, not human ground truth. Replace them with 150-250 manual labels and ratings before claiming final assignment results.", "- Replies are retrieved verbatim from historical support messages; they may contain stale policy or account-specific language.", "- The offline reply judge is a proxy. Set up an LLM judge separately and record its rubric and agreement with human ratings."]
	(result_dir / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	(result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
	report = "# Hiver support-agent experiment\n\n" + json.dumps(metrics, indent=2) + "\n\n" + "The golden set is assistant-annotated and provisional: replace its labels and ratings with 150-250 manual annotations before using these metrics as final submission results.\n"
	(root / "report/report.md").write_text(report, encoding="utf-8")
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
