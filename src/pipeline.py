"""Intent classification, retrieval, response grounding, and escalation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from prepare_data import INTENTS


ESCALATION_TERMS = ("lawsuit", "legal", "police", "fraud", "stolen", "security breach", "urgent")


@dataclass
class SupportAgent:
	conversations: pd.DataFrame
	vectorizer: TfidfVectorizer
	matrix: Any
	classifier: LogisticRegression

	@classmethod
	def fit(cls, conversations: pd.DataFrame) -> "SupportAgent":
		vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=50000)
		matrix = vectorizer.fit_transform(conversations["customer_text"].fillna(""))
		classifier = LogisticRegression(max_iter=500, class_weight="balanced")
		classifier.fit(matrix, conversations["intent"])
		return cls(conversations, vectorizer, matrix, classifier)

	def predict_intent(self, text: str) -> tuple[str, float]:
		features = self.vectorizer.transform([text])
		probabilities = self.classifier.predict_proba(features)[0]
		index = probabilities.argmax()
		return str(self.classifier.classes_[index]), float(probabilities[index])

	def retrieve(self, text: str, top_k: int = 3) -> pd.DataFrame:
		scores = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
		indices = scores.argsort()[-top_k:][::-1]
		result = self.conversations.iloc[indices].copy()
		result["similarity"] = scores[indices]
		return result

	def answer(self, text: str) -> dict[str, Any]:
		intent, confidence = self.predict_intent(text)
		matches = self.retrieve(text)
		best = matches.iloc[0]
		lowered = text.lower()
		reasons = []
		if confidence < 0.45:
			reasons.append("low intent confidence")
		if float(best["similarity"]) < 0.12:
			reasons.append("no sufficiently similar historical conversation")
		if any(term in lowered for term in ESCALATION_TERMS):
			reasons.append("sensitive or high-risk wording")
		escalate = bool(reasons)
		reply = str(best["agent_reply"])
		reply = re.sub(r"@\w+", "", reply).strip()
		return {
			"predicted_intent": intent,
			"intent_confidence": round(confidence, 4),
			"reply": reply,
			"escalate": escalate,
			"escalation_reason": "; ".join(reasons) if reasons else "routine request with grounded match",
			"retrieved_conversation_id": best["conversation_id"],
			"retrieval_similarity": round(float(best["similarity"]), 4),
		}


def evaluate_intents(agent: SupportAgent, golden: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
	predictions = []
	for _, row in golden.iterrows():
		result = agent.answer(str(row["customer_text"]))
		predictions.append({**row.to_dict(), **result})
	output = pd.DataFrame(predictions)
	labels = output["gold_intent"]
	metrics = {
		"agent_accuracy": float(accuracy_score(labels, output["predicted_intent"])),
		"agent_macro_f1": float(f1_score(labels, output["predicted_intent"], average="macro", zero_division=0)),
	}
	return output, metrics


def majority_baseline(train: pd.DataFrame, golden: pd.DataFrame) -> dict[str, float]:
	label = train["intent"].mode().iloc[0]
	predicted = [label] * len(golden)
	return {"accuracy": float(accuracy_score(golden["gold_intent"], predicted)),
			"macro_f1": float(f1_score(golden["gold_intent"], predicted, average="macro", zero_division=0))}


def ml_baseline(train: pd.DataFrame, golden: pd.DataFrame) -> dict[str, float]:
	vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
	x_train = vectorizer.fit_transform(train["customer_text"])
	x_test = vectorizer.transform(golden["customer_text"])
	model = LogisticRegression(max_iter=500, class_weight="balanced").fit(x_train, train["intent"])
	predicted = model.predict(x_test)
	return {"accuracy": float(accuracy_score(golden["gold_intent"], predicted)),
			"macro_f1": float(f1_score(golden["gold_intent"], predicted, average="macro", zero_division=0))}
