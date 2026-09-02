"""Practical lightweight NLP analysis with optional scikit-learn visualization."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path


_STOPWORDS = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "into", "using", "which", "have", "has", "will", "your", "their", "about", "based", "not"}
_POSITIVE = {"good", "useful", "accurate", "excellent", "helpful", "success", "successful", "robust", "reliable", "fast"}
_NEGATIVE = {"bad", "wrong", "poor", "failure", "failing", "slow", "error", "issue", "unsupported", "missing"}
_ENTITY = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[A-Z]{2,})\b")
_WORD = re.compile(r"[\w\u0080-\uffff]+", re.UNICODE)


def sentiment(text: str) -> dict:
    words = {word.lower() for word in _WORD.findall(text or "")}
    positive = len(words & _POSITIVE)
    negative = len(words & _NEGATIVE)
    label = "positive" if positive > negative else "negative" if negative > positive else "neutral"
    return {"label": label, "positive_hits": positive, "negative_hits": negative, "score": positive - negative}


def entities(text: str) -> list[str]:
    return sorted(set(_ENTITY.findall(text or "")))


def top_terms(texts: list[str], limit: int = 10) -> list[dict]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(word.lower() for word in _WORD.findall(text or "") if len(word) > 3 and word.lower() not in _STOPWORDS)
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]


def analyze_documents(records: list[dict], topic_count: int = 5) -> dict:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        terms = top_terms([record.get("text", "")], limit=1)
        topic = terms[0]["term"] if terms else "uncategorized"
        groups[topic].append({
            "id": record.get("id"),
            "sentiment": sentiment(record.get("text", "")),
            "entities": entities(record.get("text", "")),
        })
    ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:topic_count]
    return {
        "document_count": len(records),
        "topic_count": len(ordered),
        "topics": [{"topic": topic, "document_count": len(items), "sample_documents": items[:20]} for topic, items in ordered],
        "top_corpus_terms": top_terms([record.get("text", "") for record in records]),
        "sentiment_distribution": dict(Counter(sentiment(record.get("text", ""))["label"] for record in records)),
        "entity_frequency": dict(Counter(entity for record in records for entity in entities(record.get("text", ""))).most_common(20)),
    }


def save_topic_plot(report: dict, output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        topics = [item["topic"] for item in report.get("topics", [])]
        counts = [item["document_count"] for item in report.get("topics", [])]
        if not topics:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.bar(topics, counts, color="#2563eb")
        plt.title("Corpus topic distribution")
        plt.xlabel("Topic keyword")
        plt.ylabel("Documents")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return True
    except Exception:
        return False
