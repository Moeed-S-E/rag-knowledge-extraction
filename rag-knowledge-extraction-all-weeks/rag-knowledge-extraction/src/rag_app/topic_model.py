"""Optional LDA topic modeling with a transparent keyword fallback."""
from __future__ import annotations

from collections import Counter


def fit_topics(records: list[dict], topic_count: int = 5, top_words: int = 8) -> dict:
    texts = [str(record.get("text", "")) for record in records]
    if not texts:
        return {"provider": "none", "topics": []}
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(stop_words="english", max_features=2_000)
        matrix = vectorizer.fit_transform(texts)
        actual_topics = max(1, min(topic_count, matrix.shape[0], matrix.shape[1]))
        model = LatentDirichletAllocation(n_components=actual_topics, random_state=42, learning_method="batch", max_iter=20)
        assignments = model.fit_transform(matrix).argmax(axis=1)
        vocabulary = vectorizer.get_feature_names_out()
        topics = []
        for index, component in enumerate(model.components_):
            words = [str(vocabulary[position]) for position in component.argsort()[-top_words:][::-1]]
            topics.append({"topic_id": index, "keywords": words, "document_ids": [records[row].get("id") for row, assigned in enumerate(assignments) if assigned == index], "sample_documents": [records[row].get("id") for row, assigned in enumerate(assignments) if assigned == index][:20]})
        return {"provider": "sklearn-lda", "topic_count": actual_topics, "topics": topics}
    except Exception:
        groups: dict[str, list[str]] = {}
        for record in records:
            words = [word.lower() for word in str(record.get("text", "")).split() if len(word) > 4]
            key = words[0] if words else "uncategorized"
            groups.setdefault(key, []).append(str(record.get("id")))
        topics = [{"topic_id": index, "keywords": [key], "document_ids": ids, "sample_documents": ids[:20]} for index, (key, ids) in enumerate(sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:topic_count])]
        return {"provider": "keyword-fallback", "topic_count": len(topics), "topics": topics}
