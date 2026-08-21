# Week 5: Retrieval Evaluation

## Objectives
- Create a manual test set of 20 user queries paired with expected "ground truth" chunks.
- Implement evaluation metrics: Precision@K and Recall@K.
- Evaluate the retrieval performance of our ChromaDB semantic search.
- Document hyperparameter experimentation (tuning `K`) and refine retrieval logic.

## Approach & Implementation

1. **Metrics Implementation**: 
   - Added `precision_at_k` and `recall_at_k` in `src/ragkit/evaluation/metrics.py`. These standard Information Retrieval metrics help us evaluate the relevance of the chunks returned by our vector database.

2. **Test Set Creation**: 
   - Created a diverse manual test set of 20 queries spanning 6 distinct topics (Computer Vision, NLP, Multilingual processing, Text sanitization, HTML handling, and Vector databases). 
   - Each query is mapped precisely to 1 expected ground-truth chunk.

3. **Benchmarking Script**: 
   - Developed `evaluate_retrieval.py` which runs the entire test suite against our ChromaDB index using different settings for the retrieval window size `K`.

## Evaluation Results

We experimented with `K` (the number of chunks retrieved) at values `1`, `2`, and `3`:

| Top-K | Average Precision | Average Recall |
|---|---|---|
| K = 1 | 1.0000 | 1.0000 |
| K = 2 | 0.5000 | 1.0000 |
| K = 3 | 0.3333 | 1.0000 |

### Impact of Hyperparameter Changes

Because each query in our test dataset has exactly 1 relevant ground-truth chunk:
- **K=1 yields perfect metrics**. The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is incredibly effective at identifying the most semantically relevant text, meaning the top 1 result is always the correct one.
- **As K increases, Precision drops**. When K=2, the system returns 2 chunks, but only 1 is relevant. Precision drops to 1/2 (0.50). At K=3, it drops to 1/3 (0.33).
- **Recall remains perfect (1.0)** across all values of K because the true relevant chunk is always successfully captured in the retrieval window.

## Refinement of Retrieval Logic

Based on these evaluation findings, we can conclude that the `all-MiniLM-L6-v2` model is highly accurate for this dataset. 

For the final RAG logic, the optimal value depends on the downstream LLM generator:
- If context window tokens are extremely scarce, **K=1 or K=2** is highly optimal for these kinds of direct, fact-seeking queries.
- In a real-world scenario with a much larger corpus where information might be spread across multiple chunks, a slightly higher K (e.g. K=3 to K=5) would be preferred to maximize recall, and we would employ a **Re-ranker** to push the most precise chunks to the top before feeding them to the LLM. 
- For now, we will set our default retrieval `K=3` to be safe against complex multi-hop queries, while knowing our top result is extremely high quality.
