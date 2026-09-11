# Week 8: Topic Modeling and Topic-Filtered Retrieval

## Objectives
- Apply Topic Modeling (Latent Dirichlet Allocation) to the document corpus.
- Visualize the resulting topic clusters in a 2D PCA projection.
- Validate the topic model output by manually reviewing documents from each cluster.
- Filter out edge cases: extremely short documents and heavy jargon.
- Integrate the topic metadata into the vector database.
- Implement a topic-filtered retrieval capability in the RAG CLI.

## Methodology

### 1. Topic Modeling (LDA)
Because the initial corpus was exceptionally small (only 6 chunks), a robust mock dataset of 120 documents was generated representing three distinct themes: **AI/ML**, **Finance**, and **Space**.
- **Preprocessing**: We ignored extremely short documents (less than 3 words) to prevent noise. We also utilized `CountVectorizer` with `stop_words='english'`, `max_df=0.9`, and `min_df=2` to filter out overly common jargon and extremely rare typos.
- **Algorithm**: We used Scikit-Learn's `LatentDirichletAllocation` (LDA) configured for 3 topics.
- **Output**: The script automatically labels the topics based on their top 2 defining keywords (e.g., `SPACE-GALAXY`, `LEARNING-MODELS`).

### 2. Visualization
We performed a PCA (Principal Component Analysis) on the 384-dimensional dense sentence embeddings and projected them into 2D space. The points were color-coded according to the dominant LDA topic, providing a visual confirmation that the semantic embeddings align well with the lexical LDA topics.
- The visualization is saved at: `Week-8/reports/topic_clusters.png`

### 3. Metadata Integration & Filtered Retrieval
- The predicted dominant topic label for each document was injected directly into ChromaDB as part of the chunk's `metadata`.
- The RAG CLI (`rag_cli_v3.py`) was upgraded to accept a `--topic` flag.
- When the user searches with a topic filter, ChromaDB leverages its `where={"topic": "..."}` capability to strictly narrow down the semantic search space to only that specific cluster, enhancing retrieval precision and reducing cross-domain contamination.

## How to Run

1. **Generate the Topic Model & Visualizations**:
   This script will initialize the dataset, run LDA, save the visual clusters, and embed the metadata into ChromaDB.
   ```bash
   uv run python Week-8/scripts/topic_modeling.py
   ```

2. **Run Topic-Filtered RAG**:
   You can run the RAG CLI as usual, or optionally pass the `--topic` flag with the name of the topic cluster printed during the modeling step (e.g., "LEARNING-MODELS").
   ```bash
   uv run python Week-8/scripts/rag_cli_v3.py --topic "LEARNING-MODELS"
   ```
