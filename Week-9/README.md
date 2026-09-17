# Week 9: Named Entity Recognition & Advanced Retrieval

## Objectives
- Apply Named Entity Recognition (NER) to the corpus using `spaCy`.
- Write evaluation metrics to test the NER against 50 manually labeled samples.
- Extract key entities and store them as metadata within the ChromaDB chunks.
- Update the RAG retrieval logic to boost chunks based on specific entities.
- Document accuracy and usefulness of the extracted metadata.

## Methodology

### 1. NLP Metadata Extraction (`ner_enrichment.py`)
We utilized `spaCy` (specifically the `en_core_web_sm` model) to perform Named Entity Recognition on all 120 documents within the vector database. We specifically targeted high-value entity types:
- `ORG` (Organizations, e.g., SpaceX, Federal Reserve)
- `PERSON` (People)
- `GPE` (Geopolitical Entities, e.g., Mars, Earth)
- `PRODUCT` (Products, e.g., Transformers, Bitcoin)

The extracted entities were stored into each chunk's ChromaDB metadata under the `"entities"` key as a comma-separated string.

### 2. NER Evaluation
To validate the model's accuracy, the script incorporates an oracle (ground truth) mapping for 50 documents to test if `spaCy` correctly identifies known entities. 
- **Metrics Calculated**: Precision, Recall, and F1-Score.
- **Results**: 
  - *Precision*: ~0.60
  - *Recall*: ~0.70
  - *F1-Score*: ~0.65 
  *(Note: Minor variance depends on spaCy's tokenizer handling our mock data, but it correctly identifies the vast majority of our target entities like "Federal Reserve" and "SpaceX").*

### 3. Entity-Boosted Retrieval (`rag_cli_v4.py`)
Because semantic search (dense embeddings) sometimes struggles with exact keyword matching (e.g., matching the exact company "SpaceX" instead of a semantically similar aerospace company), we implemented **Metadata Boosting**.

When a user passes the `--entity-boost` flag:
1. The system "over-fetches" by retrieving 3x the normal amount of chunks from ChromaDB.
2. It parses the `"entities"` metadata of each retrieved chunk.
3. If the requested entity is found in the metadata, the chunk receives an artificial **distance reduction** (e.g., `-0.5`).
4. The chunks are re-sorted, heavily prioritizing documents that explicitly mention the named entity, before slicing the Top-K for the LLM.

## How to Run

**1. Run the NER Pipeline & Evaluation:**
This script will evaluate spaCy on the dataset and enrich ChromaDB.
```bash
uv run python Week-9/scripts/ner_enrichment.py
```

**2. Test Entity-Boosted RAG:**
Run the standard RAG chat, but aggressively prioritize a specific entity if it appears in the database.
```bash
uv run python Week-9/scripts/rag_cli_v4.py --entity-boost "Federal Reserve"
```
