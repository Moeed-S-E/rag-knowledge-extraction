"""Week 8: Topic Modeling and Metadata Integration."""

import sys
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ragkit.embeddings.encoder import Encoder
from ragkit.vectorstore.chroma_client import ChromaStore

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"

# Mock dataset covering 3 distinct themes
MOCK_DATA = [
    # AI / ML
    "Deep learning architectures like CNNs and Transformers are transforming computer vision.",
    "Natural Language Processing (NLP) models rely heavily on self-attention mechanisms.",
    "Reinforcement learning enables agents to learn optimal policies in complex environments.",
    "Generative adversarial networks (GANs) are widely used for image synthesis.",
    "Large language models require massive datasets and parallel GPU computing.",
    "Recurrent neural networks (RNNs) suffer from vanishing gradients on long sequences.",
    "Vector databases like ChromaDB allow efficient semantic search using dense embeddings.",
    "Transfer learning fine-tunes pre-trained weights on specific downstream tasks.",
    "Evaluating RAG systems requires metrics like Precision@K and Recall@K.",
    "Support Vector Machines (SVMs) were popular before the deep learning era.",
    # Finance / Economics
    "Global markets reacted negatively to the recent interest rate hikes by the Federal Reserve.",
    "Cryptocurrency volatility makes Bitcoin a risky but potentially high-reward asset.",
    "Supply chain disruptions have led to increased inflation and higher consumer prices.",
    "Venture capital funding has cooled down in the tech sector over the past year.",
    "Diversifying a stock portfolio can mitigate risks during an economic recession.",
    "Algorithmic trading firms use high-frequency trading strategies to exploit arbitrage.",
    "ESG (Environmental, Social, and Governance) investing is becoming a major trend.",
    "The stock market index reached an all-time high despite geopolitical tensions.",
    "Real estate investments offer a hedge against long-term inflationary pressures.",
    "Central banks use monetary policy to control inflation and stabilize currencies.",
    # Space / Astronomy
    "The James Webb Space Telescope has captured unprecedented images of distant galaxies.",
    "Mars rovers continue to search for signs of ancient microbial life on the red planet.",
    "Black holes exert such strong gravitational pull that not even light can escape.",
    "Exoplanets located in the habitable zone might support liquid water.",
    "SpaceX has successfully landed reusable rocket boosters to lower launch costs.",
    "The Milky Way galaxy is on a collision course with the Andromeda galaxy.",
    "Dark matter and dark energy make up the vast majority of the universe's mass-energy.",
    "Astronauts aboard the International Space Station conduct microgravity experiments.",
    "Asteroid mining could provide rare earth metals essential for electronics.",
    "Solar flares and coronal mass ejections can disrupt satellite communications on Earth.",
]

# We will amplify this dataset to have enough documents for LDA
EXTENDED_MOCK_DATA = []
for i in range(4):  # Duplicate 4 times with minor variations
    for idx, doc in enumerate(MOCK_DATA):
        EXTENDED_MOCK_DATA.append({"text": f"{doc} (Variation {i})", "id": f"doc_m_{idx}_{i}"})

def setup_mock_corpus():
    """Populate ChromaDB with the extended mock dataset."""
    print("[1] Setting up mock corpus...")
    encoder = Encoder(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(persist_dir=PERSIST_DIR)
    collection = store.get_or_create_collection(COLLECTION_NAME)
    
    # Check if we already added them
    existing_docs = store.search(query_embeddings=encoder.encode(["test"])[0].tolist(), k=1)
    if existing_docs and len(existing_docs.get("ids", [[]])[0]) > 0:
        # Check if the DB has enough docs
        count = len(collection.get()["ids"])
        if count > 50:
            print(f"    Corpus already has {count} documents. Skipping generation.")
            return store, collection
    
    print("    Clearing existing small collection and inserting 120 mock documents...")
    # Quick clear by deleting and recreating
    try:
        store.client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = store.get_or_create_collection(COLLECTION_NAME)
    
    texts = [d["text"] for d in EXTENDED_MOCK_DATA]
    ids = [d["id"] for d in EXTENDED_MOCK_DATA]
    metadatas = [{"source": "mock"} for _ in EXTENDED_MOCK_DATA]
    
    embeddings, _ = encoder.encode(texts)
    collection.add(documents=texts, ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
    return store, collection

def main():
    store, collection = setup_mock_corpus()
    
    print("[2] Fetching corpus from Vector DB...")
    db_data = collection.get(include=["documents", "metadatas", "embeddings"])
    documents = db_data["documents"]
    ids = db_data["ids"]
    embeddings = np.array(db_data["embeddings"])
    
    print(f"    Fetched {len(documents)} documents.")
    
    # Preprocessing & Edge Cases
    # 1. Filter out extremely short documents (< 3 words)
    valid_indices = []
    clean_docs = []
    for i, doc in enumerate(documents):
        if len(doc.split()) >= 3:
            valid_indices.append(i)
            clean_docs.append(doc)
            
    print(f"    Filtered out {len(documents) - len(clean_docs)} edge-case documents (too short).")
    
    # 2. Vectorize using CountVectorizer (handles jargon by filtering extremes)
    print("[3] Running Topic Modeling (LDA)...")
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(clean_docs)
    
    # LDA Topic Modeling
    n_topics = 3
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    doc_topic_dist = lda.fit_transform(dtm)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Print Top Words for each topic
    topic_labels = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        label = "-".join(top_words[:2]).upper()
        topic_labels[topic_idx] = label
        print(f"    Topic {topic_idx} [{label}]: {', '.join(top_words)}")
        
    # Get dominant topic for each valid document
    dominant_topics = np.argmax(doc_topic_dist, axis=1)
    
    print("[4] Validating topics (Random Sample of 3 docs per topic)...")
    for t in range(n_topics):
        print(f"\n  --- Topic: {topic_labels[t]} ---")
        topic_docs = [clean_docs[i] for i, dt in enumerate(dominant_topics) if dt == t]
        sample_size = min(3, len(topic_docs))
        for doc in random.sample(topic_docs, sample_size):
            print(f"    - {doc}")
            
    print("\n[5] Updating ChromaDB Metadata with Topic Clusters...")
    # Update ChromaDB collection
    update_ids = [ids[i] for i in valid_indices]
    
    # We must retrieve the existing metadatas to append to them, rather than overwriting
    # But for simplicity, we can just replace them since they only contain 'source'
    new_metadatas = []
    for i, t in zip(valid_indices, dominant_topics):
        meta = db_data["metadatas"][i] or {}
        meta["topic"] = topic_labels[t]
        new_metadatas.append(meta)
        
    collection.update(ids=update_ids, metadatas=new_metadatas)
    print("    Metadata successfully integrated!")
    
    print("[6] Generating Visualizations...")
    # We use PCA on the embeddings to plot them, colored by Topic
    # We only plot valid indices
    valid_embeddings = embeddings[valid_indices]
    pca = PCA(n_components=2, random_state=42)
    reduced_embs = pca.fit_transform(valid_embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embs[:, 0], 
        y=reduced_embs[:, 1], 
        hue=[topic_labels[t] for t in dominant_topics],
        palette="Set1",
        s=100,
        alpha=0.7
    )
    plt.title("Corpus Topic Clusters (PCA of Dense Embeddings)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Topic")
    
    report_dir = Path(__file__).resolve().parent.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / "topic_clusters.png"
    plt.savefig(out_path)
    print(f"    Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
