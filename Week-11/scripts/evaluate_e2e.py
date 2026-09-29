"""Automated end-to-end evaluation suite for the RAG pipeline."""

import sys
import json
import time
import importlib.util
from pathlib import Path
from statistics import mean

from fastapi.testclient import TestClient

# Dynamically import the API from Week-10
project_root = Path(__file__).resolve().parent.parent.parent
main_py_path = project_root / "Week-10" / "api" / "main.py"
spec = importlib.util.spec_from_file_location("main_api", main_py_path)
main_module = importlib.util.module_from_spec(spec)
sys.modules["main_api"] = main_module
spec.loader.exec_module(main_module)

app = main_module.app

# 30 Evaluation Questions spanning different topics/difficulties
QA_PAIRS = [
    {"q": "What is attention in deep learning?", "topic": None},
    {"q": "How do Convolutional Neural Networks work?", "topic": None},
    {"q": "What is the difference between RNN and LSTM?", "topic": None},
    {"q": "Explain the Transformer architecture.", "topic": None},
    {"q": "What is a residual connection?", "topic": None},
    {"q": "How does batch normalization help training?", "topic": None},
    {"q": "What is the vanishing gradient problem?", "topic": None},
    {"q": "Explain dropout in neural networks.", "topic": None},
    {"q": "What is the role of the activation function?", "topic": None},
    {"q": "Describe the Adam optimizer.", "topic": None},
    {"q": "What is transfer learning?", "topic": None},
    {"q": "How does fine-tuning work in NLP?", "topic": None},
    {"q": "What is BERT?", "topic": None},
    {"q": "Explain generative adversarial networks (GANs).", "topic": None},
    {"q": "What is self-supervised learning?", "topic": None},
    {"q": "How is reinforcement learning used in robotics?", "topic": None},
    {"q": "What are word embeddings?", "topic": None},
    {"q": "What is cross-entropy loss?", "topic": None},
    {"q": "Explain max pooling.", "topic": None},
    {"q": "What is weight decay?", "topic": None},
    {"q": "How do autoencoders work?", "topic": None},
    {"q": "What is the purpose of a validation set?", "topic": None},
    {"q": "Explain early stopping.", "topic": None},
    {"q": "What are hyper-parameters?", "topic": None},
    {"q": "How does stochastic gradient descent differ from standard GD?", "topic": None},
    {"q": "What is data augmentation?", "topic": None},
    {"q": "Describe few-shot learning.", "topic": None},
    {"q": "What is multi-modal learning?", "topic": None},
    {"q": "Explain the concept of model distillation.", "topic": None},
    {"q": "What happens if you ask an off-topic question about cooking pasta?", "topic": None}, # Off-topic check
]

def run_evaluation():
    print(f"Starting E2E Evaluation of {len(QA_PAIRS)} queries...")
    
    metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "hallucination_detected": 0,
        "retrieved_chunks": [],
        "retrieval_latency": [],
        "generation_latency": [],
        "total_latency": [],
        "failed_queries": []
    }

    # Use TestClient with a with-block to trigger the lifespan (startup/shutdown)
    with TestClient(app) as client:
        print("Backend models loaded. Running queries...")
        
        for idx, item in enumerate(QA_PAIRS):
            q = item["q"]
            print(f"[{idx+1}/{len(QA_PAIRS)}] Query: {q}")
            
            payload = {"question": q, "k": 3}
            if item["topic"]:
                payload["topic"] = item["topic"]
                
            response = client.post("/query", json=payload)
            
            metrics["total_requests"] += 1
            
            if response.status_code == 200:
                data = response.json()
                metrics["successful_requests"] += 1
                
                if not data["is_supported"]:
                    metrics["hallucination_detected"] += 1
                
                metrics["retrieved_chunks"].append(data["retrieved_chunks"])
                metrics["retrieval_latency"].append(data["latencies"]["retrieval_ms"])
                metrics["generation_latency"].append(data["latencies"]["generation_ms"])
                metrics["total_latency"].append(data["latencies"]["total_ms"])
                
                print(f"   -> Result: Latency {data['latencies']['total_ms']}ms | Supported: {data['is_supported']}")
            else:
                print(f"   -> Failed with status {response.status_code}")
                metrics["failed_queries"].append({"query": q, "status": response.status_code})

    # Generate Report
    report_dir = project_root / "Week-11" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.md"
    
    avg_retrieval = mean(metrics["retrieval_latency"]) if metrics["retrieval_latency"] else 0
    avg_generation = mean(metrics["generation_latency"]) if metrics["generation_latency"] else 0
    avg_total = mean(metrics["total_latency"]) if metrics["total_latency"] else 0
    avg_chunks = mean(metrics["retrieved_chunks"]) if metrics["retrieved_chunks"] else 0
    
    success_rate = (metrics["successful_requests"] / metrics["total_requests"]) * 100
    hallucination_rate = (metrics["hallucination_detected"] / metrics["successful_requests"]) * 100 if metrics["successful_requests"] else 0
    
    report_content = f"""# E2E RAG Evaluation Report

## Overview
- **Total Queries Executed**: {metrics['total_requests']}
- **Successful Requests**: {metrics['successful_requests']} ({success_rate:.1f}%)
- **Failed Requests**: {len(metrics['failed_queries'])}

## Generation Quality
- **Hallucinations Detected (LLM-as-a-Judge)**: {metrics['hallucination_detected']} / {metrics['successful_requests']} ({hallucination_rate:.1f}%)
- **Average Chunks Retrieved per Query**: {avg_chunks:.1f}

## Performance Latency (Average)
- **Retrieval Phase**: {avg_retrieval:.1f} ms
- **Generation Phase**: {avg_generation:.1f} ms
- **Total End-to-End Pipeline**: {avg_total:.1f} ms

## Known System Limitations & Bottlenecks
1. **Model Cold Starts**: The OpenRouter API and local SentenceTransformers introduce high variance in latency if cold-started. We solved local cold starts via FastAPI lifespan, but network IO to the LLM remains a bottleneck.
2. **Synchronous DB Calls**: ChromaDB `search` blocks the main thread in our API. Under extremely high concurrency, this could throttle the event loop.
3. **Sequential Pipeline**: The Hallucination verification runs *after* generation. We must wait for the LLM to generate the answer, then wait again for the LLM to verify it. This doubles the generation latency.
4. **Context Window Limits**: We hard limit `k=3` (or up to 20) but we do not actively summarize long documents, which might overflow the context window of smaller models or bloat token costs.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_evaluation()
