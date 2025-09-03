"""
Comparative Evaluation: Vector Baseline vs Hybrid RAG.

Demonstrates the performance difference of Graph-Hard-Filtering
using a two-pipeline A/B benchmark with RAGAS metrics.

Evaluator/Generator LLM: Google Gemini 2.0 Flash
Embeddings: BAAI/bge-m3 (local)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add project root to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import get_settings
from backend.generation.synthesizer import ResponseSynthesizer
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_retriever import VectorRetriever
from backend.utils.llm_patch import apply_gemini_ragas_patch

# Initialize professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Apply necessary compatibility patches for RAGAS evaluation
apply_gemini_ragas_patch()

def load_manual_dataset() -> list[dict]:
    """Load the gold-standard manual evaluation dataset."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sample",
    )
    file_path = os.path.join(data_dir, "manual_meat_eval.json")
    
    if not os.path.exists(file_path):
        logger.error(f"Manual dataset missing at {file_path}")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

async def _evaluate_pipeline(
    questions: list[dict],
    retriever: Any,
    label: str,
) -> dict:
    """Run a specific RAG pipeline through the evaluation set."""
    synthesizer = ResponseSynthesizer()
    results_pkg = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(questions):
        query = item["question"]
        logger.info(f"  [{label}] Evaluating {i + 1}/{len(questions)}: {query[:50]}...")

        # Retrieval - Polymorphic handling for Vector vs Hybrid
        if hasattr(retriever, 'aretrieve'):
            results = await retriever.aretrieve(query, top_k=3, include_images=False)
        else:
            results = await retriever.aretrieve_all(query, top_k_text=3, include_images=False)

        # Build response with production synthesizer (includes internal retries)
        response = await synthesizer.asynthesize(query, results)
        
        results_pkg["question"].append(query)
        results_pkg["answer"].append(response.response)
        results_pkg["contexts"].append([res.get("text", "") for res in results.get("text_results", [])])
        results_pkg["ground_truth"].append(item["ground_truth"])

    return results_pkg

def _save_benchmark_report(baseline_score: Any, hybrid_score: Any, benchmarks_dir: str) -> None:
    """Consolidate scores into CSV and JSON reports for auditability."""
    import pandas as pd
    os.makedirs(benchmarks_dir, exist_ok=True)

    baseline_score.to_pandas().to_csv(os.path.join(benchmarks_dir, "baseline_detailed_report.csv"), index=False)
    hybrid_score.to_pandas().to_csv(os.path.join(benchmarks_dir, "hybrid_detailed_report.csv"), index=False)

    summary = {
        "baseline": {
            "faithfulness": baseline_score.get("faithfulness", 0),
            "answer_relevancy": baseline_score.get("answer_relevancy", 0),
        },
        "hybrid": {
            "faithfulness": hybrid_score.get("faithfulness", 0),
            "answer_relevancy": hybrid_score.get("answer_relevancy", 0),
        },
        "improvements": {
            "faithfulness_pct": ((hybrid_score.get("faithfulness", 0) / (baseline_score.get("faithfulness", 0) or 1)) - 1) * 100,
            "answer_relevancy_pct": ((hybrid_score.get("answer_relevancy", 0) / (baseline_score.get("answer_relevancy", 0) or 1)) - 1) * 100,
        },
    }

    with open(os.path.join(benchmarks_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Benchmark artifacts successfully exported to {benchmarks_dir}/")

async def main() -> None:
    """Main orchestration for the A/B evaluation suite."""
    load_dotenv()
    settings = get_settings()
    
    # 1. Initialize evaluation models
    gemini_llm = ChatGoogleGenerativeAI(model=settings.google_model)
    bge_embeddings = HuggingFaceEmbeddings(model_name=settings.text_embedding_model or "BAAI/bge-m3")

    # 2. Configure RAGAS runtime for robust Free Tier operation
    from ragas.run_config import RunConfig
    eval_config = RunConfig(timeout=120, max_retries=10, max_wait=120, max_workers=1)

    eval_questions = load_manual_dataset()
    if not eval_questions:
        return

    # 3. Execution Phase
    logger.info("Phase 1: Baseline Evaluation...")
    baseline_data = await _evaluate_pipeline(eval_questions, VectorRetriever(), label="Baseline")
    baseline_score = evaluate(
        Dataset.from_dict(baseline_data),
        metrics=[faithfulness, answer_relevancy],
        llm=gemini_llm, 
        embeddings=bge_embeddings,
        run_config=eval_config
    )

    logger.info("Phase 2: Hybrid Evaluation...")
    hybrid_data = await _evaluate_pipeline(eval_questions, HybridRetriever(), label="Hybrid")
    hybrid_score = evaluate(
        Dataset.from_dict(hybrid_data),
        metrics=[faithfulness, answer_relevancy],
        llm=gemini_llm,
        embeddings=bge_embeddings,
        run_config=eval_config
    )

    # 4. Persistence Phase
    benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks")
    _save_benchmark_report(baseline_score, hybrid_score, benchmarks_dir)

if __name__ == "__main__":
    asyncio.run(main())