"""
Comparative Evaluation: Vector Baseline vs Hybrid RAG.

Runs a two-pipeline A/B benchmark using RAGAS metrics to measure
the impact of graph-based hard filtering on retrieval quality.

Evaluator LLM: Gemma 4 31B IT (high RPD, no rate limiting concerns)
Embeddings: BAAI/bge-small-en-v1.5 (local, via HuggingFace)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

import pydantic.v1 as pydantic_v1
sys.modules['langchain_core.pydantic_v1'] = pydantic_v1

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import get_settings
from backend.generation.synthesizer import ResponseSynthesizer
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_retriever import VectorRetriever
from backend.utils.llm_patch import apply_gemini_ragas_patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Apply compatibility patch for RAGAS evaluation
apply_gemini_ragas_patch()


def load_manual_dataset() -> list[dict]:
    """Load the gold-standard evaluation dataset."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sample",
    )
    file_path = os.path.join(data_dir, "manual_meat_eval.json")

    if not os.path.exists(file_path):
        logger.error(f"Evaluation dataset missing at {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _evaluate_pipeline(
    questions: list[dict],
    retriever: Any,
    synthesizer: ResponseSynthesizer,
    label: str,
) -> dict:
    """Run a RAG pipeline through the evaluation set and collect RAGAS-format data."""
    results_pkg = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(questions):
        query = item["question"]
        logger.info(f"  [{label}] Evaluating {i + 1}/{len(questions)}: {query[:60]}...")

        try:
            # Retrieval (polymorphic: HybridRetriever vs VectorRetriever)
            if hasattr(retriever, 'aretrieve'):
                results = await retriever.aretrieve(query, top_k=3, include_images=False)
            else:
                results = await retriever.aretrieve_all(query, top_k_text=3, include_images=False)

            # Synthesis with production synthesizer
            response = await synthesizer.asynthesize(query, results)

            # Collect contexts
            contexts = [res.get("text", "") for res in results.get("text_results", [])]
            for g_res in results.get("graph_results", []):
                g_text = f"Recipe: {g_res.get('name')} (Cuisine: {g_res.get('cuisine', 'N/A')}). "
                contexts.append(g_text)

            # Filter empty contexts
            contexts = [c for c in contexts if c.strip()]
            if not contexts:
                contexts = ["No relevant context found."]

            results_pkg["question"].append(query)
            results_pkg["answer"].append(response.response)
            results_pkg["contexts"].append(contexts)
            results_pkg["ground_truth"].append(item["ground_truth"])

        except Exception as e:
            logger.error(f"  [{label}] Question {i+1} failed: {e}")
            results_pkg["question"].append(query)
            results_pkg["answer"].append(f"Error: {str(e)}")
            results_pkg["contexts"].append(["Error during retrieval."])
            results_pkg["ground_truth"].append(item["ground_truth"])

        # Small delay between questions to be respectful to API
        await asyncio.sleep(2)

    return results_pkg


def _run_ragas_evaluation(data: dict, ragas_llm, ragas_emb, eval_config, label: str):
    """Run RAGAS evaluate() synchronously (required by RAGAS internals)."""
    logger.info(f"  Running RAGAS metrics for [{label}]...")
    dataset = Dataset.from_dict(data)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=eval_config,
    )


def _save_benchmark_report(baseline_score: Any, hybrid_score: Any, benchmarks_dir: str) -> None:
    """Save evaluation results as CSV and JSON for auditability."""
    import pandas as pd
    os.makedirs(benchmarks_dir, exist_ok=True)

    baseline_score.to_pandas().to_csv(
        os.path.join(benchmarks_dir, "baseline_detailed_report.csv"), index=False
    )
    hybrid_score.to_pandas().to_csv(
        os.path.join(benchmarks_dir, "hybrid_detailed_report.csv"), index=False
    )

    base_df = baseline_score.to_pandas()
    hybr_df = hybrid_score.to_pandas()

    b_faith = base_df["faithfulness"].mean() if "faithfulness" in base_df.columns else 0.0
    b_rel = base_df["answer_relevancy"].mean() if "answer_relevancy" in base_df.columns else 0.0
    h_faith = hybr_df["faithfulness"].mean() if "faithfulness" in hybr_df.columns else 0.0
    h_rel = hybr_df["answer_relevancy"].mean() if "answer_relevancy" in hybr_df.columns else 0.0

    summary = {
        "evaluator_model": "gemma-4-31b-it",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "dataset_size": len(baseline_score.to_pandas()),
        "baseline": {
            "faithfulness": round(b_faith, 4),
            "answer_relevancy": round(b_rel, 4),
        },
        "hybrid": {
            "faithfulness": round(h_faith, 4),
            "answer_relevancy": round(h_rel, 4),
        },
        "delta": {
            "faithfulness_abs": round(h_faith - b_faith, 4),
            "answer_relevancy_abs": round(h_rel - b_rel, 4),
            "faithfulness_pct": round(((h_faith / b_faith) - 1) * 100, 2) if b_faith else 0,
            "answer_relevancy_pct": round(((h_rel / b_rel) - 1) * 100, 2) if b_rel else 0,
        },
    }

    with open(os.path.join(benchmarks_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Benchmark artifacts exported to {benchmarks_dir}/")
    logger.info(f"  Baseline — Faithfulness: {b_faith:.4f}, Relevancy: {b_rel:.4f}")
    logger.info(f"  Hybrid   — Faithfulness: {h_faith:.4f}, Relevancy: {h_rel:.4f}")


async def main() -> None:
    """Main A/B evaluation orchestration."""
    load_dotenv()
    settings = get_settings()

    # Use Gemma 4 31B IT for evaluation (unlimited RPM/TPM as reported by user)
    eval_model = "gemma-4-31b-it"
    logger.info(f"Evaluator LLM: {eval_model}")
    logger.info(f"Embedding model: {settings.text_embedding_model}")

    gemini_llm = ChatGoogleGenerativeAI(
        model=eval_model,
        api_key=settings.google_api_key,
        temperature=0.0,
    )
    bge_embeddings = HuggingFaceEmbeddings(
        model_name=settings.text_embedding_model or "BAAI/bge-small-en-v1.5"
    )

    # RAGAS wrapper for compatibility
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        ragas_emb = LangchainEmbeddingsWrapper(bge_embeddings)
    except ImportError:
        ragas_llm = gemini_llm
        ragas_emb = bge_embeddings

    # RAGAS runtime config
    from ragas.run_config import RunConfig
    eval_config = RunConfig(timeout=120, max_retries=10, max_wait=120, max_workers=1)

    eval_questions = load_manual_dataset()
    if not eval_questions:
        logger.error("No evaluation questions loaded. Aborting.")
        return

    logger.info(f"Loaded {len(eval_questions)} evaluation questions")

    # Create shared synthesizer (uses production model for answer generation)
    synthesizer = ResponseSynthesizer()

    # Phase 1: Baseline (pure vector search, no graph filtering)
    logger.info("=" * 60)
    logger.info("Phase 1: Baseline Evaluation (Vector-Only)")
    logger.info("=" * 60)
    baseline_data = await _evaluate_pipeline(
        eval_questions, VectorRetriever(), synthesizer, label="Baseline"
    )
    baseline_score = _run_ragas_evaluation(
        baseline_data, ragas_llm, ragas_emb, eval_config, label="Baseline"
    )

    # Phase 2: Hybrid (graph + vector + reranking)
    logger.info("=" * 60)
    logger.info("Phase 2: Hybrid Evaluation (Graph + Vector + Rerank)")
    logger.info("=" * 60)
    hybrid_data = await _evaluate_pipeline(
        eval_questions, HybridRetriever(), synthesizer, label="Hybrid"
    )
    hybrid_score = _run_ragas_evaluation(
        hybrid_data, ragas_llm, ragas_emb, eval_config, label="Hybrid"
    )

    # Save results
    benchmarks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmarks"
    )
    _save_benchmark_report(baseline_score, hybrid_score, benchmarks_dir)


if __name__ == "__main__":
    asyncio.run(main())