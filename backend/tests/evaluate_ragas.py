"""
Comparative Evaluation: Vector Baseline vs Hybrid RAG.

Demonstrates the performance difference of Graph-Hard-Filtering
using a two-pipeline A/B benchmark with RAGAS metrics.

Evaluator/Generator LLM: Google Gemini 2.0 Flash (Monkey Patched)
Embeddings: BAAI/bge-m3 (local)
"""

import asyncio
import json
import glob
import logging
import os
import sys

# --- MONKEY PATCH FIX FOR GEMINI TEMPERATURE WITH RAGAS ---
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
original_agenerate = ChatGoogleGenerativeAI._agenerate
async def patched_agenerate(self, *args, **kwargs):
    # Extract params, removing temperature if invalid/zero
    gen_kwargs = self._get_ls_params() if hasattr(self, '_get_ls_params') else {}
    if "temperature" in gen_kwargs:
        del gen_kwargs["temperature"]
    # Force kwargs to not contain temperature
    if "temperature" in kwargs:
         del kwargs["temperature"]
    return await original_agenerate(self, *args, **kwargs)
ChatGoogleGenerativeAI._agenerate = patched_agenerate
# -------------------------------------------------------------------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from backend.config import get_settings
from backend.generation.synthesizer import ResponseSynthesizer
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


# ================================================================
# Dataset
# ================================================================

def load_manual_dataset() -> list[dict]:
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sample",
    )
    file_path = os.path.join(data_dir, "manual_meat_eval.json")
    
    if not os.path.exists(file_path):
        logger.error(f"Cannot find manual dataset at {file_path}")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Successfully loaded {len(data)} manual test questions.")
    return data


# ================================================================
# Pipeline Evaluation
# ================================================================
async def _evaluate_pipeline(
    questions: list[dict],
    retriever: VectorRetriever | HybridRetriever,
    label: str,
) -> dict:
    synthesizer = ResponseSynthesizer()

    q_list: list[str] = []
    a_list: list[str] = []
    c_list: list[list[str]] = []
    g_list: list[str] = []

    for i, item in enumerate(questions):
        # Defensive Rate Limiting: 60s sleep between questions to respect 15 RPM limits
        if i > 0:
            logger.info(f"  [{label}] Rate Limit Defense: Sleeping for 60s...")
            await asyncio.sleep(60)

        query = item["question"]
        logger.info(f"  [{label}] Processing question {i + 1}/{len(questions)}")

        if isinstance(retriever, HybridRetriever):
            results = await retriever.aretrieve(query, top_k=3, include_images=False)
        else:
            results = await retriever.aretrieve_all(query, top_k_text=3, include_images=False)

        response = await synthesizer.asynthesize(query, results)
        contexts = [res.get("text", "") for res in results.get("text_results", [])]

        q_list.append(query)
        a_list.append(response.response)
        c_list.append(contexts)
        g_list.append(item["ground_truth"])

    return {"question": q_list, "answer": a_list, "contexts": c_list, "ground_truth": g_list}

# ================================================================
# Report Generation
# ================================================================

def _save_benchmark_report(
    baseline_score: dict,
    hybrid_score: dict,
    benchmarks_dir: str,
) -> None:
    import pandas as pd

    os.makedirs(benchmarks_dir, exist_ok=True)

    baseline_df = baseline_score.to_pandas()
    hybrid_df = hybrid_score.to_pandas()

    baseline_df.to_csv(os.path.join(benchmarks_dir, "baseline_detailed_report.csv"), index=False)
    hybrid_df.to_csv(os.path.join(benchmarks_dir, "hybrid_detailed_report.csv"), index=False)

    baseline_faith = baseline_score.get("faithfulness", 0)
    baseline_relevancy = baseline_score.get("answer_relevancy", 0)
    hybrid_faith = hybrid_score.get("faithfulness", 0)
    hybrid_relevancy = hybrid_score.get("answer_relevancy", 0)

    summary_data = {
        "baseline": {
            "faithfulness": baseline_faith,
            "answer_relevancy": baseline_relevancy,
        },
        "hybrid": {
            "faithfulness": hybrid_faith,
            "answer_relevancy": hybrid_relevancy,
        },
        "improvements": {
            "faithfulness_pct": ((hybrid_faith / (baseline_faith or 1)) - 1) * 100,
            "answer_relevancy_pct": ((hybrid_relevancy / (baseline_relevancy or 1)) - 1) * 100,
        },
    }

    with open(os.path.join(benchmarks_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    logger.info(f"Audit trail saved to {benchmarks_dir}/")


# ================================================================
# Main Entry Point
# ================================================================

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Starting MANUAL Meat Evaluation (Beef & Chicken)")
    logger.info("=" * 60)

    settings = get_settings()
    gemini_llm = ChatGoogleGenerativeAI(model=settings.google_model, max_retries=2)
    bge_embeddings = HuggingFaceEmbeddings(model_name=settings.text_embedding_model)

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        ragas_emb = LangchainEmbeddingsWrapper(bge_embeddings)
    except ImportError:
        ragas_llm = gemini_llm
        ragas_emb = bge_embeddings

    # RATE LIMITER CONFIG: Optimized for Google AI Free Tier (15 RPM)
    from ragas.run_config import RunConfig
    eval_config = RunConfig(timeout=120, max_retries=10, max_wait=120, max_workers=1)

    # 1. LOAD MANUAL DATASET
    eval_questions = load_manual_dataset()

    if not eval_questions:
        logger.error("No evaluation questions found. Aborting.")
        return

    # 2. EVALUATE BASELINE
    logger.info("[1] Evaluating Pure Vector Baseline...")
    baseline_data = await _evaluate_pipeline(eval_questions, VectorRetriever(), label="Baseline")
    baseline_dataset = Dataset.from_dict(baseline_data)
    baseline_score = evaluate(
        baseline_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=eval_config,
        raise_exceptions=False
    )

    # 3. EVALUATE HYBRID
    logger.info("[2] Evaluating Hybrid RAG Architecture...")
    hybrid_data = await _evaluate_pipeline(eval_questions, HybridRetriever(), label="Hybrid")
    hybrid_dataset = Dataset.from_dict(hybrid_data)
    hybrid_score = evaluate(
        hybrid_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=eval_config,
        raise_exceptions=False
    )

    # 4. PRINT RESULTS AND SAVE REPORT
    logger.info("Comparative Benchmark Complete:")
    logger.info(f"Baseline -> Faithfulness: {baseline_score.get('faithfulness', 0):.4f}, Relevancy: {baseline_score.get('answer_relevancy', 0):.4f}")
    logger.info(f"Hybrid   -> Faithfulness: {hybrid_score.get('faithfulness', 0):.4f}, Relevancy: {hybrid_score.get('answer_relevancy', 0):.4f}")

    benchmarks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmarks"
    )
    _save_benchmark_report(baseline_score, hybrid_score, benchmarks_dir)


if __name__ == "__main__":
    asyncio.run(main())