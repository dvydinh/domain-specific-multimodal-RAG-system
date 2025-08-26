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

# --- MONKEY PATCH BẮT BUỘC ĐỂ SỬA LỖI TEMPERATURE CỦA GEMINI VỚI RAGAS ---
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
original_agenerate = ChatGoogleGenerativeAI._agenerate
async def patched_agenerate(self, *args, **kwargs):
    # Lấy thông số từ đối tượng, loại bỏ temperature nếu nó bằng 0 hoặc không hợp lệ
    gen_kwargs = self._get_ls_params() if hasattr(self, '_get_ls_params') else {}
    if "temperature" in gen_kwargs:
        del gen_kwargs["temperature"]
    # Ép kwargs của hàm không được chứa temperature
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
# Dataset Generation
# ================================================================

def generate_synthetic_dataset(
    n: int,
    llm: ChatGoogleGenerativeAI,
    embeddings: HuggingFaceEmbeddings,
) -> list[dict]:
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sample",
    )
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return []

    docs: list[Document] = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for recipe in data:
                content = f"Recipe: {recipe.get('recipe_name', '')}\n"
                content += f"Cuisine: {recipe.get('cuisine', '')}\n"
                content += f"Tags: {', '.join(recipe.get('tags', []))}\n"

                ingredients = recipe.get("ingredients", [])
                if ingredients:
                    content += "Ingredients:\n"
                    for ing in ingredients:
                        content += f"- {ing.get('quantity', '')} {ing.get('name', '')}\n"

                instructions = recipe.get("instructions", "")
                if isinstance(instructions, list):
                    instructions = " ".join(instructions)
                content += f"Instructions: {instructions}\n"

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "recipe_name": recipe.get("recipe_name", ""),
                    },
                ))

    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=embeddings,
    )

    logger.info(f"Generating {n} synthetic evaluation questions using RAGAS...")
    testset = generator.generate_with_langchain_docs(
        docs,
        test_size=n,
        distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2},
    )

    df = testset.to_pandas()
    dataset = [
        {"question": row["question"], "ground_truth": row["ground_truth"]}
        for _, row in df.iterrows()
    ]

    logger.info(f"Generated {len(dataset)} evaluation questions")
    return dataset


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
    logger.info("Starting A/B Comparative Evaluation: Pure Vector vs Hybrid")
    logger.info("=" * 60)

    settings = get_settings()

    if not settings.google_api_key:
        logger.error("GOOGLE_API_KEY is not set. Cannot run evaluation.")
        return

    logger.info(f"Evaluator/Generator LLM: {settings.google_model}")
    logger.info(f"Embeddings: {settings.text_embedding_model}")

    gemini_llm = ChatGoogleGenerativeAI(
        model=settings.google_model,
        max_retries=2,
    )
    bge_embeddings = HuggingFaceEmbeddings(model_name=settings.text_embedding_model)

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        ragas_emb = LangchainEmbeddingsWrapper(bge_embeddings)
    except ImportError:
        ragas_llm = gemini_llm
        ragas_emb = bge_embeddings

    # DÙNG GEMINI ĐỂ GENERATE DATASET LUÔN
    eval_questions = generate_synthetic_dataset(
        n=settings.eval_test_size,
        llm=gemini_llm, 
        embeddings=bge_embeddings,
    )

    if not eval_questions:
        logger.error("No evaluation questions generated. Aborting.")
        return

    logger.info("[1] Evaluating Pure Vector Baseline...")
    baseline_data = await _evaluate_pipeline(
        eval_questions, VectorRetriever(), label="Baseline"
    )
    baseline_dataset = Dataset.from_dict(baseline_data)

    baseline_score = evaluate(
        baseline_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    logger.info("[2] Evaluating Hybrid RAG Architecture...")
    hybrid_data = await _evaluate_pipeline(
        eval_questions, HybridRetriever(), label="Hybrid"
    )
    hybrid_dataset = Dataset.from_dict(hybrid_data)

    hybrid_score = evaluate(
        hybrid_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    logger.info("Comparative Benchmark Complete:")
    logger.info(
        f"Baseline -> Faithfulness: {baseline_score.get('faithfulness', 0):.4f}, "
        f"Answer Relevance: {baseline_score.get('answer_relevancy', 0):.4f}"
    )
    logger.info(
        f"Hybrid   -> Faithfulness: {hybrid_score.get('faithfulness', 0):.4f}, "
        f"Answer Relevance: {hybrid_score.get('answer_relevancy', 0):.4f}"
    )

    benchmarks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmarks",
    )
    _save_benchmark_report(baseline_score, hybrid_score, benchmarks_dir)


if __name__ == "__main__":
    asyncio.run(main())