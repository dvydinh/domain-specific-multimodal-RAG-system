"""
Custom A/B Evaluation: Vector Baseline vs Hybrid RAG.

Implements RAGAS-equivalent metrics (Faithfulness, Answer Relevancy) using
native LLM-as-Judge prompts instead of the RAGAS framework, avoiding
structured output / AFC compatibility issues with Gemma models.

Evaluator LLM: gemma-4-31b-it (via google-genai SDK, not LangChain)
Embeddings:    BAAI/bge-small-en-v1.5 (local, via sentence-transformers)

Methodology:
  - Faithfulness: Decompose answer into atomic claims, verify each against context.
  - Answer Relevancy: Generate synthetic questions from answer, measure cosine
    similarity against original question using the embedding model.

Rate Limiting: 5s sleep between every LLM call (safe for 15 RPM free tier).
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.generation.synthesizer import ResponseSynthesizer
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_retriever import VectorRetriever
from backend.utils.json_parser import extract_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limited LLM call wrapper (Google GenAI SDK — no LangChain, no AFC)
# ---------------------------------------------------------------------------

# Global timestamp tracker for rate limiting
_last_call_time = 0.0
RATE_LIMIT_SLEEP = 5.0  # seconds between calls (safe for 15 RPM)


def _call_llm(model: genai.GenerativeModel, prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemma 4 via google-genai SDK with rate limiting and retry logic.

    Uses plain text generation (NOT structured output / function calling)
    to avoid AFC compatibility issues entirely.
    """
    global _last_call_time

    for attempt in range(max_retries):
        # Rate limiter: ensure at least RATE_LIMIT_SLEEP seconds between calls
        elapsed = time.time() - _last_call_time
        if elapsed < RATE_LIMIT_SLEEP:
            sleep_time = RATE_LIMIT_SLEEP - elapsed
            logger.debug(f"Rate limiter: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        try:
            _last_call_time = time.time()
            response = model.generate_content(prompt)

            # Extract text from response parts
            if response.candidates and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                text_parts = []
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                return "\n".join(text_parts).strip()

            logger.warning(f"Empty response on attempt {attempt + 1}")
            return ""

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limit (429). Waiting {wait}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
            elif "500" in error_str or "503" in error_str:
                wait = 10 * (attempt + 1)
                logger.warning(f"Server error ({error_str[:40]}). Waiting {wait}s before retry")
                time.sleep(wait)
            else:
                logger.error(f"LLM call failed: {e}")
                if attempt == max_retries - 1:
                    return ""
                time.sleep(5)

    return ""


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness (LLM-as-Judge)
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """Given the following answer, extract all distinct factual claims as a JSON list of strings.
Only extract claims that state specific facts. Ignore filler phrases like "Based on the provided documents".
If the answer says it cannot find information or is a refusal, return an empty list [].

Answer: {answer}

Respond with ONLY a valid JSON array of strings, nothing else. Example:
["claim 1", "claim 2"]"""

CLAIM_VERIFICATION_PROMPT = """Given the following context and a single claim, determine if the claim is supported by the context.
Respond with ONLY a JSON object with two fields:
- "verdict": "yes" if the claim is supported, "no" if not supported
- "reason": brief explanation

Context:
{context}

Claim: {claim}

Respond with ONLY valid JSON. Example:
{{"verdict": "yes", "reason": "The context mentions this explicitly."}}"""


def score_faithfulness(model: genai.GenerativeModel, answer: str, contexts: list[str]) -> float:
    """
    Faithfulness score: fraction of claims in the answer that are supported by context.

    Implements the same logic as RAGAS Faithfulness:
      1. Extract atomic claims from the answer
      2. Verify each claim against the provided contexts
      3. Score = supported_claims / total_claims
    """
    # Skip refusal answers
    if "cannot find this information" in answer.lower() or not answer.strip():
        return 1.0  # Refusals are perfectly faithful (no hallucination)

    # Step 1: Extract claims
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    raw_claims = _call_llm(model, prompt)

    try:
        # Try to parse as JSON list
        claims = json.loads(raw_claims) if raw_claims.strip() else []
        if not isinstance(claims, list):
            claims = []
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        import re
        match = re.search(r'\[.*\]', raw_claims, re.DOTALL)
        if match:
            try:
                claims = json.loads(match.group(0))
            except json.JSONDecodeError:
                claims = []
        else:
            claims = []

    if not claims:
        return 1.0  # No claims to verify = faithful by default

    # Step 2: Verify each claim against context
    context_str = "\n\n".join(contexts)
    supported = 0

    for claim in claims:
        if not isinstance(claim, str) or not claim.strip():
            continue

        prompt = CLAIM_VERIFICATION_PROMPT.format(context=context_str, claim=claim)
        raw_verdict = _call_llm(model, prompt)

        verdict_data = extract_json(raw_verdict, {"verdict": "no"})
        if verdict_data.get("verdict", "").lower() == "yes":
            supported += 1
            logger.debug(f"  ✓ Supported: {claim[:60]}")
        else:
            logger.debug(f"  ✗ Not supported: {claim[:60]}")

    total = len([c for c in claims if isinstance(c, str) and c.strip()])
    score = supported / total if total > 0 else 1.0
    logger.info(f"  Faithfulness: {supported}/{total} claims supported = {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy (Embedding similarity)
# ---------------------------------------------------------------------------

QUESTION_GEN_PROMPT = """Given the following answer, generate 3 distinct questions that this answer could be responding to.
The questions should be diverse and cover different aspects of the answer.
If the answer is a refusal or says it cannot find information, generate questions about the topic the refusal addresses.

Answer: {answer}

List exactly 3 questions, one question per line. Do NOT use numbering, bullet points, markdown formatting, or JSON. Just output the plain text questions."""


def score_answer_relevancy(
    model: genai.GenerativeModel,
    embedder: SentenceTransformer,
    question: str,
    answer: str,
) -> float:
    """
    Answer Relevancy score: cosine similarity between original question and
    synthetic questions generated from the answer.

    Implements the same logic as RAGAS Answer Relevancy:
      1. Generate N synthetic questions from the answer
      2. Embed original question and synthetic questions
      3. Score = mean cosine similarity
    """
    if not answer.strip():
        return 0.0

    # Step 1: Generate synthetic questions
    prompt = QUESTION_GEN_PROMPT.format(answer=answer)
    raw_questions = _call_llm(model, prompt)

    import re
    lines = [line.strip() for line in raw_questions.split('\n')]
    questions = []
    for line in lines:
        cleaned = re.sub(r'^(\d+[\.\)]\s*|-\s*|\*\s*)', '', line).strip()
        if cleaned and len(cleaned) > 5:
            questions.append(cleaned)
    if not questions:
        return 0.0

    # Step 2: Compute cosine similarity
    original_emb = embedder.encode([question], normalize_embeddings=True)

    synth_embs = embedder.encode(questions, normalize_embeddings=True)

    similarities = np.dot(synth_embs, original_emb.T).flatten()
    score = float(np.mean(similarities))

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    logger.info(f"  Answer Relevancy: {score:.4f} (from {len(questions)} synthetic questions)")
    return score


# ---------------------------------------------------------------------------
# Pipeline evaluation
# ---------------------------------------------------------------------------


async def evaluate_pipeline(
    questions: list[dict],
    retriever: Any,
    synthesizer: ResponseSynthesizer,
    eval_model: genai.GenerativeModel,
    embedder: SentenceTransformer,
    label: str,
) -> dict:
    """
    Run a full pipeline evaluation: retrieve, synthesize, then score.

    Returns per-question scores and aggregate means.
    """
    per_question = []

    for i, item in enumerate(questions):
        query = item["question"]
        logger.info(f"[{label}] Question {i + 1}/{len(questions)}: {query[:60]}...")

        try:
            # --- Retrieval & Synthesis ---
            if retriever is None:
                # Raw LLM / No RAG Baseline
                prompt = f"Please answer the following question accurately: {query}"
                answer = _call_llm(eval_model, prompt)
                contexts = []
            else:
                # Standard RAG Pipeline
                if hasattr(retriever, 'aretrieve'):
                    results = await retriever.aretrieve(query, top_k=6, include_images=False)
                else:
                    results = await retriever.aretrieve_all(query, top_k_text=6, include_images=False)

                response = await synthesizer.asynthesize(query, results)
                answer = response.response

                # Collect contexts
                contexts = [res.get("text", "") for res in results.get("text_results", [])]
                for g_res in results.get("graph_results", []):
                    g_text = f"Recipe: {g_res.get('name')} (Cuisine: {g_res.get('cuisine', 'N/A')}). "
                    contexts.append(g_text)
                contexts = [c for c in contexts if c.strip()]

                if not contexts:
                    contexts = ["No relevant context found."]

            # --- Score (synchronous LLM calls with rate limiting) ---
            faith = score_faithfulness(eval_model, answer, contexts)
            relevancy = score_answer_relevancy(eval_model, embedder, query, answer)

            per_question.append({
                "question": query,
                "answer": answer[:200],
                "faithfulness": faith,
                "answer_relevancy": relevancy,
                "context_count": len(contexts),
                "ground_truth": item["ground_truth"],
            })

            logger.info(
                f"[{label}] Q{i+1} scored — Faith: {faith:.4f}, Rel: {relevancy:.4f}"
            )

        except Exception as e:
            logger.error(f"[{label}] Q{i+1} failed: {e}")
            per_question.append({
                "question": query,
                "answer": f"Error: {str(e)}",
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_count": 0,
                "ground_truth": item["ground_truth"],
            })

    # Aggregate
    faith_scores = [q["faithfulness"] for q in per_question]
    rel_scores = [q["answer_relevancy"] for q in per_question]

    return {
        "per_question": per_question,
        "mean_faithfulness": float(np.mean(faith_scores)) if faith_scores else 0.0,
        "mean_answer_relevancy": float(np.mean(rel_scores)) if rel_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_benchmark_report(
    baseline: dict, hybrid: dict, benchmarks_dir: str, eval_model_name: str
) -> None:
    """Save detailed CSV and summary JSON reports."""
    import pandas as pd
    os.makedirs(benchmarks_dir, exist_ok=True)

    # Detailed per-question CSVs
    pd.DataFrame(baseline["per_question"]).to_csv(
        os.path.join(benchmarks_dir, "baseline_detailed_report.csv"), index=False
    )
    pd.DataFrame(hybrid["per_question"]).to_csv(
        os.path.join(benchmarks_dir, "hybrid_detailed_report.csv"), index=False
    )

    b_f = baseline["mean_faithfulness"]
    b_r = baseline["mean_answer_relevancy"]
    h_f = hybrid["mean_faithfulness"]
    h_r = hybrid["mean_answer_relevancy"]

    summary = {
        "evaluator_model": eval_model_name,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "dataset_size": len(baseline["per_question"]),
        "baseline": {
            "faithfulness": round(b_f, 4),
            "answer_relevancy": round(b_r, 4),
        },
        "hybrid": {
            "faithfulness": round(h_f, 4),
            "answer_relevancy": round(h_r, 4),
        },
        "delta": {
            "faithfulness_abs": round(h_f - b_f, 4),
            "answer_relevancy_abs": round(h_r - b_r, 4),
            "faithfulness_pct": round(((h_f / b_f) - 1) * 100, 2) if b_f else 0,
            "answer_relevancy_pct": round(((h_r / b_r) - 1) * 100, 2) if b_r else 0,
        },
    }

    with open(os.path.join(benchmarks_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Evaluator: {eval_model_name}")
    logger.info(f"Dataset:   {len(baseline['per_question'])} questions")
    logger.info(f"Baseline — Faithfulness: {b_f:.4f}, Relevancy: {b_r:.4f}")
    logger.info(f"Hybrid   — Faithfulness: {h_f:.4f}, Relevancy: {h_r:.4f}")
    logger.info(f"Delta    — Faith: {h_f - b_f:+.4f}, Rel: {h_r - b_r:+.4f}")
    logger.info(f"Reports saved to: {benchmarks_dir}/")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def main() -> None:
    """Main A/B evaluation orchestration."""
    load_dotenv()
    settings = get_settings()

    eval_model_name = "gemma-4-31b-it"
    logger.info(f"Evaluator LLM: {eval_model_name}")
    logger.info(f"Embedding model: {settings.text_embedding_model}")

    # Initialize Google GenAI SDK directly (bypass LangChain entirely)
    genai.configure(api_key=settings.google_api_key)
    eval_model = genai.GenerativeModel(eval_model_name)

    # Local embedding model (no API calls needed)
    embedder = SentenceTransformer(settings.text_embedding_model or "BAAI/bge-small-en-v1.5")

    # Load evaluation dataset
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sample",
    )
    eval_path = os.path.join(data_dir, "manual_meat_eval.json")

    if not os.path.exists(eval_path):
        logger.error(f"Evaluation dataset not found: {eval_path}")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)

    logger.info(f"Loaded {len(eval_questions)} evaluation questions")

    # Shared synthesizer (uses production GOOGLE_MODEL from .env)
    synthesizer = ResponseSynthesizer()

    # ===== Phase 1: Baseline (Vector-Only) =====
    logger.info("=" * 60)
    logger.info("PHASE 1: Baseline Evaluation (Vector-Only)")
    logger.info("=" * 60)

    baseline = await evaluate_pipeline(
        eval_questions, VectorRetriever(), synthesizer,
        eval_model, embedder, label="Baseline",
    )

    # ===== Phase 2: Hybrid (Graph + Vector + Rerank) =====
    logger.info("=" * 60)
    logger.info("PHASE 2: Hybrid Evaluation (Graph + Vector + Rerank)")
    logger.info("=" * 60)

    hybrid = await evaluate_pipeline(
        eval_questions, HybridRetriever(), synthesizer,
        eval_model, embedder, label="Hybrid",
    )

    # ===== Save results =====
    benchmarks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmarks",
    )
    save_benchmark_report(baseline, hybrid, benchmarks_dir, eval_model_name)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
