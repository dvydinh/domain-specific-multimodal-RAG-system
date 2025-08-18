"""
Comparative Evaluation: Vector Baseline vs Hybrid RAG.
Demonstrates the performance leap of Graph-Hard-Filtering using Gemini 1.5.
"""

import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevance, faithfulness

from backend.generation.synthesizer import ResponseSynthesizer
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_retriever import VectorRetriever


def generate_synthetic_dataset(n: int = 10) -> list[dict]:
    """Generates synthetic data correctly parsing lists."""
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "sample", "eval_recipes.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dataset = []
    # Simplified empirical mapping for evaluating the 10 QA pairs locally
    for i, row in enumerate(data[:n]):
        q = f"How do I make {row.get('recipe_name', '')}?"
        ingredients = str(row.get('ingredients', '')).lower()
        if "pork" not in ingredients:
            q += " Strictly no pork."
            
        ans = row.get("instructions", "")
        if isinstance(ans, list):
            ans = " ".join(ans)
            
        dataset.append({"question": q, "ground_truth": ans})
        
    return dataset

async def agenerate_baseline_responses(questions: list[dict]) -> dict:
    retriever = VectorRetriever()
    synthesizer = ResponseSynthesizer()
    
    q_list, a_list, c_list, g_list = [], [], [], []
    for item in questions:
        query = item["question"]
        results = await retriever.aretrieve_all(query, top_k_text=3, include_images=False)
        response = await synthesizer.asynthesize(query, results)
        contexts = [res.get("text", "") for res in results.get("text_results", [])]
        
        q_list.append(query)
        a_list.append(response.response)
        c_list.append(contexts)
        g_list.append(item["ground_truth"])
        
    return {"question": q_list, "answer": a_list, "contexts": c_list, "ground_truth": g_list}

async def agenerate_responses(questions: list[dict]) -> dict:
    retriever = HybridRetriever()
    synthesizer = ResponseSynthesizer()
    
    q_list, a_list, c_list, g_list = [], [], [], []
    for item in questions:
        query = item["question"]
        results = await retriever.aretrieve(query, top_k=3, include_images=False)
        response = await synthesizer.asynthesize(query, results)
        contexts = [res.get("text", "") for res in results.get("text_results", [])]
        
        q_list.append(query)
        a_list.append(response.response)
        c_list.append(contexts)
        g_list.append(item["ground_truth"])
        
    return {"question": q_list, "answer": a_list, "contexts": c_list, "ground_truth": g_list}

async def main():
    print("=" * 60)
    print("Starting A/B Comparative Evaluation: Pure Vector vs Hybrid")
    print("=" * 60)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set. Cannot run evaluation.")
        return
        
    # Setup LLM and Embeddings wrappers for zero-cost Ragas evaluation
    print("Binding Gemini 1.5 Flash and Local BGE-M3 for evaluation...")
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # Correct Ragas Wrappers for Gemini
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        ragas_emb = LangchainEmbeddingsWrapper(bge_embeddings)
    except ImportError:
        # Fallback for older Ragas versions that accept Langchain native objects directly
        ragas_llm = gemini_llm
        ragas_emb = bge_embeddings

    eval_questions = generate_synthetic_dataset(n=10)
    
    print("\n[1] Evaluating Pure Vector Baseline...")
    baseline_data = await agenerate_baseline_responses(eval_questions)
    baseline_dataset = Dataset.from_dict(baseline_data)
    
    baseline_score = evaluate(
        baseline_dataset, 
        metrics=[faithfulness, answer_relevance],
        llm=ragas_llm, 
        embeddings=ragas_emb
    )
    
    print("\n[2] Evaluating Hybrid RAG Architecture...")
    hybrid_data = await agenerate_responses(eval_questions)
    hybrid_dataset = Dataset.from_dict(hybrid_data)
    
    hybrid_score = evaluate(
        hybrid_dataset, 
        metrics=[faithfulness, answer_relevance],
        llm=ragas_llm, 
        embeddings=ragas_emb
    )
    
    print(f"\nComparative Benchmark Complete:")
    print(f"Baseline -> Faithfulness: {baseline_score.get('faithfulness', 0):.4f}, Answer Relevance: {baseline_score.get('answer_relevance', 0):.4f}")
    print(f"Hybrid   -> Faithfulness: {hybrid_score.get('faithfulness', 0):.4f}, Answer Relevance: {hybrid_score.get('answer_relevance', 0):.4f}")

if __name__ == "__main__":
    asyncio.run(main())
