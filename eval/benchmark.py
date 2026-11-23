"""
eval/benchmark.py
=================
Automated RAG Evaluation Harness for the Multi-Document Conversational RAG Engine.

Evaluates 10 golden benchmark cases:
  - 7 in-domain questions (spanning PDF, DOCX, XLSX)
  - 3 out-of-scope distractor questions

Metrics Computed:
  1. Hit Rate@K (percentage of in-domain queries retrieving expected_source in top-K)
  2. Mean Reciprocal Rank (MRR)
  3. Citation Precision (fraction of citations matching expected_source)
  4. Abstention Accuracy (fraction of distractors rejected with abstained=True)
  5. Mean End-to-End Latency (ms)

CLI Usage:
  python -m eval.benchmark --offline      # Instant offline CI run (< 2s, zero network/downloads)
  python -m eval.benchmark                # Live run against local Ollama + HuggingFace models
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from core.config import Settings
from core.generate import build_llm, generate_answer
from core.ingest import BaseEmbedder, SentenceTransformerEmbedder, ingest_file
from core.memory import SQLiteMemoryStore
from core.retrieve import BaseReranker, CrossEncoderReranker, retrieve_and_rerank
from core.schemas import ChatRequest, ChatResponse


# ===========================================================================
# Metric Calculation Utilities
# ===========================================================================


def compute_hit_rate_at_k(
    eval_results: list[dict[str, Any]],
) -> float:
    """
    Hit Rate@K: fraction of in-domain queries where expected_source was retrieved in top-K.
    """
    in_domain = [r for r in eval_results if not r["expected_abstain"]]
    if not in_domain:
        return 0.0
    hits = sum(1 for r in in_domain if r["hit"])
    return round(hits / len(in_domain), 4)


def compute_mrr(
    eval_results: list[dict[str, Any]],
) -> float:
    """
    Mean Reciprocal Rank (MRR): average of (1 / first_matching_rank) across in-domain queries.
    """
    in_domain = [r for r in eval_results if not r["expected_abstain"]]
    if not in_domain:
        return 0.0
    reciprocal_ranks = [r["reciprocal_rank"] for r in in_domain]
    return round(sum(reciprocal_ranks) / len(in_domain), 4)


def compute_citation_precision(
    eval_results: list[dict[str, Any]],
) -> float:
    """
    Citation Precision: total citations matching expected_source / total citations returned.
    """
    in_domain = [r for r in eval_results if not r["expected_abstain"]]
    total_citations = sum(r["num_citations"] for r in in_domain)
    if total_citations == 0:
        return 0.0
    correct_citations = sum(r["correct_citations"] for r in in_domain)
    return round(correct_citations / total_citations, 4)


def compute_abstention_accuracy(
    eval_results: list[dict[str, Any]],
) -> float:
    """
    Abstention Accuracy: fraction of out-of-scope queries correctly refused with abstained=True.
    """
    distractors = [r for r in eval_results if r["expected_abstain"]]
    if not distractors:
        return 0.0
    correct_refusals = sum(1 for r in distractors if r["abstained"])
    return round(correct_refusals / len(distractors), 4)


def compute_mean_latency_ms(
    eval_results: list[dict[str, Any]],
) -> float:
    """Mean end-to-end latency in milliseconds across all evaluated queries."""
    if not eval_results:
        return 0.0
    total = sum(r["latency_ms"] for r in eval_results)
    return round(total / len(eval_results), 2)


# ===========================================================================
# Offline Test Doubles for CI Benchmarking
# ===========================================================================


class OfflineEvalEmbedder(BaseEmbedder):
    """Deterministic offline embedder mapping tokens to 384-dim unit vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 384
            for i, ch in enumerate(text[:12]):
                vec[i % 384] += (ord(ch) % 10) / 10.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec[0] = 1.0
            results.append(vec)
        return results


class OfflineEvalReranker(BaseReranker):
    """
    Deterministic offline reranker:
      - Assigns high logits (> 1.0) when passage contains query keywords.
      - Assigns low logits (< -4.0) for distractor queries (triggering abstention).
    """

    DISTRACTOR_TERMS = {
        "world cup", "soccer", "1998", "chocolate", "lava cake",
        "molten", "fudge", "nobel", "physics", "1921",
    }

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, passage in pairs:
            q_lower = query.lower()
            p_lower = passage.lower()

            # Check if this is a known out-of-scope distractor query
            if any(term in q_lower for term in self.DISTRACTOR_TERMS):
                scores.append(-4.5)
                continue

            # In-domain match: check token overlap
            q_tokens = set(q_lower.split())
            p_tokens = set(p_lower.split())
            overlap = len(q_tokens.intersection(p_tokens))

            if overlap >= 2:
                scores.append(2.5)
            elif overlap == 1:
                scores.append(0.5)
            else:
                scores.append(-3.0)

        return scores


class OfflineEvalLLM(Runnable):
    """Deterministic offline LLM returning grounded answers citing [Source 1]."""

    def invoke(self, input: Any, config: Any = None) -> str:
        return "Based on [Source 1], the specified requirement is verified in the documentation."


# ===========================================================================
# Benchmark Runner Core
# ===========================================================================


def evaluate_single_case(
    case: dict[str, Any],
    settings: Settings,
    embedder: BaseEmbedder,
    reranker: BaseReranker,
    llm: Runnable,
    memory_store: SQLiteMemoryStore,
    top_k: int = 4,
) -> dict[str, Any]:
    """Execute a single evaluation case and record metrics."""
    req = ChatRequest(
        session_id=f"eval-session-{case['id']}",
        question=case["question"],
        top_k=top_k,
    )

    resp: ChatResponse = generate_answer(
        request=req,
        settings=settings,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        memory_store=memory_store,
    )

    expected_src = case.get("expected_source")
    expected_abstain = case.get("expected_abstain", False)

    # Retrieval evaluation: run retrieve_and_rerank to inspect raw ranks
    passages, abstained = retrieve_and_rerank(
        query=resp.standalone_query,
        settings=settings,
        top_k=top_k,
        embedder=embedder,
        reranker=reranker,
    )

    hit = False
    reciprocal_rank = 0.0
    retrieved_sources = [p.chunk.source_filename for p in passages]

    if not expected_abstain and expected_src:
        for rank_idx, src in enumerate(retrieved_sources, start=1):
            if src == expected_src:
                hit = True
                reciprocal_rank = 1.0 / rank_idx
                break

    # Citation evaluation
    num_citations = len(resp.citations)
    correct_citations = 0
    if not expected_abstain and expected_src:
        correct_citations = sum(
            1 for c in resp.citations if c.source_filename == expected_src
        )

    return {
        "id": case["id"],
        "category": case.get("category", "general"),
        "question": case["question"],
        "expected_source": expected_src,
        "expected_abstain": expected_abstain,
        "abstained": resp.abstained,
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "num_citations": num_citations,
        "correct_citations": correct_citations,
        "retrieved_sources": retrieved_sources,
        "latency_ms": resp.latency_ms,
        "answer_snippet": resp.answer[:100],
    }


def run_benchmark(
    golden_path: str | Path = "eval/golden_dataset.json",
    output_path: str | Path = "eval/last_report.json",
    offline: bool = False,
    sample_docs_dir: str | Path = "sample_docs",
) -> dict[str, Any]:
    """
    Run full evaluation harness over golden dataset.

    Safeguard 1: In offline mode, runs inside tempfile.TemporaryDirectory()
    so dummy vectors never pollute production ./chroma_db or ./rag_memory.sqlite.
    """
    golden_file = Path(golden_path)
    if not golden_file.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_file}")

    with golden_file.open("r", encoding="utf-8") as fh:
        dataset: list[dict[str, Any]] = json.load(fh)

    sample_dir = Path(sample_docs_dir)

    # Temporary directory guard for offline CI runs
    if offline:
        tmp_dir = tempfile.TemporaryDirectory()
        base_path = Path(tmp_dir.name)
        settings = Settings(
            chroma_persist_dir=str(base_path / "chroma_db"),
            bm25_index_path=str(base_path / "chroma_db" / "bm25_corpus.json"),
            sqlite_db_path=str(base_path / "rag_memory.sqlite"),
            rerank_threshold=-1.5,
        )
        embedder: BaseEmbedder = OfflineEvalEmbedder()
        reranker: BaseReranker = OfflineEvalReranker()
        llm: Runnable = OfflineEvalLLM()
    else:
        tmp_dir = None
        settings = Settings()
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        reranker = CrossEncoderReranker(settings.reranker_model)
        llm = build_llm(settings)

    memory_store = SQLiteMemoryStore(settings.sqlite_db_path)

    try:
        # Ingest sample documents into the evaluation index
        sample_files = [
            sample_dir / "company_security_policy.pdf",
            sample_dir / "product_specs.docx",
            sample_dir / "q3_pricing_tiers.xlsx",
        ]
        for sfile in sample_files:
            if sfile.exists():
                ingest_file(path=sfile, settings=settings, embedder=embedder)

        case_results: list[dict[str, Any]] = []
        for case in dataset:
            result = evaluate_single_case(
                case=case,
                settings=settings,
                embedder=embedder,
                reranker=reranker,
                llm=llm,
                memory_store=memory_store,
                top_k=4,
            )
            case_results.append(result)

        # Aggregate metrics
        hit_rate = compute_hit_rate_at_k(case_results)
        mrr = compute_mrr(case_results)
        citation_precision = compute_citation_precision(case_results)
        abstention_accuracy = compute_abstention_accuracy(case_results)
        mean_latency = compute_mean_latency_ms(case_results)

        report: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "offline" if offline else "live",
            "total_queries": len(dataset),
            "in_domain_queries": sum(1 for c in dataset if not c.get("expected_abstain")),
            "distractor_queries": sum(1 for c in dataset if c.get("expected_abstain")),
            "metrics": {
                "hit_rate_at_k": hit_rate,
                "mean_reciprocal_rank_mrr": mrr,
                "citation_precision": citation_precision,
                "abstention_accuracy": abstention_accuracy,
                "mean_latency_ms": mean_latency,
            },
            "cases": case_results,
        }

        # Write last_report.json
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        return report

    finally:
        if tmp_dir is not None:
            try:
                tmp_dir.cleanup()
            except Exception:
                # On Windows, ChromaDB's memory-mapped binary files hold handles
                # until process exit; ignore cleanup failure.
                pass


# ===========================================================================
# CLI Entrypoint & Report Printer
# ===========================================================================


def print_report_table(report: dict[str, Any]) -> None:
    """Print a clean ASCII summary table of benchmark results."""
    m = report["metrics"]
    print("\n" + "=" * 65)
    print("       RAG ENGINE AUTOMATED BENCHMARK EVALUATION REPORT       ")
    print("=" * 65)
    print(f" Mode:                   {report['mode'].upper()}")
    print(f" Total Test Cases:       {report['total_queries']} ({report['in_domain_queries']} in-domain, {report['distractor_queries']} distractors)")
    print("-" * 65)
    print(f" Hit Rate@4:             {m['hit_rate_at_k'] * 100:.1f}%")
    print(f" Mean Reciprocal Rank:   {m['mean_reciprocal_rank_mrr']:.4f}")
    print(f" Citation Precision:     {m['citation_precision'] * 100:.1f}%")
    print(f" Abstention Accuracy:    {m['abstention_accuracy'] * 100:.1f}%")
    print(f" Mean End-to-End Latency:{m['mean_latency_ms']:.2f} ms")
    print("=" * 65)

    print("\nPer-Query Breakdown:")
    for c in report["cases"]:
        status = "REFUSED" if c["abstained"] else ("HIT" if c["hit"] else "MISS")
        cat = c["category"].ljust(12)
        q_snippet = (c["question"][:38] + "...").ljust(42)
        print(f"  [{c['id']}] {cat} | {q_snippet} | {status} ({c['latency_ms']:.1f}ms)")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Engine Benchmark Evaluation")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline evaluation using deterministic mock models (fast CI verification)",
    )
    parser.add_argument(
        "--golden-path",
        default="eval/golden_dataset.json",
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--output-path",
        default="eval/last_report.json",
        help="Output report destination path",
    )
    args = parser.parse_args()

    report = run_benchmark(
        golden_path=args.golden_path,
        output_path=args.output_path,
        offline=args.offline,
    )
    print_report_table(report)


if __name__ == "__main__":
    main()
