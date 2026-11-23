"""
tests/test_eval.py
==================
Offline unit and integration tests for the evaluation harness — Phase 4.

Guarantees:
- 100% offline — zero Ollama server, HuggingFace, or external network calls.
- Zero API cost ($0) — uses mock embedders and rerankers.
- Runs full benchmark evaluation in < 2 seconds.
- Full coverage of:
    * Metric calculations (Hit Rate@K, MRR, Citation Precision, Abstention Accuracy, Mean Latency)
    * Golden dataset schema validation (10 cases, 7 in-domain, 3 distractors)
    * End-to-end benchmark execution in offline mode with temporary storage

Run with:
    pytest tests/test_eval.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.benchmark import (
    compute_abstention_accuracy,
    compute_citation_precision,
    compute_hit_rate_at_k,
    compute_mean_latency_ms,
    compute_mrr,
    run_benchmark,
)


# ===========================================================================
# 1. Test Evaluation Metric Calculations
# ===========================================================================


class TestMetricCalculations:

    def test_compute_hit_rate_at_k(self) -> None:
        """Hit Rate@K computes the fraction of in-domain queries with a hit."""
        results = [
            {"expected_abstain": False, "hit": True},
            {"expected_abstain": False, "hit": False},
            {"expected_abstain": False, "hit": True},
            {"expected_abstain": False, "hit": True},
            {"expected_abstain": True, "hit": False},  # Distractor ignored
        ]
        # 3 hits out of 4 in-domain = 0.75
        assert compute_hit_rate_at_k(results) == pytest.approx(0.75)

    def test_compute_mrr(self) -> None:
        """MRR computes the average reciprocal rank of the first hit."""
        results = [
            {"expected_abstain": False, "reciprocal_rank": 1.0},   # rank 1
            {"expected_abstain": False, "reciprocal_rank": 0.5},   # rank 2
            {"expected_abstain": False, "reciprocal_rank": 0.0},   # miss
            {"expected_abstain": True, "reciprocal_rank": 0.0},    # Distractor ignored
        ]
        # (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert compute_mrr(results) == pytest.approx(0.50)

    def test_compute_citation_precision(self) -> None:
        """Citation Precision computes correct_citations / total_citations."""
        results = [
            {"expected_abstain": False, "num_citations": 2, "correct_citations": 2},
            {"expected_abstain": False, "num_citations": 2, "correct_citations": 1},
            {"expected_abstain": True, "num_citations": 0, "correct_citations": 0},
        ]
        # (2 + 1) / (2 + 2) = 3 / 4 = 0.75
        assert compute_citation_precision(results) == pytest.approx(0.75)

    def test_compute_abstention_accuracy(self) -> None:
        """Abstention Accuracy computes the fraction of distractors refused."""
        results = [
            {"expected_abstain": True, "abstained": True},
            {"expected_abstain": True, "abstained": True},
            {"expected_abstain": True, "abstained": False},  # missed refusal
            {"expected_abstain": False, "abstained": False}, # In-domain ignored
        ]
        # 2 out of 3 = 0.6667
        assert compute_abstention_accuracy(results) == pytest.approx(0.6667, abs=1e-3)

    def test_compute_mean_latency_ms(self) -> None:
        """Mean latency across all queries."""
        results = [
            {"latency_ms": 100.0},
            {"latency_ms": 200.0},
            {"latency_ms": 300.0},
        ]
        assert compute_mean_latency_ms(results) == pytest.approx(200.0)


# ===========================================================================
# 2. Test Golden Dataset Schema & Consistency
# ===========================================================================


class TestGoldenDataset:

    def test_golden_dataset_structure(self) -> None:
        """Verify golden_dataset.json contains 10 valid, properly structured cases."""
        golden_path = Path("eval/golden_dataset.json")
        assert golden_path.exists(), "eval/golden_dataset.json must exist."

        with golden_path.open("r", encoding="utf-8") as fh:
            cases: list[dict[str, Any]] = json.load(fh)

        assert len(cases) == 10, "Golden dataset must contain exactly 10 test cases."

        in_domain = [c for c in cases if not c.get("expected_abstain")]
        distractors = [c for c in cases if c.get("expected_abstain")]

        assert len(in_domain) == 7, "Must contain exactly 7 in-domain questions."
        assert len(distractors) == 3, "Must contain exactly 3 distractor questions."

        # Verify required keys
        for c in cases:
            assert "id" in c
            assert "question" in c
            assert "expected_abstain" in c
            if not c["expected_abstain"]:
                assert c["expected_source"] in {
                    "company_security_policy.pdf",
                    "product_specs.docx",
                    "q3_pricing_tiers.xlsx",
                }


# ===========================================================================
# 3. Test End-to-End Offline Benchmark Run
# ===========================================================================


class TestOfflineBenchmarkRun:

    def test_benchmark_offline_execution(self, tmp_path: Path) -> None:
        """
        Executes run_benchmark(offline=True) using sample_docs/ and verifies:
          - Hit Rate@K >= 0.85
          - MRR >= 0.80
          - Abstention Accuracy == 1.0 (all 3 distractors refused)
          - JSON report is written to output_path
        """
        output_report = tmp_path / "test_report.json"

        report = run_benchmark(
            golden_path="eval/golden_dataset.json",
            output_path=str(output_report),
            offline=True,
            sample_docs_dir="sample_docs",
        )

        assert report["mode"] == "offline"
        assert report["total_queries"] == 10
        assert report["in_domain_queries"] == 7
        assert report["distractor_queries"] == 3

        metrics = report["metrics"]
        assert metrics["hit_rate_at_k"] >= 0.85
        assert metrics["mean_reciprocal_rank_mrr"] >= 0.80
        assert metrics["abstention_accuracy"] == 1.0
        assert metrics["mean_latency_ms"] > 0.0

        # Verify output file was created and is valid JSON
        assert output_report.exists()
        with output_report.open("r", encoding="utf-8") as fh:
            saved_report = json.load(fh)
        assert saved_report["total_queries"] == 10
