"""Retrieval evaluation: Recall@k, MRR, nDCG@k (report objective 5).

Compares four strategies over a labeled query set:
  bm25 | dense | hybrid (fused) | hybrid+rerank

Eval file format (JSONL), one query per line:
  {"query": "...", "relevant": ["2503.23013", "2402.01767"]}

Usage:
  python -m evaluation.evaluate --eval-file evaluation/datasets/sample_eval.jsonl \
      --k 1 5 10
Papers referenced by the eval set must already be ingested (scripts/ingest.py).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from backend.app.services.pipeline import get_pipeline
from backend.app.services.query_processor import process_query


# ------------------------------------------------------------------ metrics
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, pid in enumerate(ranked, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, pid in enumerate(ranked[:k], start=1)
        if pid in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------- strategies
def run_strategy(pipe, strategy: str, query: str, k_max: int) -> list[str]:
    pq = process_query(query)
    if strategy == "bm25":
        return [pid for pid, _ in pipe.bm25.retrieve(pq.sparse_query, top_k=k_max)]
    if strategy == "dense":
        return [pid for pid, _ in pipe.dense.retrieve(pq.dense_query, top_k=k_max)]
    if strategy == "hybrid":
        return [pid for pid, _ in pipe.hybrid.retrieve(pq, top_k=k_max)]
    if strategy == "hybrid+rerank":
        ranked = pipe._retrieve_and_rerank(pq, top_k=k_max)
        return [r["arxiv_id"] for r in ranked]
    raise ValueError(strategy)


def evaluate(eval_file: Path, ks: list[int]) -> dict:
    pipe = get_pipeline()
    rows = [json.loads(line) for line in eval_file.read_text().splitlines() if line.strip()]
    k_max = max(ks + [10])
    strategies = ["bm25", "dense", "hybrid", "hybrid+rerank"]
    report: dict = {s: {f"recall@{k}": 0.0 for k in ks} for s in strategies}
    for s in strategies:
        report[s]["mrr"] = 0.0
        report[s].update({f"ndcg@{k}": 0.0 for k in ks})

    for row in rows:
        relevant = set(row["relevant"])
        for s in strategies:
            ranked = run_strategy(pipe, s, row["query"], k_max)
            report[s]["mrr"] += mrr(ranked, relevant)
            for k in ks:
                report[s][f"recall@{k}"] += recall_at_k(ranked, relevant, k)
                report[s][f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)

    n = max(1, len(rows))
    for s in strategies:
        for metric in report[s]:
            report[s][metric] = round(report[s][metric] / n, 4)
    report["_n_queries"] = len(rows)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", type=Path,
                    default=Path("evaluation/datasets/sample_eval.jsonl"))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()

    report = evaluate(args.eval_file, args.k)
    n = report.pop("_n_queries")
    print(f"\nRetrieval evaluation over {n} queries\n" + "=" * 64)
    metrics = list(next(iter(report.values())).keys())
    header = f"{'strategy':<16}" + "".join(f"{m:>12}" for m in metrics)
    print(header)
    print("-" * len(header))
    for s, vals in report.items():
        print(f"{s:<16}" + "".join(f"{vals[m]:>12.4f}" for m in metrics))
    print()


if __name__ == "__main__":
    main()
