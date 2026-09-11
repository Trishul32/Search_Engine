"""Bulk metadata ingestion.

Two sources, matching the Review-2 'Datasets' slides:
  1. Live arXiv API by query/category (real-time corpus building).
  2. The Kaggle arXiv metadata snapshot
     (https://www.kaggle.com/datasets/Cornell-University/arxiv) —
     a JSONL file with one paper per line.

Only metadata is ingested; PDFs stay lazy.

Usage:
  python -m scripts.ingest --arxiv-query "retrieval augmented generation" --max 200
  python -m scripts.ingest --categories cs.IR cs.CL --max 300
  python -m scripts.ingest --kaggle-json arxiv-metadata-oai-snapshot.json --max 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.arxiv_client import PaperMeta, search_arxiv
from backend.app.services.pipeline import get_pipeline


def from_kaggle(path: Path, limit: int, categories: list[str] | None) -> list[PaperMeta]:
    metas: list[PaperMeta] = []
    wanted = set(categories or [])
    with path.open() as fh:
        for line in fh:
            if len(metas) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = (row.get("categories") or "").split()
            if wanted and not (wanted & set(cats)):
                continue
            aid = row.get("id", "")
            metas.append(
                PaperMeta(
                    arxiv_id=aid,
                    title=(row.get("title") or "").replace("\n", " ").strip(),
                    abstract=(row.get("abstract") or "").replace("\n", " ").strip(),
                    authors=row.get("authors", ""),
                    categories=",".join(cats),
                    published=(row.get("update_date") or "")[:10],
                    pdf_url=f"https://arxiv.org/pdf/{aid}",
                )
            )
    return metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-query", type=str, default="")
    ap.add_argument("--categories", nargs="*", default=[])
    ap.add_argument("--kaggle-json", type=Path, default=None)
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    pipe = get_pipeline()
    metas: list[PaperMeta] = []

    if args.kaggle_json:
        metas = from_kaggle(args.kaggle_json, args.max, args.categories)
    elif args.arxiv_query:
        metas = search_arxiv(args.arxiv_query, max_results=args.max)
    elif args.categories:
        q = " OR ".join(f"cat:{c}" for c in args.categories)
        metas = search_arxiv(q, max_results=args.max)
    else:
        ap.error("provide --arxiv-query, --categories, or --kaggle-json")

    added = pipe.ingest_metadata(metas)
    print(f"Fetched {len(metas)} papers; newly indexed {added}.")
    print(json.dumps(pipe.stats(), indent=2))


if __name__ == "__main__":
    main()
