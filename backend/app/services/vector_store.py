"""FAISS vector stores with metadata filtering and disk persistence.

Two indexes are maintained:

* ``paper``  — one vector per paper (title + abstract), used by the dense
  half of hybrid retrieval and by the recommender.
* ``chunk``  — one vector per structure-aware chunk of lazily-loaded PDFs,
  used at QA time.

Both use ``IndexFlatIP`` over L2-normalized vectors (cosine). Flat indexes
are exact and fastest below ~1M vectors; ``maybe_train_ivf`` upgrades the
chunk index to IVF once the corpus grows, matching the scaling analysis in
the report (post-filtering is used with IVF to preserve index efficiency).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_IVF_THRESHOLD = 50_000  # vectors; below this, exact flat search wins


class FaissStore:
    """id-mapped FAISS index + JSON-persisted metadata, thread-safe."""

    def __init__(self, name: str, dim: int, directory: Path):
        self.name = name
        self.dim = dim
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._meta: dict[int, dict] = {}
        self._next_id = 0
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._load()

    # ------------------------------------------------------------ storage
    @property
    def _index_path(self) -> Path:
        return self.dir / f"{self.name}.faiss"

    @property
    def _meta_path(self) -> Path:
        return self.dir / f"{self.name}.meta.json"

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                idx = faiss.read_index(str(self._index_path))
                if idx.d != self.dim:
                    logger.warning(
                        "%s index dim %d != embedder dim %d; rebuilding",
                        self.name, idx.d, self.dim,
                    )
                    return
                self.index = idx
                raw = json.loads(self._meta_path.read_text())
                self._meta = {int(k): v for k, v in raw["meta"].items()}
                self._next_id = raw["next_id"]
                logger.info("Loaded %s index (%d vectors)", self.name, self.index.ntotal)
            except Exception as exc:
                logger.warning("Failed to load %s index (%s); starting fresh", self.name, exc)

    def save(self) -> None:
        with self._lock:
            faiss.write_index(self.index, str(self._index_path))
            self._meta_path.write_text(
                json.dumps({"next_id": self._next_id,
                            "meta": {str(k): v for k, v in self._meta.items()}})
            )

    # ---------------------------------------------------------------- ops
    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> list[int]:
        assert len(vectors) == len(metadatas)
        with self._lock:
            ids = np.arange(self._next_id, self._next_id + len(vectors), dtype="int64")
            self.index.add_with_ids(np.asarray(vectors, dtype="float32"), ids)
            for i, m in zip(ids, metadatas):
                self._meta[int(i)] = m
            self._next_id += len(vectors)
            return ids.tolist()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        filter_fn=None,
        overfetch: int = 4,
    ) -> list[tuple[int, float, dict]]:
        """ANN search with optional metadata post-filtering.

        Post-filtering (search wider, then filter) preserves index efficiency
        under selective filters, per Amanbayev et al.'s findings cited in the
        report. ``overfetch`` controls how much wider we search when a filter
        is active.
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []
            k = min(self.index.ntotal, top_k * (overfetch if filter_fn else 1))
            q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
            scores, ids = self.index.search(q, k)
            out: list[tuple[int, float, dict]] = []
            for score, idx in zip(scores[0], ids[0]):
                if idx == -1:
                    continue
                meta = self._meta.get(int(idx), {})
                if filter_fn and not filter_fn(meta):
                    continue
                out.append((int(idx), float(score), meta))
                if len(out) >= top_k:
                    break
            return out

    def get_vector(self, vid: int) -> np.ndarray | None:
        with self._lock:
            try:
                return self.index.reconstruct(int(vid))
            except Exception:
                return None

    def contains(self, pred) -> bool:
        with self._lock:
            return any(pred(m) for m in self._meta.values())

    def all_meta(self) -> dict[int, dict]:
        with self._lock:
            return dict(self._meta)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        with self._lock:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            self._meta = {}
            self._next_id = 0
            for p in (self._index_path, self._meta_path):
                p.unlink(missing_ok=True)

    def maybe_train_ivf(self) -> None:
        """Upgrade flat -> IVF for large corpora (called by admin reindex)."""
        with self._lock:
            if self.index.ntotal < _IVF_THRESHOLD:
                return
            nlist = int(np.sqrt(self.index.ntotal) * 4)
            quantizer = faiss.IndexFlatIP(self.dim)
            ivf = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            vecs = np.vstack([self.index.reconstruct(int(i)) for i in self._meta])
            ids = np.array(list(self._meta.keys()), dtype="int64")
            ivf.train(vecs)
            ivf.add_with_ids(vecs, ids)
            ivf.nprobe = max(8, nlist // 32)
            self.index = ivf
            logger.info("Upgraded %s index to IVF (nlist=%d)", self.name, nlist)
