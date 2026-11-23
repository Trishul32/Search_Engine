"""Pure-NumPy stand-in for the small subset of FAISS used by vector_store.

Used only when ``import faiss`` fails (e.g. Windows Smart App Control
blocking the native DLL). Exact inner-product search; same results as
``IndexIDMap2(IndexFlatIP)``, just without FAISS's SIMD speed.
"""
from __future__ import annotations

import numpy as np

METRIC_INNER_PRODUCT = 0


class IndexFlatIP:
    def __init__(self, d: int, *_, **__):
        self.d = d
        self._vecs = np.zeros((0, d), dtype="float32")
        self._ids = np.zeros((0,), dtype="int64")
        self.nprobe = 1

    @property
    def ntotal(self) -> int:
        return int(self._ids.shape[0])

    def train(self, _vecs) -> None:  # IVF compatibility; flat needs no training
        pass

    def add_with_ids(self, vecs, ids) -> None:
        vecs = np.asarray(vecs, dtype="float32").reshape(-1, self.d)
        ids = np.asarray(ids, dtype="int64").reshape(-1)
        self._vecs = np.vstack([self._vecs, vecs])
        self._ids = np.concatenate([self._ids, ids])

    def search(self, q, k: int):
        q = np.asarray(q, dtype="float32").reshape(-1, self.d)
        n = self.ntotal
        scores = np.full((q.shape[0], k), -np.inf, dtype="float32")
        out_ids = np.full((q.shape[0], k), -1, dtype="int64")
        if n == 0 or k <= 0:
            return scores, out_ids
        sims = q @ self._vecs.T
        kk = min(k, n)
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r in range(q.shape[0]):
            order = top[r][np.argsort(-sims[r, top[r]])]
            scores[r, :kk] = sims[r, order]
            out_ids[r, :kk] = self._ids[order]
        return scores, out_ids

    def reconstruct(self, vid: int):
        hit = np.nonzero(self._ids == int(vid))[0]
        if hit.size == 0:
            raise KeyError(vid)
        return self._vecs[hit[0]].copy()


def IndexIDMap2(inner: IndexFlatIP) -> IndexFlatIP:
    return inner


IndexIVFFlat = IndexFlatIP


def write_index(index: IndexFlatIP, path: str) -> None:
    with open(path, "wb") as fh:
        np.savez(fh, d=np.array(index.d), vecs=index._vecs, ids=index._ids)


def read_index(path: str) -> IndexFlatIP:
    with open(path, "rb") as fh:
        data = np.load(fh)
        idx = IndexFlatIP(int(data["d"]))
        idx.add_with_ids(data["vecs"], data["ids"])
    return idx
