"""FastAPI application entrypoint.

Run:  uvicorn backend.app.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.api import router
from backend.app.core.config import settings
from backend.app.services.pipeline import get_pipeline

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # eager init: DB, FAISS indexes, retrievers, models
    logging.getLogger(__name__).info("%s ready", settings.app_name)
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Hybrid RAG search over arXiv: BM25 + dense retrieval, cross-encoder "
        "reranking, lazy PDF loading, structure-aware chunking, FAISS vector "
        "storage, single/multi-document QA with source attribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")
