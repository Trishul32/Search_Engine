.PHONY: install install-ml api ui test eval seed-demo docker

install:        ## minimal install (lexical fallbacks, runs anywhere)
	pip install -r requirements.txt

install-ml:     ## full ML stack (embeddings, reranker, local LLMs)
	pip install -r requirements-ml.txt

api:            ## run the FastAPI backend  ->  http://localhost:8000/docs
	uvicorn backend.app.main:app --reload --port 8000

ui:             ## run the Streamlit frontend ->  http://localhost:8501
	streamlit run frontend/streamlit_app.py

test:
	python -m pytest tests/ -q

eval:           ## Recall@k / MRR / nDCG: bm25 vs dense vs hybrid vs hybrid+rerank
	python -m evaluation.evaluate --k 1 5 10

seed-demo:      ## ingest 200 RAG/IR papers from the live arXiv API
	python -m scripts.ingest --arxiv-query "retrieval augmented generation" --max 200

docker:
	docker compose up --build
