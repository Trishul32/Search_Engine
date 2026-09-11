# Backend API image (extend with requirements-ml.txt for the full ML stack)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
ARG ML=false
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$ML" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
COPY backend ./backend
COPY evaluation ./evaluation
COPY scripts ./scripts
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
