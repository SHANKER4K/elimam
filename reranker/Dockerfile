FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     HF_HOME=/data     TRANSFORMERS_CACHE=/data     TOKENIZERS_PARALLELISM=false     OMP_NUM_THREADS=4     MKL_NUM_THREADS=4

WORKDIR /app

RUN pip install --no-cache-dir     --index-url https://download.pytorch.org/whl/cpu     torch==2.8.0

RUN pip install --no-cache-dir     fastapi==0.117.1     uvicorn[standard]==0.36.0     sentence-transformers==5.4.1

COPY app.py .

RUN useradd --create-home --shell /usr/sbin/nologin appuser     && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 80

CMD [uvicorn, app:app,      --host, 0.0.0.0,      --port, 80,      --workers, 1]
