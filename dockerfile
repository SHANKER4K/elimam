# -------------------------
# Stage 1: Builder
# -------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install --upgrade pip

RUN pip install --prefix=/install --no-cache-dir \
                         --extra-index-url https://download.pytorch.org/whl/cpu \
                         torch==2.13.0+cpu \
                         -r requirements.txt


RUN pip install --prefix=/intall --no-cache-dir pydantic-ai_harness

# -------------------------
# Stage 2: Runtime
# -------------------------

FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
