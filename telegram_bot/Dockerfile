# -------------------------
# Stage 1: Builder
# -------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install --prefix=/install --no-cache-dir \
       -r requirements.txt

# -------------------------
# Stage 2: Runtime
# -------------------------

FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

CMD ["python3", "-m", "app.main"]

