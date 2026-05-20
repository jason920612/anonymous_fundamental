FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip \
    && pip install -e ".[prices,gbt,dev]"

# Default data volumes
VOLUME ["/app/data", "/app/artifacts", "/app/reports"]

ENTRYPOINT ["tini", "--", "afp"]
CMD ["--help"]
