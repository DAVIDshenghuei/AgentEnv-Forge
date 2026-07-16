FROM ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv-bin

FROM python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

COPY --from=uv-bin /uv /uvx /usr/local/bin/
RUN uv --version | grep -q '^uv 0.11.29 '

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 forge
USER forge

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "agentenv_forge"]
CMD ["run", "--task", "text-normalization-001", "--action", "correct", "--seed", "42", "--output", "/tmp/agentenv-forge/trajectory.jsonl"]
