# slisaDFT community edition — CLI agent demo image (mock engine mode)
# Build:  docker build -t slisadft-agent .
# Run:    docker run --rm -e LLM_API_KEY=sk-xxx slisadft-agent \
#           slisadft adsorption --inputs '{"element":"Pt","adsorbate":"CO"}'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SLISADFT_ENGINE_MODE=mock \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . && rm -rf src slisadft.egg-info \
    && pip install --no-cache-dir .

COPY knowledge ./knowledge
COPY PRINCIPLES.md LICENSE ./

CMD ["slisadft", "check-env"]
