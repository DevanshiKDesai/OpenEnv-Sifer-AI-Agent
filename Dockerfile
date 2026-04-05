# Dockerfile — SiferTrustEnv
# ============================================================
# Builds a lightweight Python container that installs all
# dependencies, copies source files, and starts the FastAPI
# server on port 7860 for Hugging Face Spaces.
#
# Build:  docker build -t sifer-trust-env .
# Run:    docker run --rm -p 7860:7860 \
#           -e HF_TOKEN=hf_... \
#           -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
#           -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
#           sifer-trust-env
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Scaler x OpenEnv Hackathon"
LABEL description="SiferTrustEnv — Trust & Safety fraud analyst simulation"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV ENABLE_WEB_INTERFACE=true

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY sifer_env.py  .
COPY inference.py  .
COPY server.py     .
COPY openenv.yaml  .
COPY README.md     .

RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "server.py"]