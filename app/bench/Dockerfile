# Benchmark client image. Lightweight Python image for running the benchmark harnesses against a
# vLLM endpoint. Nothing uses it today: `main.py launch` ships the harness into the cluster as a
# ConfigMap and runs it on the stock `python:3.12` image named in app/bench/suite-job.yaml, so this
# exists only for a future where the harness is baked into an image instead.
#
# Self-contained: the build context is this directory, so COPY paths are relative to it.
#   docker build -t vllm-bench -f app/bench/Dockerfile app/bench
FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so a code change does not invalidate the pip cache.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# src/ holds the harness and its own config.py; data/ carries the pinned prompt profiles and the
# gsm8k subset the accuracy gate reads. config.py resolves both relative to this directory.
COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Throughput is the default. Override for the other harnesses:
#   docker run <image> src/main.py latency
#   docker run <image> src/main.py arrival_rate
ENTRYPOINT ["python"]
CMD ["src/main.py", "throughput"]
