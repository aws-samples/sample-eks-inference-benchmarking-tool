# Benchmark client image, unused today because `main.py launch` ships the harness into the cluster
# as a ConfigMap and runs it on the stock Python image named in app/bench/suite-job.yaml, so this
# exists only for a future where the harness is baked into an image instead. The build context is
# this directory, so COPY paths are relative to it.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so a code change does not invalidate the pip cache.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# src/ holds the harness and data/ the pinned profiles and eval subset, both resolved relative to here.
COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Throughput is the default, so override the argument to run one of the other harnesses.
ENTRYPOINT ["python"]
CMD ["src/main.py", "throughput"]
