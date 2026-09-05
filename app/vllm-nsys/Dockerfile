# vLLM container plus the Nsight Systems CLI, for the profile.sh beside this file. nsys must be
# present at process start because CUDA and NCCL tracing cannot attach to a running server, the
# vLLM container does not ship it, and this build exposes no in-process profiling endpoints, so
# profiling needs this image. nsys comes from the standalone devtools container rather than a CUDA
# devel image, which no longer ships it, and its tags carry an OS suffix while the install tree
# differs from the old CUDA layout; copying /opt/nvidia wholesale sidesteps both path shapes and
# the next stage locates the binary by name.

FROM nvcr.io/nvidia/devtools/nsight-systems-cli:2026.3.1-ubuntu22.04 AS nsys

FROM public.ecr.aws/deep-learning-containers/vllm:0.21.0-gpu-py312

COPY --from=nsys /opt/nvidia /opt/nvidia

# nsys needs runtime libs a slim image may lack, best-effort since the version gate below fails the build.
RUN apt-get update -qq 2>/dev/null && \
    apt-get install -y --no-install-recommends libglib2.0-0 2>/dev/null || \
    echo "WARN: apt unavailable; relying on libs already present"

# Symlink whatever nsys is found, then fail the build rather than the run if the tracer cannot start.
RUN set -eux; \
    nsys_bin="$(find /opt/nvidia -type f -name nsys | head -1)"; \
    test -n "$nsys_bin" || { echo "FATAL: no nsys under /opt/nvidia" >&2; \
                             find /opt/nvidia -maxdepth 3 -type d; exit 1; }; \
    echo "found: $nsys_bin"; \
    ln -sf "$nsys_bin" /usr/local/bin/nsys; \
    nsys --version

# CPU sampling needs CAP_SYS_ADMIN, without which nsys still captures CUDA and NVTX.
ENV NSYS_NOTE="CPU sampling requires CAP_SYS_ADMIN on the pod securityContext"
