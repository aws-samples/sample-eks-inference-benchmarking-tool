# vLLM DLC + Nsight Systems CLI, for the profile.sh beside this file.
#
# WHY THIS IMAGE EXISTS
#   nsys must be present in the container at process start: CUDA/NCCL tracing cannot be
#   attached to an already-running server. The vLLM Deep Learning Container does not ship
#   nsys, and this build's vLLM 0.21.0 exposes neither /start_profile nor /stop_profile
#   (verified on the node), so there is no in-process fallback. Profiling needs this image.
#
# WHERE nsys COMES FROM, AND THE TWO TRAPS
#   1. NVIDIA decoupled Nsight Systems from cuda:*-devel, so
#      `COPY --from=nvidia/cuda:12.8.0-devel-*` finds nothing. Use the standalone tool
#      container nvcr.io/nvidia/devtools/nsight-systems-cli instead.
#   2. Its tags carry an OS suffix: "2026.3.1" does not resolve, "2026.3.1-ubuntu22.04" does.
#      And the install tree is /opt/nvidia/nsight-systems-CLI/<ver>/..., not
#      /opt/nvidia/nsight-systems/ as in the old CUDA images.
#
#   Copying /opt/nvidia wholesale sidesteps both path shapes; stage 2 then locates the binary
#   by name, so a future tag that moves or renames the tree still works.

FROM nvcr.io/nvidia/devtools/nsight-systems-cli:2026.3.1-ubuntu22.04 AS nsys

FROM public.ecr.aws/deep-learning-containers/vllm:0.21.0-gpu-py312

COPY --from=nsys /opt/nvidia /opt/nvidia

# nsys needs a few runtime libs a slim serving image may lack (glib is the usual one).
# Best-effort: if apt is unusable, the `nsys --version` gate below still fails the build.
RUN apt-get update -qq 2>/dev/null && \
    apt-get install -y --no-install-recommends libglib2.0-0 2>/dev/null || \
    echo "WARN: apt unavailable; relying on libs already present"

# Symlink whatever nsys we find, then FAIL THE BUILD (not the run, on metered GPU time) if
# the tracer cannot actually start.
RUN set -eux; \
    nsys_bin="$(find /opt/nvidia -type f -name nsys | head -1)"; \
    test -n "$nsys_bin" || { echo "FATAL: no nsys under /opt/nvidia" >&2; \
                             find /opt/nvidia -maxdepth 3 -type d; exit 1; }; \
    echo "found: $nsys_bin"; \
    ln -sf "$nsys_bin" /usr/local/bin/nsys; \
    nsys --version

# CPU sampling needs CAP_SYS_ADMIN; the profiling manifest requests it. Without it nsys still
# captures CUDA/NVTX, it just drops OS-runtime sampling.
ENV NSYS_NOTE="CPU sampling requires CAP_SYS_ADMIN on the pod securityContext"
