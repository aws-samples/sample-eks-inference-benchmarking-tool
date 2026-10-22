# Generic EFA-enabled vLLM serving image. Adds EFA/libfabric and the aws-ofi-nccl NCCL plugin on top
# of a vLLM base so multi-node runs carry NCCL collectives over EFA RDMA instead of TCP. Any recipe
# that serves one model across more than one node builds from this; it is not model-specific.
#
# Build and push with the shared kaniko script:
#   REPOSITORY=<your-repo> CONTEXT_DIR=app/vllm-efa ./scripts/push-image-kaniko.sh <tag>
#
# VERIFY the versions below against your cluster before a release. The defaults target the
# vllm-kimi-k3-efa:nccl2.29.7-efa1.49.0 image the Kimi K3 recipes expect. The plugin, NCCL and the
# host EFA driver must agree, or NCCL silently falls back to TCP; the recipe's verify-efa-nccl check
# and NCCL_DEBUG log are what prove EFA is actually in use.
ARG BASE_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${BASE_IMAGE}

ARG EFA_INSTALLER_VERSION=1.49.0
ARG AWS_OFI_NCCL_VERSION=1.16.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends curl ca-certificates build-essential autoconf libtool \
 && rm -rf /var/lib/apt/lists/*

# EFA userspace (libfabric + rdma-core) from the AWS EFA installer. --skip-kmod and --no-verify
# because the EFA kernel driver lives on the host, not in the container.
RUN cd /tmp \
 && curl -fsSL -O "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
 && tar -xzf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
 && cd aws-efa-installer \
 && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
 && rm -rf /tmp/aws-efa-installer*

# aws-ofi-nccl: the plugin that routes NCCL collectives over libfabric/EFA.
RUN cd /tmp \
 && curl -fsSL -O "https://github.com/aws/aws-ofi-nccl/releases/download/v${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}.tar.gz" \
 && tar -xzf "aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}.tar.gz" \
 && cd "aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}" \
 && ./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
 && make -j"$(nproc)" && make install \
 && rm -rf /tmp/aws-ofi-nccl*

ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
ENV FI_PROVIDER=efa

# Startup check the recipes call before serving.
COPY verify-efa-nccl /usr/local/bin/verify-efa-nccl
RUN chmod +x /usr/local/bin/verify-efa-nccl
