# Use NVIDIA PyTorch 25.03 which includes Blackwell (sm_120 / g7e) CUDA kernel support.
# 24.03 only supports up to sm_90 (H100) and will fail with "no kernel image" on g7e nodes.
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /app

# Install nvidia-modelopt and other dependencies.
# Use nvidia-modelopt[torch] instead of [all] to avoid pulling in the 'datasets'
# optional dependency, which conflicts with dill==0.3.9 pinned in the 25.03 base image.
#
# The base image sets PIP_CONSTRAINT=/etc/pip/constraint.txt which pins
# nvidia-modelopt==0.25.0. modelopt 0.25.0 has a broken HuggingFace plugin
# that injects a 'dtype' kwarg Qwen2ForCausalLM.__init__() doesn't accept.
# Remove the modelopt pin from the constraint file, then upgrade to 0.29.0+
# which has the refactored plugin (context-manager based, signature-agnostic).
# Use nvidia-modelopt[torch] instead of [all] to avoid the 'datasets' dependency
# that conflicts with dill==0.3.9 pinned in the 25.03 base image.
#
# transformers>=5.12.0: Gemma 4 (model_type "gemma4", released Apr 2026) is only
# recognized by transformers >= 5.8.1. The previously-built image had 5.3.0, which
# fails to load Gemma 4 with "Transformers does not recognize this architecture".
# Pinning a floor of 5.12.0 keeps Gemma 4 support while remaining compatible with
# the existing Qwen2.5 path (Qwen2 is supported across all these versions).
RUN sed -i -E -e '/nvidia.modelopt/d' -e '/^regex([ =<>!~]|$)/d' -e '/^tokenizers([ =<>!~]|$)/d' -e '/^huggingface[-_]hub([ =<>!~]|$)/d' -e '/^safetensors([ =<>!~]|$)/d' /etc/pip/constraint.txt
COPY app/quantize/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Copy the app LAST so code changes don't invalidate the pip cache.
COPY app/quantize/src/ /app/src/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Quantization is the default. Override to run the checkpoint repair pass instead:
#   docker run <image> /app/src/fix_amax.py
ENTRYPOINT ["python"]
CMD ["/app/src/quantize.py"]
