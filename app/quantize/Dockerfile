# Quantizer image on NVIDIA PyTorch 25.08, which ships torch 2.8 (required by nvidia-modelopt) and
# the Blackwell CUDA kernels.
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

# Install nvidia-modelopt[torch] rather than [all], because [all] pulls in a datasets dependency
# that conflicts with the dill version pinned in the base image. The base image also constrains
# modelopt to a version whose HuggingFace plugin injects a dtype kwarg some architectures reject,
# so the pin is removed before upgrading to the refactored plugin. The transformers floor keeps
# Gemma 4 recognised while staying compatible with the existing Qwen2.5 path.
RUN sed -i -E -e '/nvidia.modelopt/d' -e '/^regex([ =<>!~]|$)/d' -e '/^tokenizers([ =<>!~]|$)/d' -e '/^huggingface[-_]hub([ =<>!~]|$)/d' -e '/^safetensors([ =<>!~]|$)/d' -e '/^setuptools([ =<>!~]|$)/d' /etc/pip/constraint.txt
COPY app/quantize/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Copy the app LAST so code changes don't invalidate the pip cache.
COPY app/quantize/src/ /app/src/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Quantization is the default, so override the argument to run the checkpoint repair pass instead.
ENTRYPOINT ["python"]
CMD ["/app/src/quantize.py"]
