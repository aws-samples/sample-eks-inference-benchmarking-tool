"""Configuration for the quantization tool.

Every environment variable that quantize.py and fix_amax.py read, in one place. Nothing else
under src/ calls os.getenv. Values are read once at import so a setting cannot change midway
through a run.
"""
import os


# ---------------------------------------------------------------------------- #
#                                 ENV READERS                                  #
# ---------------------------------------------------------------------------- #

def env_str(name: str, default: str) -> str:
    """Read a string variable, treating whitespace-only as unset."""
    return os.getenv(name, default).strip() or default


def env_int(name: str, default: int) -> int:
    """Read an integer variable."""
    return int(os.getenv(name, str(default)))


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean variable, accepting 1/true/yes in any case."""
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes")


def env_csv(name: str, default: str = "") -> list[str]:
    """Read a comma-separated string list, dropping empty entries."""
    return [x.strip() for x in env_str(name, default).split(",") if x.strip()]


# ---------------------------------------------------------------------------- #
#                             MODEL & HUGGINGFACE                              #
# ---------------------------------------------------------------------------- #

# HuggingFace model ID or local path to quantize.
MODEL_ID = env_str("MODEL_ID", "/models/quantized/Qwen2.5-7B-Instruct-fp8")

# Target quantization format: fp8, int8, or awq.
QUANTIZATION_FORMAT = env_str("QUANTIZATION_FORMAT", "fp8")

# HuggingFace token for gated model access.
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")


# ---------------------------------------------------------------------------- #
#                                STORAGE PATHS                                 #
# ---------------------------------------------------------------------------- #

# Directory for downloaded model weights (PVC mount in-cluster).
MODEL_CACHE_DIR = env_str("MODEL_CACHE_DIR", "/models/cache")

# Parent directory for quantized checkpoint output.
OUTPUT_DIR = env_str("OUTPUT_DIR", "/models/quantized")

# Existing checkpoint directory that fix_amax.py rewrites in place.
MODEL_PATH = env_str("MODEL_PATH", "/models/quantized/Qwen2.5-7B-Instruct-fp8")


# ---------------------------------------------------------------------------- #
#                                 QUANTIZATION                                 #
# ---------------------------------------------------------------------------- #

# Auto-class for loading: "causal_lm" or "image_text_to_text" (multimodal).
MODEL_LOADER = env_str("MODEL_LOADER", "causal_lm").lower()

# True = real FP8 export via export_hf_checkpoint; False = legacy path (invalid, for repro only).
MODELOPT_REAL_EXPORT = env_flag("MODELOPT_REAL_EXPORT", True)

# Whether to run a calibration forward pass for static activation scales.
CALIBRATE = env_flag("CALIBRATE", True)

# Number of calibration prompts to run.
CALIB_SAMPLES = env_int("CALIB_SAMPLES", 16)

# Glob patterns of layers to exclude from quantization (kept full precision).
QUANT_EXCLUDE_PATTERNS = env_csv("QUANT_EXCLUDE_PATTERNS")


def quantized_model_path() -> str:
    """Destination path for a quantized checkpoint, derived from the source model name."""
    return os.path.join(OUTPUT_DIR, f"{MODEL_ID.split('/')[-1]}-{QUANTIZATION_FORMAT}")
