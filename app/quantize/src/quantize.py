#!/usr/bin/env python3
"""Quantize a HuggingFace checkpoint with nvidia-modelopt and write it for vLLM to serve, loading the
source model in BF16, applying a config chosen by QUANTIZATION_FORMAT, running a calibration
forward pass so static activation scales are real rather than absent, and exporting a checkpoint.
The export path matters more than it looks: the correct one casts weights to float8_e4m3fn and
emits a quantization config, while the legacy path leaves weights in BF16 while labelling the
checkpoint as serialized FP8, which makes vLLM read BF16 bytes as FP8 and emit garbage. The legacy
path is retained only so an older flow stays reproducible. Every setting comes from config.py, and
this runs inside the quantizer image, which needs a GPU."""
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq

import config
import verify_checkpoint

# FP8 E4M3 maximum magnitude, since modelopt stores an amax where vLLM expects amax over this.
FP8_E4M3_MAX = 448.0

# Quantization config per requested format.
QUANT_CONFIGS = {
    "fp8": "FP8_DEFAULT_CFG",
    "int8": "INT8_DEFAULT_CFG",
    "awq": "INT4_AWQ_CFG",
}

# Representative calibration prompts, a diverse sample being sufficient for per-tensor max calibration.
CALIB_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a linked list.",
    "What are the main differences between Kubernetes and Docker Swarm?",
    "Summarize the plot of The Matrix.",
    "How does a transformer neural network work?",
    "Describe the water cycle for a ten year old.",
    "List three trade-offs between microservices and a monolith.",
    "Translate 'good morning, how are you?' into French and Japanese.",
    "Give me a recipe for a simple tomato pasta.",
    "What caused the fall of the Western Roman Empire?",
    "Write a haiku about distributed systems.",
    "Explain the difference between TCP and UDP.",
    "What is gradient descent and why does it work?",
    "Outline the steps to set up CI/CD for a small team.",
    "Compare FP8 and BF16 for large language model inference.",
    "Why is the sky blue during the day and red at sunset?",
]


# ---------------------------------------------------------------------------- #
#                              MODEL LOADING                                   #
# ---------------------------------------------------------------------------- #

def load_tokenizer():
    """Load the tokenizer for the source model."""
    return AutoTokenizer.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN,
                                         cache_dir=config.MODEL_CACHE_DIR)


def load_model():
    """Load the model in BF16 using the auto-class selected by config.MODEL_LOADER."""
    common = dict(dtype=torch.bfloat16, device_map="auto", token=config.HF_TOKEN,
                  cache_dir=config.MODEL_CACHE_DIR)
    if config.MODEL_LOADER == "image_text_to_text":
        from transformers import AutoModelForImageTextToText
        print("Loading model via AutoModelForImageTextToText (multimodal)...")
        return AutoModelForImageTextToText.from_pretrained(config.MODEL_ID, **common)
    print("Loading model via AutoModelForCausalLM...")
    return AutoModelForCausalLM.from_pretrained(config.MODEL_ID, **common)


def input_device(model):
    """Find the device holding the input embeddings, to place calibration batches on."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


# ---------------------------------------------------------------------------- #
#                              QUANTIZATION                                    #
# ---------------------------------------------------------------------------- #

def quant_config():
    """Build the modelopt config for the requested format, with exclusions applied."""
    attr = QUANT_CONFIGS.get(config.QUANTIZATION_FORMAT.lower())
    if attr is None:
        raise ValueError(f"Unsupported quantization format: {config.QUANTIZATION_FORMAT}. "
                         f"Expected one of {sorted(QUANT_CONFIGS)}")
    cfg = getattr(mtq, attr)
    for pattern in config.QUANT_EXCLUDE_PATTERNS:
        apply_exclusion(cfg["quant_cfg"], pattern)
        print(f"Excluding from quantization (kept full precision): {pattern}")
    return cfg


def apply_exclusion(quant_cfg, pattern):
    """Disable quantization for one module pattern, on either quant_cfg schema.

    modelopt <= 0.45 ships quant_cfg as a dict of pattern -> settings; 0.46 made it a
    list of {"quantizer_name": pattern, ...} entries evaluated in order, later entries
    winning. Appending keeps the exclusion authoritative over the earlier defaults.
    Anything else is a schema this code has not seen, and silently continuing would
    quantize modules the caller asked to protect, so it raises instead.
    """
    if isinstance(quant_cfg, dict):
        quant_cfg[pattern] = {"enable": False}
    elif isinstance(quant_cfg, list):
        quant_cfg.append({"quantizer_name": pattern, "enable": False})
    else:
        raise TypeError(f"unrecognized modelopt quant_cfg schema: {type(quant_cfg).__name__}; "
                        f"refusing to guess how to exclude {pattern!r}")


def build_forward_loop(tokenizer):
    """Return a modelopt calibration closure, or None when calibration is disabled.

    The loop runs a few prompts through the model so modelopt records activation amax values
    and produces real static input_scale tensors. Without it those tensors are never
    populated and static-activation FP8 is invalid.
    """
    if not config.CALIBRATE:
        return None

    def forward_loop(model):
        """Run the calibration prompts through the model under no_grad."""
        device = input_device(model)
        prompts = CALIB_PROMPTS[:max(1, config.CALIB_SAMPLES)]
        model.eval()
        with torch.no_grad():
            for index, prompt in enumerate(prompts):
                batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                model(**{k: v.to(device) for k, v in batch.items()})
                print(f"  calibration {index + 1}/{len(prompts)} done")

    return forward_loop


# ---------------------------------------------------------------------------- #
#                                 EXPORT                                       #
# ---------------------------------------------------------------------------- #

def save_processor(output_path: str) -> None:
    """Save the processor config needed for multimodal serving; a no-op for text-only models."""
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(config.MODEL_ID, token=config.HF_TOKEN,
                                                 cache_dir=config.MODEL_CACHE_DIR)
        processor.save_pretrained(output_path)
        print("Saved processor/preprocessor config for multimodal serving.")
    except Exception as exc:
        print(f"No processor saved (text-only model or unavailable): {exc}")


def export_real_fp8(model, tokenizer, output_path: str) -> None:
    """Export a genuine FP8 checkpoint: real float8 weights plus static scales."""
    from modelopt.torch.export import export_hf_checkpoint
    print(f"Exporting REAL FP8 checkpoint via export_hf_checkpoint -> {output_path}")
    export_hf_checkpoint(model, export_dir=output_path)
    tokenizer.save_pretrained(output_path)
    save_processor(output_path)
    print("Real FP8 export complete. Serve with --quantization=modelopt.")


def export_legacy(model, tokenizer, output_path: str) -> None:
    """Export via the legacy path, which produces an INVALID FP8 checkpoint by design.

    Saves fake-quantized BF16 weights, injects a quantization_config, and renames modelopt
    _amax tensors. It does NOT cast weights to float8_e4m3fn, so vLLM treats BF16 bytes as
    FP8 and emits garbage. Retained only so the original behavior is reproducible.
    """
    print("[legacy] Saving fake-quant model + vLLM compat fixups (NOT a real FP8 checkpoint)")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    save_processor(output_path)
    inject_quantization_config(output_path)
    for shard in glob.glob(os.path.join(output_path, "*.safetensors")):
        rename_amax_keys(shard)


def inject_quantization_config(output_path: str) -> None:
    """Write the quantization_config block vLLM needs to detect an FP8 checkpoint."""
    config_path = os.path.join(output_path, "config.json")
    with open(config_path) as handle:
        model_config = json.load(handle)
    model_config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "static",
        "ignored_layers": None,
    }
    with open(config_path, "w") as handle:
        json.dump(model_config, handle, indent=2)


def rename_amax_keys(shard_path: str) -> None:
    """Rewrite one shard's modelopt _amax tensors as the scale tensors vLLM loads."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors, modified = {}, False
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.endswith("weight_quantizer._amax"):
                tensors[key.replace("weight_quantizer._amax", "weight_scale")] = (
                    tensor / FP8_E4M3_MAX)
                modified = True
            elif key.endswith("input_quantizer._amax"):
                tensors[key.replace("input_quantizer._amax", "input_scale")] = (
                    tensor / FP8_E4M3_MAX)
                modified = True
            else:
                tensors[key] = tensor
    if modified:
        save_file(tensors, shard_path)
        print(f"[legacy] Updated {shard_path}")


# ---------------------------------------------------------------------------- #
#                               ENTRY POINT                                    #
# ---------------------------------------------------------------------------- #

def print_settings() -> None:
    """Announce the settings that decide which export path runs, before any work happens."""
    print(f"Starting quantization for {config.MODEL_ID} to {config.QUANTIZATION_FORMAT}")
    print(f"  REAL_EXPORT={config.MODELOPT_REAL_EXPORT}  CALIBRATE={config.CALIBRATE}  "
          f"CALIB_SAMPLES={config.CALIB_SAMPLES}")
    if not config.HF_TOKEN:
        print("WARNING: HUGGING_FACE_HUB_TOKEN is not set. Gated models will fail to download.")


def export_checkpoint(model, tokenizer, output_path: str) -> None:
    """Write the quantized model by whichever export path the configuration selects."""
    if config.MODELOPT_REAL_EXPORT:
        export_real_fp8(model, tokenizer, output_path)
        return
    export_legacy(model, tokenizer, output_path)
    print("Legacy save complete (see export_legacy for why this checkpoint is invalid).")


def exit_code_for(problems: list[str]) -> int:
    """Turn verification problems into an exit code, so a bad export fails the Job.

    Without this a checkpoint that vLLM will serve as garbage lands on a shared claim behind a
    Job that exited 0, and the next reader inherits it. The legacy path fails here on purpose.
    """
    if not problems:
        return 0
    if config.ALLOW_INVALID_EXPORT:
        print("\nALLOW_INVALID_EXPORT is set, so exiting 0 despite the problems above. "
              "Do not benchmark this checkpoint.")
        return 0
    print("\nExport failed verification. The checkpoint is on disk for inspection but must "
          "not be served. Set ALLOW_INVALID_EXPORT=1 only to reproduce the legacy export.")
    return 1


def verify_export(output_path: str) -> int:
    """Verify the exported checkpoint and return the exit code it earns."""
    problems, notes = verify_checkpoint.verify(
        output_path, config.QUANTIZATION_FORMAT, config.QUANT_EXCLUDE_PATTERNS, config.CALIBRATE)
    verify_checkpoint.report(output_path, problems, notes)
    return exit_code_for(problems)


def main() -> int:
    """Quantize the configured model, export it for vLLM, and refuse to pass off a bad export."""
    print_settings()
    tokenizer = load_tokenizer()
    model = load_model()
    forward_loop = build_forward_loop(tokenizer)
    print(f"Applying {config.QUANTIZATION_FORMAT} quantization using nvidia-modelopt"
          f"{' (with calibration)' if forward_loop else ' (no calibration)'}...")
    model = mtq.quantize(model, quant_config(), forward_loop=forward_loop)

    output_path = config.quantized_model_path()
    os.makedirs(output_path, exist_ok=True)
    export_checkpoint(model, tokenizer, output_path)
    return verify_export(output_path)


if __name__ == "__main__":
    sys.exit(main())
