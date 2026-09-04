#!/usr/bin/env python3
"""Rewrite modelopt _amax tensors in an existing checkpoint as the scale tensors vLLM loads.

A repair tool for checkpoints produced before quantize.py handled this itself. modelopt saves
`...weight_quantizer._amax` while vLLM expects `...weight_scale`, holding amax / 448.0. Two
rules matter and both were learned from vLLM's AutoWeightsLoader source: the `model.` prefix
must be kept, because the loader routes by top-level module name, and only the `_amax` suffix
is renamed. It also restores the prefix on shards where an earlier version stripped it. The
pass is idempotent, so shards that are already correct are skipped, and it needs only PVC
access rather than a GPU. Set MODEL_PATH to the checkpoint directory.
"""
import glob
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

import config

# FP8 E4M3 maximum representable magnitude; vLLM's scale is amax divided by this.
FP8_E4M3_MAX = 448.0

SCALE_KEYS = ("weight_scale", "input_scale")


# ---------------------------------------------------------------------------- #
#                              KEY INSPECTION                                  #
# ---------------------------------------------------------------------------- #

def is_amax_key(key: str) -> bool:
    """True if this key is a modelopt amax tensor awaiting rename."""
    return key.endswith(("weight_quantizer._amax", "input_quantizer._amax"))


def is_prefix_stripped(key: str) -> bool:
    """True if this scale key lost its required `model.` prefix in an earlier pass."""
    return key.startswith("layers.") and any(name in key for name in SCALE_KEYS)


def renamed_key(key: str) -> tuple[str, bool]:
    """Return (corrected key, whether the value must be divided by the FP8 maximum)."""
    if key.endswith("weight_quantizer._amax"):
        return key.replace("weight_quantizer._amax", "weight_scale"), True
    if key.endswith("input_quantizer._amax"):
        return key.replace("input_quantizer._amax", "input_scale"), True
    if is_prefix_stripped(key):
        return "model." + key, False
    return key, False


# ---------------------------------------------------------------------------- #
#                              SHARD REWRITING                                 #
# ---------------------------------------------------------------------------- #

def fix_shard(shard_path: str) -> bool:
    """Rewrite one shard in place, returning True if anything changed."""
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not any(is_amax_key(k) or is_prefix_stripped(k) for k in keys):
            print(f"  SKIP (already clean): {os.path.basename(shard_path)}")
            return False
        tensors = {}
        for key in keys:
            new_key, rescale = renamed_key(key)
            tensor = handle.get_tensor(key)
            tensors[new_key] = tensor / FP8_E4M3_MAX if rescale else tensor

    save_file(tensors, shard_path)
    scale_count = sum(1 for k in tensors if any(name in k for name in SCALE_KEYS))
    print(f"  FIXED: {os.path.basename(shard_path)} "
          f"({scale_count} scale keys with model. prefix)")
    return True


def report_shard(shard_path: str) -> None:
    """Print one shard's scale-key count and how many still lack the required prefix."""
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        scale_keys = [k for k in handle.keys() if any(n in k for n in SCALE_KEYS)]
    unprefixed = [k for k in scale_keys if not k.startswith("model.")]
    print(f"  {os.path.basename(shard_path)}: {len(scale_keys)} scale keys, "
          f"{len(unprefixed)} without model. prefix")


# ---------------------------------------------------------------------------- #
#                               ENTRY POINT                                    #
# ---------------------------------------------------------------------------- #

def main() -> int:
    """Fix every shard under MODEL_PATH, then report the resulting key layout."""
    print(f"Scanning safetensors shards in: {config.MODEL_PATH}")
    shards = sorted(glob.glob(os.path.join(config.MODEL_PATH, "*.safetensors")))
    if not shards:
        print("ERROR: No safetensors files found. Check MODEL_PATH.")
        return 1

    print(f"Found {len(shards)} shard(s).")
    fixed = sum(1 for shard in shards if fix_shard(shard))
    print(f"\nDone. {fixed}/{len(shards)} shard(s) updated.")

    print("\nSanity check — scale key prefixes across all shards:")
    for shard in shards:
        report_shard(shard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
