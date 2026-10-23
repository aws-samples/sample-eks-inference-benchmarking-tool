# Accuracy references

Capture an accuracy reference before the first benchmark of a `(model_tag, config)` pair. Every `measure` run then checks the live server against it and aborts if they disagree. Each runbook includes the capture as a numbered step before its first benchmark, so following a runbook covers this.

## Capturing a reference

Follow the runbook to capture a reference against a server you have checked. Tier 1 (`fingerprint`, ~30 s) gates every benchmark; tier 2 (`score`, minutes) runs once per pair. Both write the same `docs/accuracy/<model-tag>-<config>.json`:

```bash
eks-bench capture-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode fingerprint \
    --model-tag qwen7b --config fp8 \
    --model-id <served-model-name> \
    --out-dir docs/results
```

## How a run uses it

Every `measure` run gates on the fingerprint reference for the pair and fails closed if it is missing:

```
FAIL: no accuracy reference at docs/accuracy/<model-tag>-<config>.json

This gate fails closed. A missing reference is not a pass, because silently
proceeding is the exact failure this gate exists to prevent.
```

Pass `--skip-accuracy` to proceed unverified; the protocol lock records `accuracy gate : SKIPPED (--skip-accuracy) — results unverified`. References are gitignored, because each describes your server on your hardware.
