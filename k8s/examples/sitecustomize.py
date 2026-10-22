# sitecustomize.py — Auto-executed by Python at startup, which is why the patch lives here rather than
# in a wrapper script: a wrapper that patches and then execs `vllm serve` loses the patch with its own
# process.
# Makes huggingface_hub's validate_repo_id a no-op for absolute paths. It otherwise raises
# HFValidationError on a local path like /models/... before anything checks whether that directory
# exists, because config resolution routes the path through hf_hub_download. Mounted by the
# vllm:0.21.0-gpu-py312 interconnect-study experiments under experiments/. Confirm your own image
# still needs it before carrying this forward; current serving recipes on v0.29.0 do not.
try:
    import huggingface_hub.utils._validators as _v
    _orig = _v.validate_repo_id
    _v.validate_repo_id = lambda repo_id, **kw: None if str(repo_id).startswith('/') else _orig(repo_id, **kw)
except Exception:
    pass
