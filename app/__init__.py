"""
Operator-facing entry points for the EKS Inference Benchmarking Tool.

ARCHITECTURE:
    cli.py holds the Click command tree; utils.py holds the command vocabulary, help rendering
    and the exec helper. Both are importable as `app.cli` and `app.utils`, which is what the
    installed console scripts target.

    The subdirectories are not packages. app/bench/, app/quantize/ and app/vllm-nsys/ are
    container build contexts whose modules run as scripts inside their own images, and
    app/bench/src/main.py depends only on the standard library and aiohttp so it can run in a
    stock Python image in-cluster.
"""
