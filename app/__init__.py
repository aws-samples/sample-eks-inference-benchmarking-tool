"""Operator-facing entry points for the EKS Inference Benchmarking Tool, where cli.py holds the Click
command tree and utils.py holds the command vocabulary, help rendering and the exec helper, both
importable as app.cli and app.utils. The subdirectories are container build contexts rather than
packages, and app/bench/src/main.py depends only on the standard library and aiohttp so it can run
in a stock Python image in-cluster."""
