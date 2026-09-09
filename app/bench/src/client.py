"""Shared request, response-parsing and reporting logic for the benchmark harnesses, covering
everything that knows the shape of an OpenAI-compatible vLLM response. These functions existed as
near-identical copies in each harness, which is how a reasoning-token fix landed in one probe but
not another and how the measured input size ended up recorded in three formats. Anything that does
not need to know the response shape belongs in stats.py instead."""
import asyncio
import json

import aiohttp

import config
from stats import minmax_median


# ---------------------------------------------------------------------------- #
#                             FAILURE CLASSIFICATION                           #
# ---------------------------------------------------------------------------- #

# A dropped connection invalidates the sample where an HTTP 500 is the result, so they split here.
TRANSPORT_ERRORS = (
    aiohttp.ClientConnectionError,      # covers ClientOSError, ServerDisconnectedError
    aiohttp.ClientPayloadError,         # truncated response body mid-stream
    ConnectionResetError,
    asyncio.TimeoutError,
)

TRANSPORT = "transport"
SERVER = "server"


def classify_failure(exc: BaseException) -> str:
    """Return TRANSPORT or SERVER for a failed request.

    aiohttp raises ClientResponseError for an HTTP status, which is the server answering, so
    it is attributed to the server. Everything in TRANSPORT_ERRORS is the connection itself
    failing. Anything unrecognised is treated as transport, because the conservative reading
    of an unknown failure is that the sample is untrustworthy rather than that the server
    misbehaved.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return SERVER
    return TRANSPORT if isinstance(exc, TRANSPORT_ERRORS) else TRANSPORT


def request_success_line(level: int, ok: int, total: int,
                         transport: int, server: int) -> str:
    """Format one REQUEST_SUCCESS provenance line for the protocol runner to parse.

    A separate line type, like TOKEN_SHAPE and CONCURRENCY_FILL: the runner's summary-table
    regexes are matched against published result files, so adding a column would stop parsing
    every one of them.
    """
    share = (ok / total) if total else 0.0
    return (f"REQUEST_SUCCESS concurrency={level} ok={ok} total={total} "
            f"ok_fraction={share:.4f} transport_failures={transport} "
            f"server_failures={server}")


# ---------------------------------------------------------------------------- #
#                               REQUEST BUILDING                               #
# ---------------------------------------------------------------------------- #

def build_request(prompt: str, max_tokens: int, stream: bool = False) -> tuple[str, dict]:
    """Return (url, JSON body) for one request, routed by config.BENCH_ENDPOINT.

    Both routes report usage.completion_tokens and usage.prompt_tokens, so downstream token
    accounting is identical whichever one is in use.
    """
    body = {
        "model": config.MODEL_ID,
        "max_tokens": max_tokens,
        "temperature": config.BENCH_TEMPERATURE,
    }
    if stream:
        # include_usage asks vLLM for exact counts, which beats counting content chunks client-side.
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    base = config.VLLM_API_URL.rstrip("/")
    if config.BENCH_ENDPOINT == "completions":
        return f"{base}/completions", {"prompt": prompt, **body}
    return f"{base}/chat/completions", {"messages": [{"role": "user", "content": prompt}], **body}


# ---------------------------------------------------------------------------- #
#                               RESPONSE PARSING                               #
# ---------------------------------------------------------------------------- #

async def iter_sse_chunks(response):
    """Yield each decoded JSON object from a server-sent-event stream, ignoring framing.

    Malformed chunks are skipped rather than raising: a single truncated frame should cost
    one token sample, not the whole request.
    """
    async for raw in response.content:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        chunk = decode_sse_payload(payload)
        if chunk is not None:
            yield chunk


def decode_sse_payload(payload: str) -> dict | None:
    """Parse one SSE data payload, returning None if it is not valid JSON."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def token_piece(chunk: dict) -> str:
    """Extract the generated text from one streaming chunk, or "" if it carries none.

    Reasoning models stream their chain of thought in a separate field before any answer
    text — Kimi K3 uses "reasoning", other builds use "reasoning_content". Those tokens are
    real decode work the user waits through, so they must count toward TTFT, TPOT and ITL.
    Reading only "content" scores every request from a reasoning model as a failure.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    if config.BENCH_ENDPOINT == "completions":
        return choices[0].get("text") or ""
    delta = choices[0].get("delta") or {}
    return (delta.get("content") or delta.get("reasoning")
            or delta.get("reasoning_content") or "")


def usage_counts(payload: dict) -> tuple[int | None, int | None]:
    """Extract (completion_tokens, prompt_tokens) from a response or a streaming chunk.

    None means the field was absent, which is distinct from a genuine zero and lets a caller
    fall back to counting chunks instead of recording a wrong count.
    """
    usage = payload.get("usage") or {}
    return usage.get("completion_tokens"), usage.get("prompt_tokens")


# ---------------------------------------------------------------------------- #
#                                  REPORTING                                   #
# ---------------------------------------------------------------------------- #

def print_header(title: str, rows: dict[str, object], notes: tuple[str, ...] = ()) -> None:
    """Print the run-provenance block that opens every result file."""
    print("=" * 72)
    print(title)
    for label, value in rows.items():
        print(f"  {label + ':':<15}{value}")
    for note in notes:
        print(f"  {note}")
    print("=" * 72)


def summarise_input_tokens(prompt_tokens: list[int]) -> dict[str, int]:
    """Reduce server-counted input sizes to the median/min/max recorded per level."""
    med, low, high = minmax_median([p for p in prompt_tokens if p])
    return {"input_tokens_median": med, "input_tokens_min": low, "input_tokens_max": high}


def token_shape_line(level: int, shape: dict, output_cap: int,
                     output_per_request: float | None = None) -> str:
    """Format one TOKEN_SHAPE provenance line for the protocol runner to parse.

    Emitted as key=value on its own line rather than as a summary-table column, because the
    runner's table regexes are matched against published result files and adding a column
    would silently stop parsing every one of them. Recording the input size at all is the
    fix for an OUTPUT cap of 1,024 having been read as a 1,024-token INPUT across several
    published reports.
    """
    parts = [
        f"TOKEN_SHAPE concurrency={level}",
        f"input_median={shape.get('input_tokens_median', 0)}",
        f"input_min={shape.get('input_tokens_min', 0)}",
        f"input_max={shape.get('input_tokens_max', 0)}",
    ]
    if output_per_request is not None:
        parts.append(f"output_per_request={output_per_request}")
    parts.append(f"output_cap={output_cap}")
    return " ".join(parts)


def print_token_shape(results: list[dict], output_cap: int, with_output_per_request: bool) -> None:
    """Print a TOKEN_SHAPE line per level, then the plain-language input/output summary."""
    usable = [r for r in results if "error" not in r]
    if not usable:
        return
    for row in usable:
        per_request = row.get("output_tokens_per_request") if with_output_per_request else None
        print(token_shape_line(row["concurrency"], row, output_cap, per_request))
    print(f"  measured input: {usable[0].get('input_tokens_median', 0)} tokens (median, "
          f"server-counted, INCLUDES the chat template)"
          f"   output cap: {output_cap} tokens")
    print("  a profile's target_input_tokens counts the prompt text alone, so it reads lower; "
          "see docs/prompt_tokens/")
    if with_output_per_request:
        print("  tokens/sec above counts OUTPUT tokens only, not input")
