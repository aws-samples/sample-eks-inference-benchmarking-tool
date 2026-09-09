"""Mock OpenAI-compatible server for verifying harness request formation locally, so adapter changes
can be checked on a laptop before they spend GPU time. Nothing on the measurement path imports this
module and no results file is ever produced against it. Two behaviours are deliberately hostile
because a friendly mock hides bugs: a request carrying no output cap gets roughly 2,000 completion
tokens, mimicking a real model generating to EOS so a cap-propagation bug shows up locally as a
conspicuously wrong output_per_request, and MOCK_FAIL_EVERY=N returns HTTP 500 for every Nth
request per worker, deterministically rather than randomly, so a check can break a run on purpose
and prove the success-floor exclusion fires. Both /v1/chat/completions and /v1/completions are
served, mirroring the two BENCH_ENDPOINT values a protocol can pin, and usage figures are real word
counts so TOKEN_SHAPE parsing sees plausible server-side accounting."""
import asyncio
import itertools
import json
import os
import time

from aiohttp import web

MODEL_ID = os.environ.get("MOCK_MODEL_ID", "mock-model")
PORT = int(os.environ.get("MOCK_PORT", "8199"))
FAIL_EVERY = int(os.environ.get("MOCK_FAIL_EVERY", "0"))

# Long enough that an uncapped request yields far more tokens than any profile's output budget.
COMPLETION = "The answer, considered carefully, is that it depends on the workload. " * 182

_request_counter = itertools.count(1)


def _completion_tokens(body: dict) -> int:
    """Resolve the effective output length the way a real server would.

    vLLM honors `max_completion_tokens` (current OpenAI name) and `max_tokens`
    (legacy name); absence of both means generate to EOS — here, the full canned
    completion.
    """
    cap = body.get("max_completion_tokens", body.get("max_tokens"))
    words = COMPLETION.split()
    return min(int(cap), len(words)) if cap else len(words)


def _should_fail() -> bool:
    """True on every FAIL_EVERY-th request when failure injection is enabled."""
    return FAIL_EVERY > 0 and next(_request_counter) % FAIL_EVERY == 0


async def models(_request):
    return web.json_response({"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "mock"}]})


async def _stream_chat(request, body: dict, ptok: int, ctok: int) -> web.StreamResponse:
    """SSE streaming response, shaped like vLLM's: delta chunks, then a final chunk
    carrying usage, then [DONE]. Exists so streaming harness modes (trace replay) have
    wire-level coverage locally instead of meeting their first real stream in-cluster.
    """
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    words = COMPLETION.split()[:ctok]
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": body.get("model", MODEL_ID)}
    for start in range(0, len(words), 8):  # a few tokens per chunk, like a real server
        chunk = dict(base, choices=[{"index": 0, "finish_reason": None,
                     "delta": {"content": " ".join(words[start:start + 8]) + " "}}])
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(0.002)
    final = dict(base,
                 choices=[{"index": 0, "delta": {}, "finish_reason": "length"}],
                 usage={"prompt_tokens": ptok, "completion_tokens": ctok,
                        "total_tokens": ptok + ctok})
    await response.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
    await response.write_eof()
    return response


async def chat(request):
    body = await request.json()
    if _should_fail():
        return web.json_response({"error": "injected failure (MOCK_FAIL_EVERY)"},
                                 status=500)
    await asyncio.sleep(0.05)
    prompt = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
    ptok = max(1, len(prompt.split()))
    ctok = _completion_tokens(body)
    if body.get("stream"):
        return await _stream_chat(request, body, ptok, ctok)
    return web.json_response({
        "id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", MODEL_ID),
        "choices": [{"index": 0, "message": {"role": "assistant",
                     "content": " ".join(COMPLETION.split()[:ctok])},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                  "total_tokens": ptok + ctok},
    })


async def completions(request):
    body = await request.json()
    if _should_fail():
        return web.json_response({"error": "injected failure (MOCK_FAIL_EVERY)"},
                                 status=500)
    await asyncio.sleep(0.05)
    ptok = max(1, len(str(body.get("prompt", "")).split()))
    ctok = _completion_tokens(body)
    return web.json_response({
        "id": "cmpl-mock", "object": "text_completion", "created": int(time.time()),
        "model": body.get("model", MODEL_ID),
        "choices": [{"index": 0, "text": " ".join(COMPLETION.split()[:ctok]),
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                  "total_tokens": ptok + ctok},
    })


def build_app() -> web.Application:
    """Assemble the route table; separated from run for in-process test use."""
    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/completions", completions)
    return app


if __name__ == "__main__":
    web.run_app(build_app(), host="127.0.0.1", port=PORT, print=None)
