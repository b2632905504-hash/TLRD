"""
OpenAI-compatible API teacher for generating chat completions.

Provides ``generate_chat_outputs`` which accepts a list of conversation
message-lists and returns a list of generated text outputs using an
OpenAI-compatible chat API (OpenAI, Azure, vLLM serve, etc.).

Requests are issued concurrently via ``asyncio`` + ``AsyncOpenAI`` with a
configurable concurrency limit and per-request retry logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _generate_single(
    client,  # AsyncOpenAI
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    semaphore: asyncio.Semaphore,
    idx: int,
    max_retries: int,
) -> tuple[int, str]:
    """Generate a single chat completion with retry logic."""
    async with semaphore:
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                )
                text = response.choices[0].message.content or ""
                return idx, text
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "request %d failed (attempt %d/%d): %s – retrying in %ds",
                        idx,
                        attempt + 1,
                        max_retries + 1,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
        logger.error(
            "request %d failed after %d attempts: %s",
            idx,
            max_retries + 1,
            last_exc,
        )
        return idx, ""


async def _generate_all(
    messages_list: List[List[Dict[str, str]]],
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    timeout: Optional[float],
    max_retries: int,
    concurrency: int,
    log_dir: Optional[str],
) -> List[str]:
    from openai import AsyncOpenAI

    client_kwargs: Dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    elif os.environ.get("OPENAI_API_KEY"):
        client_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _generate_single(
            client=client,
            messages=msgs,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            semaphore=semaphore,
            idx=i,
            max_retries=max_retries,
        )
        for i, msgs in enumerate(messages_list)
    ]

    # Prepare streaming log file – each completed result is flushed to disk
    # immediately so that partial progress survives crashes.
    log_file = None
    if log_dir:
        log_path = Path(log_dir) / "response.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        logger.info("streaming responses to %s", log_path)

    results: List[Optional[str]] = [None] * len(messages_list)
    done_count = 0
    total = len(tasks)
    log_interval = max(1, total // 20)

    try:
        for coro in asyncio.as_completed(tasks):
            idx, text = await coro
            results[idx] = text
            done_count += 1

            # Stream this result to log file immediately
            if log_file is not None:
                record = {"idx": idx, "done": done_count, "total": total, "text": text}
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()

            if done_count % log_interval == 0 or done_count == total:
                logger.info("progress: %d/%d completions done", done_count, total)
    finally:
        if log_file is not None:
            log_file.close()

    await client.close()
    return [r if r is not None else "" for r in results]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_chat_outputs(
    messages_list: List[List[Dict[str, str]]],
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_completion_tokens: int = 16384,
    timeout: Optional[float] = None,
    max_retries: int = 2,
    concurrency: int = 32,
    log_dir: Optional[str] = None,
) -> List[str]:
    """Generate chat completions for a batch of message lists.

    Uses an OpenAI-compatible chat completions API with async concurrency.

    Each completed result is streamed to ``<log_dir>/response.log`` (JSONL)
    as soon as it arrives, so partial progress survives crashes.

    Args:
        messages_list: List of conversation message lists, each containing
            dicts with ``role`` and ``content`` keys.
        model: Model name / deployment to request.
        base_url: Optional API base URL (overrides ``OPENAI_BASE_URL``).
        api_key: Optional API key (defaults to ``OPENAI_API_KEY`` env var).
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_completion_tokens: Maximum tokens per completion.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries per failed request.
        concurrency: Maximum number of concurrent requests.
        log_dir: Optional directory to write ``response.log`` into.

    Returns:
        List of generated text strings, one per input message list.
        Failed requests return an empty string ``""``.
    """
    logger.info(
        "generating %d completions via OpenAI API (model=%s, concurrency=%d)",
        len(messages_list),
        model,
        concurrency,
    )

    # If there is already a running event loop (e.g. Jupyter), use
    # nest_asyncio or create a new thread. Otherwise use asyncio.run().
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = _generate_all(
        messages_list=messages_list,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        max_retries=max_retries,
        concurrency=concurrency,
        log_dir=log_dir,
    )

    if loop is not None and loop.is_running():
        # Already inside an event loop – run in a new thread to avoid
        # "cannot call asyncio.run() while another loop is running".
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
