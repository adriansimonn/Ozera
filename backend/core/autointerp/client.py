"""
Claude through the Anthropic SDK: one feature's requests in parallel (the Messages API), and many
features' through the Message Batches API (half price, most batches done within an hour, at
most 24 h).

Replies are reduced to what the pipeline needs (text, stop reason, usage), so it can run on a
fake client in tests. A refusal (stop_reason "refusal") is returned like any reply and never
retried: the pipeline marks the feature refused, and its tokens are still billed.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

from .budget import COUNT_GROUP_BYTES, COUNT_MARGIN_TOKENS, MAX_RETRIES, MODELS, PARALLEL_REQUESTS, REQUEST_TIMEOUT_SECONDS

# A batch holds at most this many requests (the API's limit) and bytes (the API allows 256 MB;
# smaller uploads are quicker to retry)
BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 64 << 20
# Uploading one batch can take a while
BATCH_CREATE_TIMEOUT_SECONDS = 600


@dataclass
class Usage:
    """Tokens a request was billed for, by type."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        return cls(**{key: int(data.get(key, 0)) for key in cls().to_dict()})

    @classmethod
    def from_api(cls, usage) -> "Usage":
        return cls(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        )


@dataclass
class Reply:
    """A request's outcome."""

    model: str
    text: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    error: Optional[str] = None
    # No usage came back, but the request may have run and been billed (a timeout, a lost
    # connection, or the deadline cut it off): charge its upper bound
    maybe_billed: bool = False

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def json(self) -> Optional[dict]:
        """The structured output, or None if the reply has none or it doesn't parse."""
        if self.error or self.refused or not self.text:
            return None
        try:
            data = json.loads(self.text)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None


def request_params(model_id: str, system: str, user: str, schema: dict, max_tokens: int) -> dict:
    """A Messages API request with a structured output, at the model's lowest thinking setting."""
    m = MODELS[model_id]
    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    if m.effort is not None:
        output_config["effort"] = m.effort
    params = {
        "model": model_id,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": output_config,
    }
    if m.thinking is not None:
        params["thinking"] = m.thinking
    return params


def reply_from_message(message, model_id: str) -> Reply:
    text = next((block.text for block in message.content if getattr(block, "type", None) == "text"), None)
    return Reply(model_id, text, message.stop_reason, Usage.from_api(message.usage))


class ClaudeClient:
    """The SDK's async client (an injected one in tests), with parallel sends under a deadline."""

    def __init__(self, api_key: Optional[str] = None, sdk=None, parallel: int = PARALLEL_REQUESTS):
        if sdk is None:
            import anthropic

            sdk = anthropic.AsyncAnthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
        self._sdk = sdk
        self.parallel = parallel

    async def count(self, params: dict) -> int:
        """A request's exact input tokens (the count_tokens endpoint)."""
        body = {key: value for key, value in params.items() if key != "max_tokens"}
        return (await self._sdk.messages.count_tokens(**body)).input_tokens

    async def count_all(self, requests: list[dict]) -> list[int]:
        semaphore = asyncio.Semaphore(self.parallel)

        async def one(params: dict) -> int:
            async with semaphore:
                return await self.count(params)

        return list(await asyncio.gather(*(one(p) for p in requests)))

    async def count_total(self, requests: list[dict], overheads: Optional[dict[str, int]] = None) -> dict[str, int]:
        """
        At least the input tokens of these requests together, by model, with far fewer calls
        than one per request (for a job's thousands of requests). Requests sharing a frame (the
        model, system prompt and output settings) are counted as that frame's tokens with a
        one-token message, once, and their messages' texts together, as blocks of one message
        of at most COUNT_GROUP_BYTES; each request is allowed COUNT_MARGIN_TOKENS for tokens
        that merge across the joins. A group the API finds too long is split and counted again.
        Frames' tokens are kept in overheads, when given, for calls with the same frames.
        """
        import anthropic

        frames: dict[str, dict] = {}
        texts: dict[str, list[str]] = {}
        for params in requests:
            frame = {key: value for key, value in params.items() if key not in ("messages", "max_tokens")}
            key = json.dumps(frame, sort_keys=True)
            frames.setdefault(key, frame)
            texts.setdefault(key, []).extend(m["content"] for m in params["messages"])
        semaphore = asyncio.Semaphore(self.parallel)

        async def counted(params: dict) -> int:
            async with semaphore:
                return await self.count(params)

        async def group(model_id: str, contents: list[str]) -> int:
            blocks = [{"type": "text", "text": text} for text in contents]
            try:
                return await counted({"model": model_id, "messages": [{"role": "user", "content": blocks}]})
            except anthropic.BadRequestError:
                if len(contents) < 2:
                    raise
                half = len(contents) // 2
                return sum(await asyncio.gather(group(model_id, contents[:half]), group(model_id, contents[half:])))

        totals: dict[str, int] = {}
        groups: list[tuple[str, Awaitable[int]]] = []
        for key, frame in frames.items():
            model_id, contents = frame["model"], texts[key]
            if overheads is None or key not in overheads:
                overhead = await counted({**frame, "messages": [{"role": "user", "content": "x"}]})
                if overheads is not None:
                    overheads[key] = overhead
            else:
                overhead = overheads[key]
            totals[model_id] = totals.get(model_id, 0) + len(contents) * (overhead + COUNT_MARGIN_TOKENS)
            groups += [(model_id, group(model_id, part)) for part in _byte_groups(contents, COUNT_GROUP_BYTES)]
        for (model_id, _), n in zip(groups, await asyncio.gather(*(g for _, g in groups))):
            totals[model_id] += n
        return totals

    async def send(self, params: dict) -> Reply:
        import anthropic

        model_id = params["model"]
        try:
            message = await self._sdk.messages.create(**params)
        except anthropic.APIStatusError as e:
            # The API answered with an error: nothing was billed
            return Reply(model_id, error=f"{e.status_code}: {e.message}")
        except anthropic.APIError as e:
            return Reply(model_id, error=e.__class__.__name__, maybe_billed=True)
        return reply_from_message(message, model_id)

    async def send_all(self, requests: list[dict], deadline: Optional[float] = None) -> list[Reply]:
        """
        Every request, PARALLEL_REQUESTS at a time; those not done by the deadline (a
        time.monotonic() value) are cancelled, and count as maybe billed if they had started.
        """
        semaphore = asyncio.Semaphore(self.parallel)
        started: set[int] = set()

        async def one(i: int, params: dict) -> Reply:
            async with semaphore:
                started.add(i)
                return await self.send(params)

        tasks = [asyncio.ensure_future(one(i, p)) for i, p in enumerate(requests)]
        if not tasks:
            return []
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        out = []
        for i, (params, task) in enumerate(zip(requests, tasks)):
            if task in done:
                out.append(task.result())  # send() returns API errors; anything else is a bug and raises
            else:
                out.append(Reply(params["model"], error="Ran out of time", maybe_billed=i in started))
        return out

    # Message Batches

    async def submit_batches(self, requests: list[tuple[str, dict]]) -> list[str]:
        """Submit (custom_id, params) requests as one batch or more (see split_batches); returns their IDs."""
        return [await self.submit_batch(part) for part in split_batches(requests)]

    async def submit_batch(self, requests: list[tuple[str, dict]]) -> str:
        """Submit (custom_id, params) requests, within the API's limits, as one batch; returns its ID."""
        sdk = self._sdk.with_options(timeout=BATCH_CREATE_TIMEOUT_SECONDS) if hasattr(self._sdk, "with_options") else self._sdk
        batch = await sdk.messages.batches.create(
            requests=[{"custom_id": custom_id, "params": params} for custom_id, params in requests],
        )
        return batch.id

    async def submit_parts(
        self,
        requests: Iterable[tuple[str, dict]],
        done: dict[int, str],
        on_created: Callable[[int, str, int], None],
    ) -> list[str]:
        """
        Submit requests (an iterable, built as they're sent) in parts of at most BATCH_MAX_REQUESTS
        and BATCH_MAX_BYTES, one batch each. Parts in done ({part: batch ID}, from an earlier
        attempt) aren't sent again, so a stage that stopped part-way resumes without paying
        twice; on_created(part, batch ID, requests) records each new batch as soon as it exists.
        Returns every part's batch ID.
        """
        ids: list[str] = []
        part: list[tuple[str, dict]] = []
        size = 0

        async def send() -> None:
            n = len(ids)
            if n in done:
                ids.append(done[n])
            else:
                batch_id = await self.submit_batch(part)
                on_created(n, batch_id, len(part))
                ids.append(batch_id)

        for custom_id, params in requests:
            n = _request_bytes(custom_id, params)
            if part and (len(part) >= BATCH_MAX_REQUESTS or size + n > BATCH_MAX_BYTES):
                await send()
                part, size = [], 0
            part.append((custom_id, params))
            size += n
        if part:
            await send()
        return ids

    async def batch_status(self, batch_id: str) -> dict:
        """{"status": "in_progress" | "canceling" | "ended", "counts": {...}}."""
        batch = await self._sdk.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        return {
            "status": batch.processing_status,
            "counts": {
                "processing": counts.processing,
                "succeeded": counts.succeeded,
                "errored": counts.errored,
                "canceled": counts.canceled,
                "expired": counts.expired,
            },
        }

    async def cancel_batch(self, batch_id: str) -> None:
        """Stop a batch: requests not yet processed are canceled (and not billed)."""
        await self._sdk.messages.batches.cancel(batch_id)

    async def batch_replies(self, batch_id: str, models: "dict[str, str] | str") -> dict[str, Reply]:
        """
        An ended batch's replies by custom_id (models: each custom_id's model, or the one model of
        every request). Errored, canceled and expired requests aren't billed.
        """
        out: dict[str, Reply] = {}
        async for item in await self._sdk.messages.batches.results(batch_id):
            model_id = models if isinstance(models, str) else models.get(item.custom_id, "")
            result = item.result
            if result.type == "succeeded":
                out[item.custom_id] = reply_from_message(result.message, model_id)
            elif result.type == "errored":
                out[item.custom_id] = Reply(model_id, error=f"errored: {getattr(result.error, 'type', '')}")
            else:
                out[item.custom_id] = Reply(model_id, error=result.type)
        return out


def _request_bytes(custom_id: str, params: dict) -> int:
    return len(json.dumps({"custom_id": custom_id, "params": params}, ensure_ascii=False).encode()) + 2


def split_batches(requests: list[tuple[str, dict]]) -> list[list[tuple[str, dict]]]:
    """Requests in batches of at most BATCH_MAX_REQUESTS and BATCH_MAX_BYTES (as JSON)."""
    parts: list[list[tuple[str, dict]]] = []
    size = 0
    for custom_id, params in requests:
        n = _request_bytes(custom_id, params)
        if not parts or len(parts[-1]) >= BATCH_MAX_REQUESTS or size + n > BATCH_MAX_BYTES:
            parts.append([])
            size = 0
        parts[-1].append((custom_id, params))
        size += n
    return parts


def _byte_groups(texts: list[str], max_bytes: int) -> list[list[str]]:
    """Texts in consecutive groups of at most max_bytes (UTF-8) each (a longer text alone)."""
    groups: list[list[str]] = []
    size = max_bytes
    for text in texts:
        n = len(text.encode())
        if size + n > max_bytes:
            groups.append([])
            size = 0
        groups[-1].append(text)
        size += n
    return groups
