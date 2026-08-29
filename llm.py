"""Groq chat completions.

Two failure modes are handled differently on purpose:

  * 429 rate limit  -> wait out `retry-after` and retry THE SAME model. A free
                       key rate-limits constantly, and silently switching model
                       on a 429 would mean half your answers come from a
                       different model than the other half.
  * anything else   -> one attempt on the fallback model, then give up.
"""
from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

import config
import tracing

log = logging.getLogger("llm")


@lru_cache(maxsize=1)
def _client():
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise SystemExit(
            "GROQ_API_KEY is not set. Put it in .env (see .env.example) or export it."
        )
    return Groq(api_key=config.GROQ_API_KEY)


@tracing.traceable(run_type="llm", name="groq.chat")
def chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> str:
    primary = model or config.GROQ_MODEL
    kwargs = {
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
    }
    if seed is not None:
        kwargs["seed"] = seed
    # Reasoning models (gpt-oss) spend the token budget on a hidden reasoning
    # channel before writing any content. Left at the default effort they routinely
    # exhaust max_tokens and return content='' with finish_reason='length'.
    if config.GROQ_REASONING_EFFORT and "gpt-oss" in primary:
        kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT

    try:
        return _complete(primary, system, user, kwargs)
    except Exception as exc:  # noqa: BLE001
        _abort_if_fatal(exc)
        if primary == config.GROQ_FALLBACK_MODEL:
            raise
        log.warning("model %s failed (%s); falling back to %s", primary, exc, config.GROQ_FALLBACK_MODEL)
        return _complete(config.GROQ_FALLBACK_MODEL, system, user, kwargs)


def _abort_if_fatal(exc: Exception) -> None:
    """Stop the whole run on an error no other model or retry could fix.

    A bad key fails identically on every model, so falling back just doubles the
    doomed requests and buries the real cause under a wall of warnings. Callers
    deliberately catch `Exception` to keep going on a per-chunk failure — SystemExit
    is a BaseException, so it escapes those handlers and stops the run.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status not in (401, 403):
        return
    raise SystemExit(
        f"Groq rejected the API key ({status}).\n"
        f"  Check GROQ_API_KEY in .env — it is currently "
        f"{'empty' if not config.GROQ_API_KEY else config.GROQ_API_KEY[:7] + '…'}\n"
        f"  In Docker, .env is read when the container is CREATED. After editing it:\n"
        f"      docker compose up -d --force-recreate\n"
        f"  To ingest without any LLM calls at all:\n"
        f"      python ingest.py data/chunks.json --no-questions"
    )


def _complete(model: str, system: str, user: str, kwargs: dict) -> str:
    last: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = _client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            if not content:
                raise RuntimeError(_empty_content_reason(model, choice, kwargs))
            # `_complete` is not itself traced, so this lands on the `chat` run —
            # which is the one that should say which model actually answered, since
            # a fallback means that is not the model the caller asked for.
            #
            # Token counts must go under the literal key `usage_metadata`, with
            # LangSmith's own field names: its _extract_usage() reads only that key
            # and ignores everything else. Logging Groq's prompt_tokens /
            # completion_tokens as plain metadata shows the numbers on the run but
            # leaves the token and cost panels empty, which looks like the API
            # returned no usage at all.
            usage = getattr(response, "usage", None)
            tracing.add_metadata(
                model=model,
                attempts=attempt + 1,
                finish_reason=choice.finish_reason,
                # ls_provider / ls_model_name are what LangSmith prices the run on
                ls_provider="groq",
                ls_model_name=model,
                usage_metadata=_usage_metadata(usage),
            )
            return content
        except Exception as exc:  # noqa: BLE001
            wait = _rate_limit_wait(exc)
            if wait is None:
                raise
            last = exc
            log.warning(
                "rate limited on %s, waiting %.1fs (attempt %d/%d)",
                model, wait, attempt + 1, config.LLM_MAX_RETRIES,
            )
            time.sleep(wait)
    raise RuntimeError(f"{model}: still rate limited after {config.LLM_MAX_RETRIES} attempts") from last


def _usage_metadata(usage) -> dict | None:
    """Groq's usage object in the shape LangSmith prices runs from.

    Returns None when the response carried no usage, so the key is simply absent
    rather than present-but-zero — a zero would read as "this call was free".
    """
    if usage is None:
        return None
    fields = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    present = {k: int(v) for k, v in fields.items() if v is not None}
    return present or None


def _empty_content_reason(model: str, choice, kwargs: dict) -> str:
    """Explain an empty completion instead of quietly returning ''.

    The usual cause is a reasoning model exhausting max_tokens on its hidden
    reasoning channel, which surfaces as finish_reason='length' with no content.
    Returning '' from here would put empty summaries in the database and print
    blank answers, with nothing anywhere saying why.
    """
    if choice.finish_reason == "length":
        reasoning = getattr(choice.message, "reasoning", None)
        if reasoning:
            return (
                f"{model} used all {kwargs.get('max_tokens')} tokens on reasoning and "
                f"produced no answer. Raise LLM_MAX_TOKENS, lower "
                f"GROQ_REASONING_EFFORT (currently {config.GROQ_REASONING_EFFORT!r}), "
                f"or use a non-reasoning model such as llama-3.3-70b-versatile."
            )
        return f"{model} hit the {kwargs.get('max_tokens')}-token limit before writing anything."
    return f"{model} returned empty content (finish_reason={choice.finish_reason!r})."


_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*(ms|s|m)", re.I)


def _rate_limit_wait(exc: Exception) -> float | None:
    """Seconds to wait if this is a 429, else None (meaning: not retryable here)."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = str(exc)
    if status != 429 and "rate limit" not in text.lower():
        return None

    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = headers.get(key)
        if raw:
            try:
                return min(float(str(raw).rstrip("s")), 120.0)
            except ValueError:
                pass

    # Groq puts the wait in the message body: "...try again in 8.5s"
    match = _RETRY_AFTER.search(text)
    if match:
        value, unit = float(match.group(1)), match.group(2).lower()
        seconds = value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
        return min(seconds + 0.5, 120.0)

    return 10.0
