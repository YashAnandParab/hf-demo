"""LangSmith tracing — optional, and a no-op when it is not configured.

There is no LangChain in this project, so this wraps the bare `langsmith` SDK.
Every decorator below is resolved ONCE at import time:

  * key present and `langsmith` importable -> the real `@traceable`
  * anything else                          -> the identity decorator

which means an unconfigured machine pays nothing at all at call time, and the
pipeline modules can decorate unconditionally without any `if enabled:` noise.

Tracing must never be able to fail a query. A broken endpoint, an expired key or
a missing package degrades to "no traces", never to a raised exception.

Enable it with, in .env:

    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=hf-demo-rag

Set LANGSMITH_TRACING=false to turn it off while leaving the key in place.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

import config

log = logging.getLogger("tracing")

F = TypeVar("F", bound=Callable[..., Any])

_ENABLED = False
_traceable_impl: Callable[..., Callable[[F], F]] | None = None


def _identity_decorator(*_args, **_kwargs):
    """Stand-in for @traceable that returns the function untouched."""

    def decorate(func: F) -> F:
        return func

    # tolerate bare @traceable as well as @traceable(run_type=...)
    if len(_args) == 1 and callable(_args[0]) and not _kwargs:
        return _args[0]
    return decorate


def _setup() -> None:
    global _ENABLED, _traceable_impl

    if not config.LANGSMITH_TRACING:
        log.debug("LangSmith tracing disabled by LANGSMITH_TRACING")
        return
    if not config.LANGSMITH_API_KEY:
        log.debug("LangSmith tracing off: no LANGSMITH_API_KEY set")
        return

    try:
        from langsmith import traceable as _traceable
    except ImportError:
        log.warning(
            "LANGSMITH_API_KEY is set but the langsmith package is not installed; "
            "tracing is off. Install it with: pip install langsmith"
        )
        return

    # The SDK reads these from the environment, so a key supplied through .env
    # (which python-dotenv has already loaded into os.environ) works the same as
    # one exported by hand.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT

    _traceable_impl = _traceable
    _ENABLED = True
    log.info("LangSmith tracing on, project %r", config.LANGSMITH_PROJECT)


_setup()


def enabled() -> bool:
    return _ENABLED


def traceable(*args, **kwargs):
    """`langsmith.traceable` when configured, otherwise a pass-through."""
    if _traceable_impl is None:
        return _identity_decorator(*args, **kwargs)
    return _traceable_impl(*args, **kwargs)


def hide(*keys: str) -> Callable[[dict], dict]:
    """Build a `process_inputs` hook that replaces the named arguments.

    Query embeddings are 1024 floats. Logged verbatim they bury every trace under
    a wall of numbers and tell you nothing you could not get from the dimension.
    """

    def process(inputs: dict) -> dict:
        out = dict(inputs)
        for key in keys:
            value = out.get(key)
            if isinstance(value, (list, tuple)):
                out[key] = f"<{len(value)} floats>"
            elif key in out:
                out[key] = "<hidden>"
        return out

    return process


def vector_output(outputs: Any) -> dict:
    """`process_outputs` hook for a function returning an embedding.

    Logs the dimension instead of the vector. The SDK may hand this the raw return
    value or a `{"output": ...}` wrapper depending on version, so both are handled.
    """
    value = outputs.get("output", outputs) if isinstance(outputs, dict) else outputs
    try:
        return {"dimensions": len(value)}
    except TypeError:
        return {"dimensions": 0}


def add_metadata(**fields: Any) -> None:
    """Attach key/values to the run currently being traced. No-op when off.

    Used for the things that are not arguments or return values but are what you
    actually want to filter traces on later — hit counts per arm, how far the
    reranker moved things, which model answered.
    """
    if not _ENABLED:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.metadata.update(fields)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not attach trace metadata: %s", exc)


def flush() -> None:
    """Ship anything still queued. Called before a CLI process exits.

    Traces are batched on a background thread, so a one-shot `python query.py`
    can otherwise exit before its own trace has been sent.
    """
    if not _ENABLED:
        return
    try:
        from langsmith import Client

        Client().flush()
    except Exception as exc:  # noqa: BLE001
        log.debug("langsmith flush failed: %s", exc)
