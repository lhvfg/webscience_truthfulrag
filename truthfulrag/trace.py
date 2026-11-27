"""
Per-sample tracing support (added for the Web Science course report).

Two pieces:

1. A ContextVar-based cost counter. ``start_counter()`` is called once per
   sample; every LLM call made inside that asyncio task (including calls
   pushed to worker threads via ``asyncio.to_thread``, which copies the
   context) increments the same dict. Concurrent samples therefore never
   mix their counts.

2. String-matching helpers used to fill the ``has_answer`` /
   ``answer_in_context`` / ``answer_in_graph`` fields. They reuse the
   project's own ``normalize_answer`` so that "contains the answer" means
   the same thing as in the official evaluation.

Field definitions: SEUCourseReportTexTemplate/plan/plan.md, section 4.
"""
from contextvars import ContextVar
from typing import Dict, Iterable, Optional

from .evaluate import normalize_answer

_COUNTER: ContextVar[Optional[Dict]] = ContextVar("TruthfulRAG_trace_counter", default=None)


def start_counter() -> Dict:
    """Create a fresh counter for the current task and return it."""
    counter = {"n_llm_calls": 0, "n_tokens": 0}
    _COUNTER.set(counter)
    return counter


def record_llm_call(n_tokens: int = 0) -> None:
    """Called by the LLM backends after every completion request."""
    counter = _COUNTER.get()
    if counter is not None:
        counter["n_llm_calls"] += 1
        counter["n_tokens"] += int(n_tokens or 0)


def contains_answer(text: Optional[str], answer) -> bool:
    """True if any normalized gold answer occurs in the normalized text."""
    if not text:
        return False
    answers = answer if isinstance(answer, list) else [answer]
    norm_text = f" {normalize_answer(str(text))} "
    for a in answers:
        norm_a = normalize_answer(str(a))
        # Pad with spaces so that "us" does not match inside "campus".
        if norm_a and f" {norm_a} " in norm_text:
            return True
    return False


def any_contains_answer(texts: Iterable[str], answer) -> bool:
    return any(contains_answer(t, answer) for t in texts)
