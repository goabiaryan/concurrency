from __future__ import annotations

import random
from typing import List, Tuple

_CHAT_OPENERS = [
    "You are a concise assistant. ",
    "Answer clearly and briefly. ",
    "For this exercise, respond helpfully. ",
]

_TOPICS = [
    "summarize best practices for observability in distributed systems",
    "explain idempotency for HTTP APIs",
    "outline steps to debug elevated p99 latency",
    "compare batching vs continuous batching for LLM serving",
    "describe GPU memory fragmentation symptoms",
    "list risks when colocating training and inference",
    "explain backpressure in request queues",
    "what to log for LLM inference SLOs",
    "how to validate autoscaling under bursty traffic",
    "tradeoffs of FP16 vs INT8 for inference",
]

_REASONING = [
    "Think step by step, then give the final answer in one short paragraph.",
    "First list assumptions, then conclude.",
    "Use bullet reasoning internally, output only the conclusion.",
]


def _words(n: int, rng: random.Random) -> str:
    base = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "
    parts: List[str] = []
    while sum(len(p) for p in parts) < n * 6:
        parts.append(base.split()[rng.randint(0, len(base.split()) - 1)])
    text = " ".join(parts)
    words = text.split()
    return " ".join(words[: max(1, n)])


def estimate_tokens(text: str) -> int:
    """Rough English token estimate without tiktoken."""
    return max(1, len(text) // 4)


def build_prompt_for_target_tokens(
    rng: random.Random,
    target_tokens: int,
    style_weights: Tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> Tuple[str, str]:
    """
    Returns (user_message, style_tag).
    style: chat | completion | reasoning
    """
    r = rng.random()
    chat_w, comp_w, reas_w = style_weights
    if r < chat_w:
        style = "chat"
        opener = rng.choice(_CHAT_OPENERS)
        topic = rng.choice(_TOPICS)
        filler_words = max(0, int(target_tokens * 0.75) - 20)
        body = _words(filler_words, rng)
        msg = f"{opener}{topic}. Context: {body}"
        return msg, style
    if r < chat_w + comp_w:
        style = "completion"
        seed = rng.choice(
            [
                "The incident report begins:",
                "Release notes draft:",
                "Runbook excerpt:",
                "Customer ticket:",
            ]
        )
        filler_words = max(0, int(target_tokens * 0.75) - 15)
        msg = f"{seed}\n\n{_words(filler_words, rng)}"
        return msg, style
    style = "reasoning"
    topic = rng.choice(_TOPICS)
    instr = rng.choice(_REASONING)
    filler_words = max(0, int(target_tokens * 0.75) - 25)
    msg = f"{instr} Task: {topic}. Details: {_words(filler_words, rng)}"
    return msg, style


def random_prompt_token_target(rng: random.Random, low: int, high: int) -> int:
    return rng.randint(int(low), int(high))


def random_prompt_token_from_tiers(
    rng: random.Random,
    tiers: list[dict],
) -> tuple[int, str]:
    """
    tiers: [{"weight": 0.45, "min": 50, "max": 300, "name": "short"}, ...]
    Returns (target_tokens, tier_name).
    """
    if not tiers:
        return 128, "default"
    tw = sum(max(0.0, float(t.get("weight", 1))) for t in tiers)
    if tw <= 0:
        tw = float(len(tiers))
        weights = [1.0 / tw] * len(tiers)
    else:
        weights = [max(0.0, float(t.get("weight", 0))) / tw for t in tiers]
    r = rng.random()
    acc = 0.0
    chosen = tiers[-1]
    for t, w in zip(tiers, weights):
        acc += w
        if r <= acc:
            chosen = t
            break
    lo = int(chosen.get("min", 50))
    hi = int(chosen.get("max", lo + 1))
    if hi < lo:
        lo, hi = hi, lo
    name = str(chosen.get("name", f"{lo}-{hi}"))
    return rng.randint(lo, hi), name


def trim_to_est_tokens(text: str, target: int) -> str:
    """Coarsely trim to approximate token budget."""
    if estimate_tokens(text) <= target:
        return text
    # ~4 chars per token
    max_chars = max(20, target * 4)
    return text[:max_chars]


def messages_for_chat_api(user_text: str) -> List[dict]:
    return [{"role": "user", "content": user_text}]
