"""Async OpenAI-compatible chat client (streaming) for TTFT / TPOT."""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from stress_suite import prompts
from stress_suite.errors import classify_error


def _tpot(total_s: float, ttft_s: float, out_tokens: int) -> float:
    decode_tokens = max(1, out_tokens - 1)
    decode_s = max(1e-9, total_s - ttft_s)
    return decode_s / decode_tokens


async def _apply_network_chaos(
    rng: random.Random,
    chaos: dict[str, Any] | None,
) -> str | None:
    """Returns error string to short-circuit request, or None to continue."""
    if not chaos:
        return None
    el = chaos.get("extra_latency_ms")
    if el is not None:
        if isinstance(el, (list, tuple)) and len(el) >= 2:
            lo, hi = int(el[0]), int(el[1])
            if hi >= lo >= 0 and (lo > 0 or hi > 0):
                await asyncio.sleep(rng.randint(lo, hi) / 1000.0)
    ap = float(chaos.get("connection_abort_probability") or 0)
    if ap > 0 and rng.random() < ap:
        return "chaos_simulated_connection_abort"
    ap2 = float(chaos.get("request_drop_probability") or 0)
    if ap2 > 0 and rng.random() < ap2:
        return "chaos_simulated_request_drop"
    return None


async def streaming_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    rng: random.Random,
    chaos: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{str(base_url).rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    ts_utc = datetime.now(timezone.utc).isoformat()
    chaos_err = await _apply_network_chaos(rng, chaos)
    if chaos_err:
        return {
            "ok": False,
            "error": chaos_err,
            "error_class": classify_error(chaos_err, False),
            "ts_utc_start": ts_utc,
            "ttft_s": 0.0,
            "total_s": 0.0,
            "decode_s": 0.0,
            "output_tokens_est": 0,
            "usage_prompt_tokens": None,
            "usage_completion_tokens": None,
            "tpot_s_per_tok": float("nan"),
            "output_tokens_per_s": 0.0,
            "prompt_tokens_est": prompts.estimate_tokens(
                messages[0].get("content", "") if messages else ""
            ),
            "prefill_proxy_s": None,
            "decode_phase_s": None,
        }

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    parts: list[str] = []
    err: Optional[str] = None
    usage_prompt: Optional[int] = None
    usage_completion: Optional[int] = None

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            timeout=httpx.Timeout(timeout),
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                err = f"http_{resp.status_code}:{body[:500]!r}"
            else:
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        err = str(chunk["error"])
                        break
                    u = chunk.get("usage")
                    if isinstance(u, dict):
                        if "prompt_tokens" in u:
                            usage_prompt = int(u["prompt_tokens"])
                        if "completion_tokens" in u:
                            usage_completion = int(u["completion_tokens"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter()
                        parts.append(piece)
    except Exception as e:  # noqa: BLE001
        err = str(e)

    t1 = time.perf_counter()
    text = "".join(parts)
    out_est = max(1, prompts.estimate_tokens(text))
    if usage_completion is not None and usage_completion > 0:
        out_tok = usage_completion
    else:
        out_tok = out_est
    ttft_s = (ttft - t0) if ttft is not None else (t1 - t0)
    total_s = t1 - t0
    decode_s = max(0.0, total_s - ttft_s)
    ok = err is None and bool(text)
    out_per_s = (out_tok / total_s) if total_s > 0 else 0.0
    # Prefill vs decode (proxy): TTFT ~ prefill+queue; remainder ~ decode
    prefill_proxy_s = ttft_s
    decode_phase_s = decode_s
    return {
        "ok": ok,
        "error": err,
        "error_class": classify_error(err, ok),
        "ts_utc_start": ts_utc,
        "ttft_s": ttft_s,
        "total_s": total_s,
        "decode_s": decode_s,
        "output_tokens_est": out_est,
        "usage_prompt_tokens": usage_prompt,
        "usage_completion_tokens": usage_completion,
        "output_tokens_effective": out_tok,
        "tpot_s_per_tok": _tpot(total_s, ttft_s, out_tok),
        "output_tokens_per_s": out_per_s,
        "prompt_tokens_est": prompts.estimate_tokens(
            messages[0].get("content", "") if messages else ""
        ),
        "prefill_proxy_s": prefill_proxy_s,
        "decode_phase_s": decode_phase_s,
    }


def _pick_prompt_target(
    rng: random.Random,
    phase_cfg: dict[str, Any] | None,
    default_low: int,
    default_high: int,
) -> tuple[int, str]:
    if phase_cfg and phase_cfg.get("prompt_token_tiers"):
        return prompts.random_prompt_token_from_tiers(
            rng, list(phase_cfg["prompt_token_tiers"])
        )
    pr = (phase_cfg or {}).get("prompt_token_est_range")
    if pr is not None and len(pr) >= 2:
        return prompts.random_prompt_token_target(rng, int(pr[0]), int(pr[1])), "uniform"
    return prompts.random_prompt_token_target(rng, default_low, default_high), "uniform"


def _chaos_for_phase(global_chaos: dict | None, phase_cfg: dict | None) -> dict | None:
    if phase_cfg and phase_cfg.get("network_chaos") is not None:
        c = phase_cfg.get("network_chaos")
        return c if c else None
    return global_chaos if global_chaos else None


async def warmup_container(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    n: int,
    max_tokens: int,
    gap_ms: float,
    timeout: float,
    rng: random.Random,
    phase_cfg: dict[str, Any] | None = None,
    global_chaos: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chaos = _chaos_for_phase(global_chaos, phase_cfg)
    out: list[dict[str, Any]] = []
    for i in range(n):
        if gap_ms > 0:
            await asyncio.sleep(gap_ms / 1000.0)
        target, tier = _pick_prompt_target(rng, phase_cfg, 32, 96)
        user, style = prompts.build_prompt_for_target_tokens(rng, target)
        user = prompts.trim_to_est_tokens(user, target)
        msgs = prompts.messages_for_chat_api(user)
        r = await streaming_chat_completion(
            client,
            base_url,
            model,
            msgs,
            max_tokens=max_tokens,
            timeout=timeout,
            rng=rng,
            chaos=chaos,
        )
        r["phase"] = "warmup"
        r["container_port_hint"] = base_url
        r["seq"] = i
        r["prompt_tier"] = tier
        r["style"] = style
        out.append(r)
    return out


async def wave_for_container(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    concurrent: int,
    total_requests: int,
    jitter_ms: tuple[int, int],
    prompt_range: tuple[int, int],
    output_range: tuple[int, int],
    timeout: float,
    rng: random.Random,
    wave_idx: int,
    cid: int,
    phase_cfg: dict[str, Any] | None = None,
    global_chaos: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chaos = _chaos_for_phase(global_chaos, phase_cfg)
    sem = asyncio.Semaphore(max(1, concurrent))
    results: list[dict[str, Any]] = []
    count_lock = asyncio.Lock()
    res_lock = asyncio.Lock()
    sent = 0

    async def one() -> None:
        nonlocal sent
        async with sem:
            async with count_lock:
                if sent >= total_requests:
                    return
                sent += 1
                my_idx = sent
            jlow, jhigh = jitter_ms
            await asyncio.sleep(rng.randint(jlow, jhigh) / 1000.0)
            target, tier = _pick_prompt_target(
                rng,
                phase_cfg,
                prompt_range[0],
                prompt_range[1],
            )
            user, style = prompts.build_prompt_for_target_tokens(rng, target)
            user = prompts.trim_to_est_tokens(user, target)
            o_low, o_high = output_range
            max_out = rng.randint(int(o_low), int(o_high))
            msgs = prompts.messages_for_chat_api(user)
            r = await streaming_chat_completion(
                client,
                base_url,
                model,
                msgs,
                max_tokens=max_out,
                timeout=timeout,
                rng=rng,
                chaos=chaos,
            )
            r["phase"] = f"wave_{wave_idx}"
            r["wave"] = wave_idx
            r["container_id"] = cid
            r["request_idx"] = my_idx
            r["style"] = style
            r["prompt_tier"] = tier
            async with res_lock:
                results.append(r)

    tasks = [asyncio.create_task(one()) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    return results


async def sustained_rps(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    duration_s: float,
    rps: float,
    prompt_range: tuple[int, int],
    output_range: tuple[int, int],
    timeout: float,
    rng: random.Random,
    cid: int,
    phase_cfg: dict[str, Any] | None = None,
    global_chaos: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chaos = _chaos_for_phase(global_chaos, phase_cfg)
    results: list[dict[str, Any]] = []
    t_end = time.perf_counter() + duration_s
    interval = 1.0 / max(0.01, rps)
    next_t = time.perf_counter()
    seq = 0
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if now < next_t:
            await asyncio.sleep(min(0.05, next_t - now))
            continue
        merged = dict(phase_cfg or {})
        target, tier = _pick_prompt_target(rng, merged, prompt_range[0], prompt_range[1])
        user, style = prompts.build_prompt_for_target_tokens(rng, target)
        user = prompts.trim_to_est_tokens(user, target)
        o_low, o_high = output_range
        max_out = rng.randint(int(o_low), int(o_high))
        msgs = prompts.messages_for_chat_api(user)
        r = await streaming_chat_completion(
            client,
            base_url,
            model,
            msgs,
            max_tokens=max_out,
            timeout=timeout,
            rng=rng,
            chaos=chaos,
        )
        r["phase"] = "sustained"
        r["container_id"] = cid
        r["seq"] = seq
        r["style"] = style
        r["prompt_tier"] = tier
        results.append(r)
        seq += 1
        next_t += interval
    return results
