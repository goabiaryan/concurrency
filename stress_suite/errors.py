from __future__ import annotations


def classify_error(err: str | None, ok: bool) -> str:
    if ok:
        return "ok"
    if err is None or err == "":
        return "unknown"
    el = err.lower()
    if "chaos_simulated" in el or "chaos_" in el:
        return "chaos_simulated"
    if "timeout" in el or "timed out" in el or "read timeout" in el:
        return "timeout"
    if "connect" in el or "connection refused" in el or "connection reset" in el:
        return "connection"
    if "http_429" in el or "429" in el[:20]:
        return "rate_limit"
    if el.startswith("http_4") or "http_4" in el[:12]:
        return "http_4xx"
    if el.startswith("http_5") or "http_5" in el[:12]:
        return "http_5xx"
    if "oom" in el or "out of memory" in el or "outofmemory" in el:
        return "oom"
    if "cuda" in el or "cublas" in el or "cudnn" in el:
        return "cuda"
    if "allocate" in el or "allocation" in el:
        return "allocation"
    return "other"
