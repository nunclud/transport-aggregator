"""
Redis cache-aside helper for read-heavy endpoints (search, routes, metrics).

Set REDIS_URL (e.g. a free Upstash instance) to enable caching. Without it —
or if the connection fails for any reason — `cached()` transparently falls
back to calling `build()` directly on every request, so the API behaves
exactly like the uncached prototype.
"""
from __future__ import annotations
import os
import json
from typing import Any, Callable

_client = None
_checked = False


def _get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis
        client = redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _client = client
    except Exception:
        _client = None
    return _client


def cached(key: str, ttl: int, build: Callable[[], Any]):
    """Return build()'s result, transparently cached in Redis under `key`."""
    client = _get_client()
    if client is None:
        return build()

    try:
        raw = client.get(key)
        if raw is not None:
            return json.loads(raw)
    except Exception:
        pass

    value = build()
    if value is not None:
        try:
            client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
    return value
