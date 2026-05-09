"""Shared HTTP helpers for model provider adapters."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class HTTPModelError(RuntimeError):
    """Raised when a provider API request fails."""


def resolve_api_key(*, api_key: str | None = None, api_key_env: str | None = None) -> str:
    """Resolve an API key from an explicit value or an environment variable."""
    if api_key:
        return api_key
    if not api_key_env:
        raise HTTPModelError("No API key or API key environment variable provided.")
    value = os.environ.get(api_key_env)
    if not value:
        raise HTTPModelError(f"Missing API key environment variable: {api_key_env}")
    return value


def post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: int,
    max_retries: int = 8,
    base_backoff_sec: float = 2.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_exc = HTTPModelError(f"{exc.code} {exc.reason}: {body}")
            degraded = exc.code == 400 and "DEGRADED" in body
            retryable = exc.code in (408, 409, 425, 429, 500, 502, 503, 504) or degraded
            if attempt < max_retries - 1 and retryable:
                retry_after_hdr = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    retry_after = float(retry_after_hdr) if retry_after_hdr else 0.0
                except ValueError:
                    retry_after = 0.0
                backoff = base_backoff_sec * (2 ** attempt)
                if exc.code == 429 and retry_after == 0.0:
                    # Some providers enforce long cool-offs without Retry-After.
                    backoff = max(backoff, 30.0)
                if degraded:
                    # Provider-side degradation can last several minutes; back
                    # off hard so we do not burn all retries immediately.
                    backoff = max(backoff, 300.0 * (attempt + 1))
                delay = max(retry_after, backoff)
                time.sleep(delay)
                continue
            raise last_exc from exc
        except urllib.error.URLError as exc:
            last_exc = HTTPModelError(str(exc))
            if attempt < max_retries - 1:
                time.sleep(base_backoff_sec * (2 ** attempt))
                continue
            raise last_exc from exc
        except TimeoutError as exc:
            last_exc = HTTPModelError(f"timeout: {exc}")
            if attempt < max_retries - 1:
                time.sleep(base_backoff_sec * (2 ** attempt))
                continue
            raise last_exc from exc
    raise last_exc or HTTPModelError("post_json exhausted retries")
