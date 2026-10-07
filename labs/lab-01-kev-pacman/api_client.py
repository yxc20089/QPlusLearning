"""Call the notebook-local Kev server and validate decision probabilities."""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call(path, body=None):
    base = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8009").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("KEV_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["KEV_API_KEY"]
    data = None if body is None else json.dumps(body).encode("utf-8")
    started = time.perf_counter()
    try:
        with urlopen(Request(base + path, data=data, headers=headers), timeout=120) as resp:
            value = json.load(resp)
    except HTTPError as exc:
        raise RuntimeError(f"Kev returned HTTP {exc.code}. Check credentials, request schema and the server log.") from None
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("Cannot reach Kev. Check KEV_BASE_URL and whether the notebook-local server is running.") from exc
    return value, (time.perf_counter() - started) * 1000


def call_batch(path, bodies, concurrency=16):
    """Send ordered independent requests to Kev's existing batching queue.

    Kev does not expose a JSON-array bulk endpoint. Concurrent HTTP requests let
    its model thread batch independent states; no game decisions are pipelined.
    """
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= 32:
        raise ValueError('Use between 1 and 32 concurrent independent requests')
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda body: call(path, body), bodies))


def distribution(answer, keys):
    """Check API output before scoring it. API rounds to four decimal places."""
    p = answer.get("probabilities", {})
    if set(p) != set(keys) or any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError("Response probability keys or values do not match the question.")
    if not math.isclose(sum(p.values()), 1, abs_tol=0.02):
        raise ValueError("Response probabilities do not sum to approximately one.")
    if answer.get("choice") not in p or p[answer["choice"]] < max(p.values()) - 1e-4:
        raise ValueError("Response choice is not a highest-probability option.")
    # Undo serialization rounding for metrics only. Keep raw responses in artifacts.
    return {k: v / sum(p.values()) for k, v in p.items()}
