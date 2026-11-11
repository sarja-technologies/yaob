"""Polite, cached client for the Wikipedia and Wikidata action APIs.

Every response is cached on disk keyed by the request, so a corpus build is
resumable and reproducible from the cache. Text fetched here is untrusted data:
it is stored and displayed, never executed or interpreted as instructions.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import os

# Wikimedia allows 10 requests a minute to clients that send no contact details and 200 a minute to clients whose
# User-Agent carries a contact address or URL. No personal contact is sent by default, so the default pace is the
# slow one. Set MPN_CORPUS_CONTACT (an address or URL you are willing to publish to Wikimedia) to build faster.
CONTACT = os.environ.get("MPN_CORPUS_CONTACT", "").strip()
USER_AGENT = f"MPNStudioCorpusBuilder/0.1 ({CONTACT})" if CONTACT else "MPNStudioCorpusBuilder/0.1"
DEFAULT_INTERVAL = 0.4 if CONTACT else 6.4


class WikiClient:
    def __init__(self, cache_dir: Path, min_interval: float = DEFAULT_INTERVAL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self.network_calls = 0
        self.cache_hits = 0
        self._no_maxlag_until = 0.0

    def _path(self, host: str, params: dict) -> Path:
        key = host + "?" + urllib.parse.urlencode(sorted(params.items()))
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def get(self, host: str, **params) -> dict:
        params = {k: str(v) for k, v in params.items()}
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        path = self._path(host, params)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["response"]
        # maxlag asks the servers to refuse us while replication is lagged. When they have just said so repeatedly,
        # reads go out without it for a while (still one at a time at min_interval) instead of waiting twice per request.
        wire = params if time.monotonic() < self._no_maxlag_until else dict(params, maxlag="5")
        url = f"https://{host}/w/api.php"
        body = urllib.parse.urlencode(wire).encode()
        attempt = lag_waits = 0
        while attempt < 10:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.load(r)
                self._last = time.monotonic()
                self.network_calls += 1
                if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                    # the servers ask polite clients to back off; that is not a failure of this request
                    lag_waits += 1
                    if lag_waits >= 2:          # still lagged: this is a read, so ask without the maxlag courtesy (it exists to slow writers)
                        body = urllib.parse.urlencode(params).encode()
                        self._no_maxlag_until = time.monotonic() + 600
                    time.sleep(min(60.0, max(5.0, float(data["error"].get("lag") or 5))))
                    continue
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"API error: {data['error']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({
                    "host": host, "params": params,
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response": data,
                }))
                tmp.replace(path)
                return data
            except urllib.error.HTTPError as e:
                self._last = time.monotonic()
                if e.code in (429, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After")
                    time.sleep((float(retry_after) if retry_after and retry_after.isdigit() else 15 * (attempt + 1)) + 2)
                    attempt += 1
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(10 * (attempt + 1))
                attempt += 1
        raise RuntimeError(f"gave up after retries: {host} {params}")

    def retrieved_at(self, host: str, **params) -> str | None:
        params = {k: str(v) for k, v in params.items()}
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        path = self._path(host, params)
        if path.exists():
            return json.loads(path.read_text()).get("retrieved_at")
        return None
