"""
Pulls all data from the ProRankTracker API and writes it to the local
SQLite cache.  Called when the user clicks "Refresh Data" in the dashboard.

Flow:
  1. GET /v3/util/urls          → upsert all client websites
  2. For each website:
     GET /v3/urls/history/{id}  → upsert terms + rankhistory in one call

Failure handling:
  - Per-client errors are caught, logged, and skipped — one bad client
    never stops the rest of the sync.
  - Transient errors (timeout, 429 rate-limit) are retried up to
    MAX_RETRIES times with exponential back-off.
  - A summary of all failures is returned so the caller can surface them.
"""

import time
import logging
from datetime import date
from requests.exceptions import HTTPError, Timeout, RequestException

from api_client import get_urls, get_url_history
from db import upsert_websites, upsert_terms_and_history, load_websites, display_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATE_FROM  = "2024-02-01"
MAX_RETRIES = 3          # attempts per client before giving up
RETRY_CODES = {429, 500, 502, 503, 504}   # HTTP codes worth retrying
REQUEST_GAP = 0.35       # seconds between requests (~170 req/min, under the 200 limit)


def _fetch_with_retry(url_id: int, date_from: str, date_to: str, name: str) -> dict | None:
    """
    Fetch history for one client with retry/back-off.
    Returns the API data dict on success, or None if all attempts fail.
    """
    import requests
    from api_client import BASE_URL, _headers

    endpoint = f"{BASE_URL}/urls/history/{url_id}"
    params   = {"from": date_from, "to": date_to, "per_page": 10000}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("GET %s  params=%s  (attempt %d/%d)", endpoint, params, attempt, MAX_RETRIES)
            resp = requests.get(endpoint, headers=_headers(), params=params, timeout=60)

            # Surface HTTP errors explicitly so we can inspect the status code
            if not resp.ok:
                log.warning(
                    "  HTTP %d for %r — body: %s",
                    resp.status_code, name, resp.text[:300],
                )
                if resp.status_code in RETRY_CODES and attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    log.info("  Retrying in %ds…", wait)
                    time.sleep(wait)
                    continue
                return None

            body = resp.json()
            if body.get("result") != "success":
                log.warning(
                    "  API error for %r — result=%r  message=%r",
                    name,
                    body.get("result"),
                    body.get("error_message"),
                )
                return None

            return body.get("data", {})

        except Timeout:
            log.warning("  Timeout on attempt %d/%d for %r", attempt, MAX_RETRIES, name)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
        except RequestException as exc:
            log.error("  Request error for %r: %s", name, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    log.error("  Giving up on %r after %d attempts", name, MAX_RETRIES)
    return None


def sync_all(progress_callback=None) -> list[str]:
    """
    Sync all clients.  Returns a list of failure messages (empty = all good).
    """
    today    = date.today().isoformat()
    failures = []

    def _cb(msg: str):
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    # ── 1. Client list ────────────────────────────────────────────────────────
    _cb("Fetching client list from /v3/util/urls …")
    websites = get_urls()
    upsert_websites(websites)
    time.sleep(REQUEST_GAP)

    cached = load_websites()
    total  = len(cached)
    _cb(f"Found {total} clients. Starting history sync…")

    # ── 2. History per client ─────────────────────────────────────────────────
    for i, site in enumerate(cached):
        name = display_name(site)
        _cb(f"[{i+1}/{total}] {name}")

        try:
            data = _fetch_with_retry(site["id"], DATE_FROM, today, name)
        except Exception as exc:
            msg = f"[{i+1}/{total}] Unexpected error for {name!r}: {exc}"
            log.error(msg)
            failures.append(msg)
            time.sleep(REQUEST_GAP)
            continue

        if data is None:
            msg = f"[{i+1}/{total}] No data returned for {name!r} (url_id={site['id']})"
            log.warning(msg)
            failures.append(msg)
            time.sleep(REQUEST_GAP)
            continue

        terms = data.get("terms") or []
        log.info("  → %d terms", len(terms))

        try:
            upsert_terms_and_history(site["id"], terms)
        except Exception as exc:
            msg = f"[{i+1}/{total}] DB write failed for {name!r}: {exc}"
            log.error(msg)
            failures.append(msg)

        time.sleep(REQUEST_GAP)

    if failures:
        _cb(f"Sync finished with {len(failures)} failure(s) — see log for details.")
    else:
        _cb("Sync complete — all clients updated.")

    return failures
