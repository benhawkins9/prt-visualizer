"""
ProRankTracker API client (v3).

Endpoints used:
  GET /v3/util/urls
      Returns all tracked URLs (clients) in the account.
      Response: { "result": "success", "data": [ {id, url, business_name, ...}, ... ] }

  GET /v3/urls/history/{url_id}?from=YYYY-MM-DD&to=YYYY-MM-DD&per_page=10000
      Returns ranking history for every term tracked under a URL.
      Response: {
        "result": "success",
        "data": {
          "id": "...",
          "url": "...",
          "terms": [
            {
              "url_term_id": 123,
              "name": "keyword phrase",
              "rankhistory": [
                { "checked": "YYYY-MM-DD", "rank": 5 },
                ...
              ],
              ...
            },
            ...
          ]
        }
      }

Authentication: x-token header.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Always load from the .env file sitting next to this script,
# regardless of which directory Streamlit was launched from.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

BASE_URL = "https://api.proranktracker.com/v3"


def _headers() -> dict:
    api_key = os.getenv("PRT_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PRT_API_KEY not set in .env")
    return {"x-token": api_key}


def get_urls() -> list[dict]:
    """Return all tracked URLs (clients) in the account."""
    resp = requests.get(f"{BASE_URL}/util/urls", headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def get_url_history(url_id: int, date_from: str, date_to: str) -> dict:
    """
    Return history data for a single URL (client).

    Returns the full 'data' object:
    {
        "id": "...",
        "url": "...",
        "terms": [ { "url_term_id": ..., "name": "...", "rankhistory": [...] }, ... ]
    }
    """
    resp = requests.get(
        f"{BASE_URL}/urls/history/{url_id}",
        headers=_headers(),
        params={"from": date_from, "to": date_to, "per_page": 10000},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {})
