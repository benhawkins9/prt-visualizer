"""
Diagnostic: tests the history endpoint against the first 5 clients
and prints exactly what the API returns.

    py debug_sync.py
"""

import os
import json
import requests
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

API_KEY  = os.getenv("PRT_API_KEY", "").strip()
BASE_URL = "https://api.proranktracker.com/v3"
HEADERS  = {"x-token": API_KEY}
DATE_FROM = "2024-02-01"
TODAY     = date.today().isoformat()

# ── Fetch client list ─────────────────────────────────────────────────────────
print("Fetching /v3/util/urls …")
r = requests.get(f"{BASE_URL}/util/urls", headers=HEADERS, timeout=30)
print(f"  Status: {r.status_code}")
clients = r.json().get("data", [])
print(f"  Clients returned: {len(clients)}")
print()

# ── Test first 5 clients ──────────────────────────────────────────────────────
for client in clients[:5]:
    url_id   = client["id"]
    url_str  = client.get("url", "")
    biz_name = client.get("business_name", "")
    name     = biz_name.strip() if biz_name and biz_name.strip() else url_str

    endpoint = f"{BASE_URL}/urls/history/{url_id}"
    params   = {"from": DATE_FROM, "to": TODAY, "per_page": 10000}

    print(f"Client: {name}")
    print(f"  url_id   : {url_id}")
    print(f"  GET {endpoint}")
    print(f"  params   : {params}")

    resp = requests.get(endpoint, headers=HEADERS, params=params, timeout=60)
    print(f"  HTTP status : {resp.status_code}")

    try:
        body = resp.json()
    except Exception as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw response: {resp.text[:300]}")
        print()
        continue

    result = body.get("result")
    print(f"  result field: {result}")

    if result == "success":
        data  = body.get("data", {})
        terms = data.get("terms") or []
        print(f"  terms count : {len(terms)}")
        if terms:
            t = terms[0]
            rh = t.get("rankhistory") or []
            print(f"  first term  : {t.get('name')!r}  (term_type={t.get('term_type')!r}, {len(rh)} history rows)")
    else:
        print(f"  ERROR body  : {json.dumps(body)[:400]}")

    print()
