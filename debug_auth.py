"""
Run this directly to verify the API key loads correctly and the request headers
are sent as expected.

    py debug_auth.py
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
print(f".env path:  {env_path}")
print(f".env exists: {env_path.exists()}")

load_dotenv(dotenv_path=env_path, override=True)

raw_key = os.getenv("PRT_API_KEY", "")
stripped_key = raw_key.strip()

print(f"\nRaw key from os.getenv: {repr(raw_key)}")
print(f"Stripped key:           {repr(stripped_key)}")
print(f"Key length:             {len(stripped_key)}")

headers = {"x-token": stripped_key}
print(f"\nHeaders being sent: {headers}")

url = "https://api.proranktracker.com/v3/util/urls"
print(f"\nGET {url}")

try:
    resp = requests.get(url, headers=headers, timeout=15)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")
except requests.RequestException as e:
    print(f"Request error: {e}")
