#!/usr/bin/env python3
"""Read-only diagnostic of how netkeirin exposes historical race IDs."""
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

CASES = [
    ("tachikawa_2023_01_04", "2023010428"),
    ("maebashi_2023_02_26", "2023022622"),
    ("odawara_2026_10_20", "2026102036"),
]
URL = "https://keirin.netkeiba.com/db/race_program/?kaisai_group_id="
AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
out = []
for label, group in CASES:
    record = {"case": label, "kaisai_group_id": group}
    try:
        response = requests.get(URL + group, headers={"User-Agent": AGENT}, timeout=25,
                                allow_redirects=True)
        text = response.text
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.get_text(" ", strip=True)[:160] if soup.title else ""
        ids = sorted(set(re.findall(r"race_id=(\d{12})", text)))
        hrefs = [tag.get("href", "") for tag in soup.find_all("a", href=True)]
        race_links = [urlparse(h).path + ("?race_id=" + h.split("race_id=", 1)[1][:12]
                      if "race_id=" in h else "") for h in hrefs if "race" in h.lower()][:8]
        record.update({
            "http_status": response.status_code,
            "final_url": response.url,
            "html_bytes": len(response.content),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "page_title": title,
            "regex_race_id_count": len(ids),
            "regex_examples": ids[:4],
            "links_with_race_id": sum("race_id=" in h for h in hrefs),
            "query_race_links_sample": race_links,
            "kaisai_group_id_seen": text.count("kaisai_group_id"),
            "race_id_text_occurrences": text.count("race_id"),
            "form_count": len(soup.find_all("form")),
            "blocked_indicators": [needle for needle in ["Access Denied", "Forbidden", "Too Many Requests", "Cloudflare", "Bot Detection", "not found", "ページが見つかりません"] if needle.lower() in text.lower()],
        })
    except requests.RequestException as exc:
        record["request_error"] = type(exc).__name__
        record["error_summary"] = str(exc)[:200]
    out.append(record)
    time.sleep(3)
target = Path("reports/netkeirin_discovery_diagnostic.json")
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps({"tests": out}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"tests": out}, ensure_ascii=False))
