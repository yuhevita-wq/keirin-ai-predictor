#!/usr/bin/env python3
"""Live HTTP and parser proof for the requested Komatsushima meeting."""
import json
import sys
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.scraper import KeirinScraper
from data.discovery import race_ids_from_html
from config import NETKEIRIN_BASE_URL
from scripts.init_db import scrape_db_result


def main():
    scraper = KeirinScraper()
    pages = []
    idsets = []
    for route in ['db/race_program', 'race/special/race_info']:
        url = f'{NETKEIRIN_BASE_URL}/{route}/?kaisai_group_id=2023090573'
        # Request real HTTP even if a prior collector populated cache.
        import time
        time.sleep(2.5)
        r = scraper.session.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or 'utf-8'
        ids = race_ids_from_html(r.text)
        counts = dict(Counter(i[:8] for i in ids if i[8:10] == '73'))
        assert counts == {'20230905': 12, '20230906': 12, '20230907': 12}, counts
        pages.append({'url': r.url, 'http_status': r.status_code, 'html_bytes': len(r.content),
                      'title': BeautifulSoup(r.text, 'lxml').title.get_text(), 'dates': counts})
        idsets.append(set(ids))
    assert idsets[0] == idsets[1]
    samples = []
    for ds in ['20230905', '20230906', '20230907']:
        rid = ds + '7301'
        res, entry = scrape_db_result(scraper, rid), scraper.scrape_race_entry(rid)
        assert len(res.get('results', [])) >= 3
        assert len(entry.get('entries', [])) >= 3
        assert res.get('payouts', {}).get('sanrentan_payout')
        assert all(2 < e['gear_ratio'] < 6 for e in entry['entries'])
        samples.append({'race_id': rid, 'results': len(res['results']), 'entries': len(entry['entries']),
                        'payouts': res['payouts'], 'first_entry': entry['entries'][0]})
    report = {'pages': pages, 'unique_meeting_races': 36, 'samples': samples}
    target = Path('reports/recovery_live_verification.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
