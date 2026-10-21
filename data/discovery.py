"""Discover published IDs from links, using the historical calendar's event starts."""
import re
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup
from config import NETKEIRIN_BASE_URL


def race_ids_from_html(html):
    ids = set()
    for a in BeautifulSoup(html, 'lxml').select('a[href]'):
        for value in parse_qs(urlparse(a['href']).query).get('race_id', []):
            if re.fullmatch(r'[0-9]{12}', value):
                ids.add(value)
    return sorted(ids)


def calendar_rows(scraper, day):
    url = (f'{NETKEIRIN_BASE_URL}/race/race_calendar/'
           f'?kaisai_year={day.year}&kaisai_month={day.month}&kaisai_type=1')
    soup = BeautifulSoup(scraper._get(url), 'lxml')
    rows = {}
    for a in soup.select('a[href]'):
        q = parse_qs(urlparse(a['href']).query)
        ds, venue = q.get('kaisai_date', [''])[0], q.get('jyocd', [''])[0]
        if re.fullmatch(r'[0-9]{8}', ds) and re.fullmatch(r'[0-9]{2}', venue):
            rows[(ds, venue)] = a.get_text(' ', strip=True)
    if not rows:
        raise ValueError('Historical calendar has no published meeting links')
    return rows


def discover_day(scraper, ds, allowed_venues):
    from scripts.init_db import get_race_ids_from_program
    day = datetime.strptime(ds, '%Y%m%d')
    rows = calendar_rows(scraper, day)
    if day.day <= 7:
        rows.update(calendar_rows(scraper, day.replace(day=1) - timedelta(days=1)))
    venues = sorted(v for (d, v) in rows if d == ds and v in allowed_venues)
    if not venues:
        raise ValueError(f'No published meetings for {ds}; refusing zero-race progress')
    ids, failures = set(), []
    for venue in venues:
        try:
            openings = sorted((d for (d, v), label in rows.items()
                               if v == venue and d <= ds and '初日' in label), reverse=True)
            candidates = [d for d in openings if (day - datetime.strptime(d, '%Y%m%d')).days <= 6]
            if not candidates:
                candidates = [(day - timedelta(days=n)).strftime('%Y%m%d') for n in range(7)]
            matching = []
            for opening in candidates:
                matching = [r for r in get_race_ids_from_program(scraper, opening + venue)
                            if r.startswith(ds + venue)]
                if matching:
                    break
            if not matching:
                raise ValueError('Published meeting has no race IDs')
            ids.update(matching)
        except Exception as exc:
            failures.append({'venue_code': venue, 'reason': type(exc).__name__})
    return ids, failures
