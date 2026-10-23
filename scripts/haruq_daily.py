#!/usr/bin/env python3
"""Discover one published day and run the existing frozen predictor unchanged."""
import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def discover(scraper, day):
    from bs4 import BeautifulSoup
    from config import NETKEIRIN_BASE_URL, VELODROME_CODES
    from data.discovery import race_ids_from_html
    base = NETKEIRIN_BASE_URL + '/race/api_race_list.html'
    query = '?pid=api_race_list&input=UTF-8&output=html&kaisai_date=' + day
    html = scraper._get(base + query)
    soup = BeautifulSoup(html, 'lxml')
    codes = list(dict.fromkeys(re.findall(r'btn_jyoname_(\d{2})', html)))
    if not codes:
        raise RuntimeError('No published venues: refusing an empty day')
    venues, failures = [], []
    for code in codes:
        if code not in VELODROME_CODES:
            raise RuntimeError('Unknown venue ' + code)
        url = base + query + '&jyocd=' + code
        try:
            body = scraper._get(url)
            ids = [r for r in race_ids_from_html(body) if r.startswith(day + code)]
            if not ids:
                raise RuntimeError('Published venue has no published race links')
            numbers = [int(r[-2:]) for r in ids]
            if numbers != list(range(1, max(numbers) + 1)):
                raise RuntimeError('Published race list has a gap: ' + str(numbers))
            venues.append({'code': code, 'name': VELODROME_CODES[code],
                           'race_ids': ids, 'source_url': url})
        except Exception as exc:
            failures.append({'code': code, 'name': VELODROME_CODES[code], 'reason': str(exc)})
    return venues, failures


def main():
    from data.scraper import KeirinScraper
    from scripts.haruq_publish_web import export_result
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    req = json.loads(Path(args.request).read_text())
    iso = date.fromisoformat(req['date']).isoformat()
    day = iso.replace('-', '')
    if req.get('scope') != 'all_published_races':
        raise ValueError('An explicit all-published-races request is required')
    scraper = KeirinScraper()
    venues, discovery_errors = discover(scraper, day)
    out = ROOT / 'flagship_daily_results'
    out.mkdir(exist_ok=True)
    public = ROOT / 'docs/haruq/results'
    daily = ROOT / 'docs/haruq/days'
    daily.mkdir(exist_ok=True)
    run = os.environ['GITHUB_RUN_ID']
    commit = os.environ['GITHUB_SHA']
    records = []
    for venue in venues:
        for rid in venue['race_ids']:
            print('HARUQ_DAILY_BEGIN', rid, flush=True)
            item = {'race_id': rid, 'velodrome': venue['name'], 'race_number': int(rid[-2:])}
            try:
                entry = scraper.scrape_race_entry(rid)
                item.update(start_time=entry.get('start_time'), grade=entry.get('grade'),
                            race_name=entry.get('race_name'))
                target = out / (rid + '.json')
                with (out / (rid + '.log')).open('w') as log:
                    process = subprocess.run([sys.executable, 'scripts/flagship_live_predict.py',
                        '--race-id', rid, '--model', args.model, '--out', str(target)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                if process.returncode:
                    lines = (out / (rid + '.log')).read_text()
                    reason = next((x for x in reversed(lines.splitlines()) if x.startswith('RuntimeError:')), lines[-800:])
                    # Only the predictor's existing explicit girls rejection means excluded.
                    status = 'excluded' if 'ガールズケイリンはハルキュー購入対象外' in reason else 'error'
                    item.update(status=status, reason=reason.removeprefix('RuntimeError: '))
                else:
                    export_result(target, public, run, commit)
                    item.update(status='predicted', prediction=json.loads((public / (rid + '.json')).read_text()))
            except Exception as exc:
                item.update(status='error', reason=type(exc).__name__ + ': ' + str(exc))
            records.append(item)
            print('HARUQ_DAILY_DONE', rid, item['status'], flush=True)
    expected = sum(len(v['race_ids']) for v in venues)
    assert len(records) == expected and len({r['race_id'] for r in records}) == expected
    counts = {s: sum(r['status'] == s for r in records) for s in ('predicted', 'excluded', 'error')}
    result = {'date': iso, 'created_at': datetime.now(timezone.utc).isoformat(),
              'run_id': run, 'source_commit': commit, 'venues': venues,
              'discovery_errors': discovery_errors, 'total_races': expected,
              'counts': counts, 'complete': not discovery_errors and counts['error'] == 0,
              'records': records}
    content = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    (daily / (day + '.json')).write_text(content)
    (out / 'day.json').write_text(content)
    print('HARUQ_DAILY_SUMMARY', json.dumps({k: result[k] for k in ('total_races', 'counts', 'complete')}, ensure_ascii=False))
    with open(os.environ.get('GITHUB_STEP_SUMMARY', str(out / 'summary.md')), 'a') as f:
        f.write(f'## HaruQ {iso}\n\nPublished races: {expected}\n\n{counts}\n\nDiscovery errors: {discovery_errors}\n')


if __name__ == '__main__':
    main()
