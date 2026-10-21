#!/usr/bin/env python3
"""Persist encrypted daily files on dedicated monthly Git branches, without Releases permissions."""
import base64
import json
import os
from pathlib import Path
import requests


def persist_day(session, repository, day, archive, report):
    base = f'https://api.github.com/repos/{repository}'
    branch = f'collected-data-{day[:7]}'

    def api(method, path, **kwargs):
        response = session.request(method, base + path, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()

    elements = []
    for file in (archive, report):
        data = file.read_bytes()
        if file == archive and not data:
            raise ValueError('Empty encrypted archive')
        blob = api('POST', '/git/blobs', json={'content': base64.b64encode(data).decode(), 'encoding': 'base64'})
        elements.append({'path': f'{"encrypted" if file == archive else "reports"}/{file.name}',
                         'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
    for attempt in range(8):
        response = session.get(base + '/git/ref/heads/' + branch, timeout=60)
        if response.status_code == 404:
            parent = None
            tree_body = {'tree': elements}
        else:
            response.raise_for_status()
            parent = response.json()['object']['sha']
            commit = api('GET', '/git/commits/' + parent)
            tree_body = {'base_tree': commit['tree']['sha'], 'tree': elements}
        tree = api('POST', '/git/trees', json=tree_body)
        commit = api('POST', '/git/commits', json={
            'message': f'Archive fresh encrypted netkeirin data for {day}',
            'tree': tree['sha'], 'parents': [parent] if parent else []})
        # Non-forced updates preserve concurrent shards. A conflict simply
        # rebuilds the tree on the new parent, retaining every other day.
        if parent:
            response = session.patch(base + '/git/refs/heads/' + branch,
                                     json={'sha': commit['sha'], 'force': False}, timeout=60)
        else:
            response = session.post(base + '/git/refs', json={
                'ref': 'refs/heads/' + branch, 'sha': commit['sha']}, timeout=60)
        if response.status_code in (409, 422):
            continue
        response.raise_for_status()
        return f'https://github.com/{repository}/blob/{branch}/encrypted/{archive.name}'
    raise RuntimeError('Concurrent archive updates exceeded retry limit')


def main():
    token = os.environ['GH_TOKEN']
    repository = os.environ['GITHUB_REPOSITORY']
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + token,
                            'Accept': 'application/vnd.github+json'})
    summary_path = Path('backfill_summary.json')
    summary = json.loads(summary_path.read_text())
    failures = []
    for row in summary['days']:
        if row['status'] == 'failed' or not row.get('races'):
            continue
        day = row['date']
        archive = Path(f'encrypted/netkeirin-{day}.sqlite.gpg')
        report = Path(f'backfill_reports/netkeirin-{day}.json')
        try:
            row['archive'] = persist_day(session, repository, day, archive, report)
            row['durable'] = True
            print('DURABLE_ENCRYPTED_ARCHIVE', day, row['races'], row['archive'], flush=True)
        except Exception as exc:
            row['durable'] = False
            failures.append(day)
            print('DURABLE_ARCHIVE_ERROR', day, type(exc).__name__, str(exc)[:300], flush=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    if failures:
        raise SystemExit(f'Durable encrypted Git archival failed for {failures}; checkpoint must not advance these dates')

if __name__ == '__main__':
    main()
