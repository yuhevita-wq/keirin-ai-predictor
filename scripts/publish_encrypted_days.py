#!/usr/bin/env python3
"""Require durable Releases before a date can advance the collection checkpoint."""
import json
import os
import subprocess
from pathlib import Path


def gh(*args):
    subprocess.run(['gh', *args], check=True)


def main():
    summary_path = Path('backfill_summary.json')
    summary = json.loads(summary_path.read_text())
    failures = []
    for row in summary['days']:
        if row['status'] == 'failed' or not row.get('races'):
            continue
        day = row['date']
        tag = f'training-data-{day[:7]}'
        archive = Path(f'encrypted/netkeirin-{day}.sqlite.gpg')
        report = Path(f'backfill_reports/netkeirin-{day}.json')
        try:
            exists = subprocess.run(['gh', 'release', 'view', tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not exists:
                # Another shard may create the monthly release at the same time.
                created = subprocess.run(['gh', 'release', 'create', tag, '--target', os.environ.get('GITHUB_SHA', 'master'), '--title', tag, '--notes', 'Fresh-source AES256 encrypted SQLite. Keep the existing collector key private.']).returncode
                if created:
                    gh('release', 'view', tag)
            gh('release', 'upload', tag, str(archive), str(report), '--clobber')
            row['durable'] = True
            row['archive'] = f'{tag}/{archive.name}'
        except subprocess.CalledProcessError:
            row['durable'] = False
            failures.append(day)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    if failures:
        raise SystemExit(f'Durable release upload failed for {failures}; checkpoint must not advance these dates')

if __name__ == '__main__':
    main()
