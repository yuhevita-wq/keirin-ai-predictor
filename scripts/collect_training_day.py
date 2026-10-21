#!/usr/bin/env python3
"""Collect a day's netkeirin result, entry, payout data using upstream parsers."""
import argparse
import json
import sys
import logging
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="", help="YYYY-MM-DD; default yesterday JST")
    p.add_argument("--max-races", type=int, default=0, help="Pilot only: 0=all races")
    args = p.parse_args()
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    day = date.fromisoformat(args.date) if args.date else today - timedelta(days=1)
    if day < date(2022, 1, 1) or day >= today or args.max_races < 0:
        raise SystemExit("Date must be 2022-01-01 through yesterday JST; max-races >= 0")

    import config
    from config import VELODROME_CODES
    from data.scraper import KeirinScraper
    from db.schema import (
        init_db, get_connection, insert_race, insert_rider,
        insert_result, insert_entry, insert_payout,
    )
    from scripts.init_db import get_race_ids_from_program, scrape_db_result

    config.SCRAPE_DELAY = max(config.SCRAPE_DELAY, 2.5)
    init_db()
    scraper = KeirinScraper()
    ds = day.strftime("%Y%m%d")
    race_ids = set()
    out_of_date_ids = set()
    errors = 0
    missing = []
    program_errors = []
    from data.discovery import discover_day
    import data.scraper as scraper_module
    scraper_module.SCRAPE_DELAY = max(scraper_module.SCRAPE_DELAY, 2.5)
    race_ids, program_errors = discover_day(scraper, ds, VELODROME_CODES)
    errors += len(program_errors)

    race_list = sorted(race_ids)
    if args.max_races:
        race_list = race_list[:args.max_races]
    counts = {"races": 0, "entries": 0, "results": 0, "payouts": 0}
    with get_connection() as conn:
        for race_id in race_list:
            try:
                res = scrape_db_result(scraper, race_id)
                entry = scraper.scrape_race_entry(race_id)
                results = res.get("results") or []
                entries = entry.get("entries") or []
                meta = res if res.get("date") and res.get("velodrome") else entry
                if not results or not entries or not meta.get("date") or not meta.get("velodrome"):
                    reason = (
                        "missing_results" if not results else
                        "missing_entries" if not entries else "missing_race_metadata"
                    )
                    missing.append({"race_id": race_id, "reason": reason})
                    logger.warning("Missing %s: %s", race_id, reason)
                    errors += 1
                    continue
                insert_race(conn, dict(meta, race_id=race_id, rider_count=len(entries or results)))
                for row in entries:
                    rid = row.get("rider_id")
                    if not rid:
                        continue
                    insert_rider(conn, {
                        "rider_id": rid, "name": row.get("rider_name") or "unknown",
                        "class": row.get("class"), "prefecture": row.get("prefecture"),
                        "period": row.get("period"),
                    })
                    insert_entry(conn, dict(row, race_id=race_id))
                    counts["entries"] += 1
                for row in results:
                    rid = row.get("rider_id")
                    if not rid:
                        continue
                    insert_rider(conn, {
                        "rider_id": rid, "name": row.get("rider_name") or "unknown",
                        "class": row.get("class"), "prefecture": row.get("prefecture"),
                        "period": row.get("period"),
                    })
                    insert_result(conn, dict(row, race_id=race_id))
                    counts["results"] += 1
                # scripts.init_db.scrape_db_result() does not populate payouts.
                # Parse the HTML already cached by that parser first, then
                # fall back to the separate result page when necessary.
                payout = res.get("payouts") or {}
                if not payout:
                    html = scraper._get(
                        f"{config.NETKEIRIN_BASE_URL}/db/result/?race_id={race_id}"
                    )
                    payout = scraper._parse_payout_table(BeautifulSoup(html, "lxml"))
                if not payout:
                    try:
                        alternate = scraper.scrape_race_result(race_id)
                        payout = alternate.get("payouts") or {}
                    except Exception as exc:
                        logger.warning("Payout page %s: %s", race_id, exc)
                if payout.get("nisyatan_payout") or payout.get("sanrentan_payout"):
                    insert_payout(conn, dict(payout, race_id=race_id))
                    counts["payouts"] += 1
                # Keep every parsed field (age, period, comment, line, all payouts)
                # alongside the upstream model tables in the encrypted SQLite.
                conn.execute('CREATE TABLE IF NOT EXISTS source_records (race_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
                conn.execute('INSERT OR REPLACE INTO source_records VALUES (?, ?)',
                             (race_id, json.dumps({'result': res, 'entry': entry, 'payouts': payout}, ensure_ascii=False)))
                conn.commit()
                counts["races"] += 1
            except Exception as exc:
                conn.rollback()
                logger.warning("Race %s: %s", race_id, exc)
                missing.append({"race_id": race_id, "reason": type(exc).__name__})
                errors += 1

    # Counts must describe committed rows, including rollback after a malformed race.
    with get_connection() as conn:
        for key, table in [('races', 'races'), ('entries', 'entries'),
                           ('results', 'race_results'), ('payouts', 'race_payouts')]:
            counts[key] = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    report = {
        "source": "netkeirin", "date": day.isoformat(),
        "discovered_races": len(race_ids), **counts,
        "ignored_other_date_races": len(out_of_date_ids),
        "ignored_other_date_race_ids": sorted(out_of_date_ids),
        "errors": errors, "pilot_limit": args.max_races,
        "missing_payout_races": counts["races"] - counts["payouts"],
        "missing_races": missing, "program_errors": program_errors,
        "complete": bool(
            not args.max_races and race_ids
            and counts["races"] == len(race_ids)
            and counts["entries"] >= counts["races"] * 3
            and counts["payouts"] == counts["races"] and not errors
        ),
    }
    report["status"] = "complete" if report["complete"] else "partial"
    # Partial shards can help later backfill, but MUST NOT be treated as
    # complete training dates or a valid backtest set.
    Path("collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if not counts["races"] or not counts["entries"] or not counts["results"]:
        raise SystemExit("No usable races, entries and results; refusing to publish an empty shard")
    if not report["complete"]:
        logger.warning(
            "Partial day: %d / %d discovered races. Encrypted archival allowed; "
            "exclude from model training until missing races have been audited.",
            counts["races"], len(race_ids),
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
