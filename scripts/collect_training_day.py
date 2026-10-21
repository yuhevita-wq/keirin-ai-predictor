#!/usr/bin/env python3
"""Collect a day's netkeirin result, entry, payout data using upstream parsers."""
import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
    errors = 0
    for venue in VELODROME_CODES:
        try:
            race_ids.update(get_race_ids_from_program(scraper, ds + venue))
        except Exception as exc:
            logger.warning("Program %s: %s", venue, exc)
            errors += 1

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
                if not results or not meta.get("date") or not meta.get("velodrome"):
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
                    })
                    insert_result(conn, dict(row, race_id=race_id))
                    counts["results"] += 1
                payout = res.get("payouts") or {}
                if payout.get("nisyatan_payout") or payout.get("sanrentan_payout"):
                    insert_payout(conn, dict(payout, race_id=race_id))
                    counts["payouts"] += 1
                conn.commit()
                counts["races"] += 1
            except Exception as exc:
                conn.rollback()
                logger.warning("Race %s: %s", race_id, exc)
                errors += 1

    report = {
        "source": "netkeirin", "date": day.isoformat(),
        "discovered_races": len(race_ids), **counts,
        "errors": errors, "pilot_limit": args.max_races,
        "complete": bool(
            not args.max_races and race_ids
            and counts["races"] == len(race_ids) and not errors
        ),
    }
    Path("collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if not report["complete"]:
        raise SystemExit("Partial/empty collection; no shard will be published")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
