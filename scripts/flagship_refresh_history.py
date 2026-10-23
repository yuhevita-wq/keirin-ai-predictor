#!/usr/bin/env python3
"""Append newly completed race days to a decrypted flagship history cache.

The MODEL IS IMMUTABLE and never imported/retrained by this script.
Publication is denied unless every discovered race has entries, finish data,
payout and no collection errors on each added calendar day.
"""
import argparse
from datetime import date, datetime, timedelta
import json
import logging
from pathlib import Path
import sqlite3
import subprocess
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

def coverage(day):
    with sqlite3.connect(DB_PATH) as con:
        day_races={x[0] for x in con.execute("SELECT race_id FROM races WHERE date=?",(day,))}
        entry_counts=dict(con.execute("""SELECT e.race_id, COUNT(*) FROM entries e
            JOIN races r USING(race_id) WHERE r.date=? GROUP BY e.race_id""",(day,)))
        result_counts=dict(con.execute("""SELECT rr.race_id, COUNT(*) FROM race_results rr
            JOIN races r USING(race_id) WHERE r.date=? AND rr.finish_position IS NOT NULL
            GROUP BY rr.race_id""",(day,)))
        payout_counts=dict(con.execute("""SELECT rp.race_id, COUNT(*) FROM race_payouts rp
            JOIN races r USING(race_id)
            WHERE r.date=? AND rp.nisyatan_payout>=100 AND rp.sanrentan_payout>=100
            GROUP BY rp.race_id""",(day,)))
    return day_races,entry_counts,result_counts,payout_counts

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-days",type=int,default=7)
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    with sqlite3.connect(DB_PATH) as con:
        last=con.execute("SELECT MAX(date) FROM races").fetchone()[0]
        before_count=con.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    if not last or before_count<98000:
        raise RuntimeError("Base encrypted cache is missing or incomplete")
    today=datetime.now(ZoneInfo("Asia/Tokyo")).date()
    end=today-timedelta(days=1)
    first=date.fromisoformat(last)+timedelta(days=1)
    days=(end-first).days+1
    if days<1:
        print("FLAGSHIP_HISTORY_ALREADY_CURRENT",last,flush=True)
        Path("history_refresh_report.json").write_text(json.dumps({"status":"unchanged","last_day":last}),encoding="utf-8")
        return
    if days>args.max_days:
        raise RuntimeError(f"Backlog {days} days > max_days {args.max_days}; manual audit required")
    report={"source":"live netkeirin collector","last_day_before":last,
            "last_day_after":None,"days":[],"races_before":before_count}
    for i in range(days):
        target=(first+timedelta(days=i)).isoformat()
        print("FLAGSHIP_COLLECTING_DAY",target,flush=True)
        proc=subprocess.run([sys.executable,"scripts/collect_training_day.py","--date",target],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,check=False)
        if proc.returncode:
            raise RuntimeError(f"Collection failed {target}: {proc.stderr[-900:]}")
        path=Path("collection_report.json")
        if not path.is_file():
            raise RuntimeError("Collector did not emit report for "+target)
        collected=json.loads(path.read_text(encoding="utf-8"))
        race_ids,entries,results,payouts=coverage(target)
        seen=len(race_ids)
        discovered=int(collected.get("discovered_races") or 0)
        if (not discovered or seen!=discovered or collected.get("errors") or
            set(race_ids)!=set(entries) or set(race_ids)!=set(results) or
            set(race_ids)!=set(payouts) or
            any(entries[r]<5 or entries[r]>9 or results[r]!=entries[r]
                for r in race_ids)):
            raise RuntimeError(f"Incomplete day {target}: discovered={discovered} "
                               f"db_races={seen} collector_errors={collected.get('errors')} "
                               f"missing_entry={list(race_ids-set(entries))[:4]} "
                               f"missing_result={list(race_ids-set(results))[:4]} "
                               f"missing_payout={list(race_ids-set(payouts))[:4]}")
        report["days"].append({"date":target,"races":seen})
        print("FLAGSHIP_DAY_COMPLETE",target,seen,flush=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if con.execute("PRAGMA integrity_check").fetchone()[0]!="ok":
            raise RuntimeError("Merged DB failed integrity_check")
        report["races_after"]=con.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        report["last_day_after"]=con.execute("SELECT MAX(date) FROM races").fetchone()[0]
    assert report["last_day_after"]==end.isoformat()
    Path("history_refresh_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FLAGSHIP_HISTORY_REFRESH_OK",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
