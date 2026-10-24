#!/usr/bin/env python3
"""Export only pre-race line/group structure from archived history for Ura V6.

Does not read results, payouts, models or live production settings.
Allows external study of whether line shapes explain payout-band composition.
"""
import csv
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB=Path("db/keirin.db")
OUT=Path("ura_haruq_independent/diagnostics/line_structure_2024_2026")
OUT.mkdir(exist_ok=True, parents=True)

def main():
    if not DB.is_file():
        raise RuntimeError("ARCHIVE_DB_NOT_AVAILABLE")
    con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
    columns={row[1] for row in con.execute("PRAGMA table_info(entries)")}
    assert {"race_id","bike_number","line_group","line_role"}<=columns
    sql="""
       SELECT r.race_id, r.date, r.grade, r.round, r.race_name,
              r.velodrome, r.rider_count,
              e.bike_number, e.line_group, e.line_role
       FROM races AS r
       LEFT JOIN entries AS e ON r.race_id=e.race_id
       WHERE r.date >= '2024-01-01' AND r.date < '2026-10-01'
       ORDER BY r.race_id, e.bike_number
    """
    fields=("race_id","date","grade","round","race_name","velodrome",
      "rider_count","entry_count","known_line_riders","known_line_pct",
      "line_pattern","line_count","max_line_size",
      "number_of_solo_lines","race_line_status","line_groups_by_bike",
      "line_roles_by_bike")
    output=OUT/"pre_race_line_features.csv.gz"
    totals=Counter()
    def emit(records, writer):
        if not records:return
        head=records[0]
        meta=dict(zip(("race_id","date","grade","round","race_name",
                       "velodrome","rider_count"),head[:7]))
        entries=[r for r in records if r[7] is not None]
        lines=defaultdict(list);assign=[];roles=[]
        for r in entries:
            bike=int(r[7])
            group=str(r[8] or "").strip()
            role=str(r[9] or "").strip()
            assign.append(f"{bike}:{group}")
            roles.append(f"{bike}:{role}")
            if group:lines[group].append(bike)
        sizes=sorted([len(v) for v in lines.values()],reverse=True)
        n=len(entries);known=sum(sizes)
        if n and known==n and n==meta["rider_count"]:
            status="fully_grouped"
            pattern="-".join(map(str,sizes))
        elif n and known:
            status="partially_grouped"
            pattern=""
        else:
            status="missing_line_groups"
            pattern=""
        writer.writerow({**meta,"entry_count":n,"known_line_riders":known,
          "known_line_pct":round(100*known/n,2) if n else 0,
          "line_pattern":pattern,"line_count":len(lines),
          "max_line_size":max(sizes,default=0),
          "number_of_solo_lines":sizes.count(1),
          "race_line_status":status,
          "line_groups_by_bike":";".join(assign),
          "line_roles_by_bike":";".join(roles)})
        totals["races"]+=1
        totals[status]+=1
    with gzip.open(output,"wt",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        current=None;batch=[]
        for r in con.execute(sql):
            if current is not None and current!=r[0]:
                emit(batch,w);batch=[]
            batch.append(r);current=r[0]
        emit(batch,w)
    con.close()
    (OUT/"report.json").write_text(json.dumps(dict(source="encrypted historical entry snapshots",
       research_only=True,records=dict(totals),
       caveat="Line_group coverage reflects historical entry source. Missing lines are never imputed from race results."),ensure_ascii=False,indent=2))
    print("PRERACE_LINES_EXPORTED",dict(totals),flush=True)

if __name__=="__main__":
    main()
