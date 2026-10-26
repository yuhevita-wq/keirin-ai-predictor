#!/usr/bin/env python3
"""Apply immutable Ura HaruQ V1 (and frozen research-only selector) to 2026-10-26.
Read-only operational experiment. Does NOT change either AI or public websites.
Only as-of racecard fields + the saved 2026 model; NEVER read today's outcomes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from config import DB_PATH
from db.schema import get_connection, insert_race, insert_rider, insert_entry
from data.scraper import KeirinScraper
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura_haruq_independent.frozen_ura_haru_v1.URA_HARU_V1_FROZEN_RULE import formation, POLICY_ID

DATE = "20261026"
DATE_SQL = "2026-10-26"
# Confirmed on date-specific live schedule. Count 72 races across 7 tracks.
TRACKS = {"27":("京王閣",12),"36":("小田原",12),"73":("小松島",12),
          "12":("青森",9),"34":("川崎",9),"42":("名古屋",11),
          "55":("和歌山",7)}
FROZEN_SHA = "b0891724787f3c300a2b58d0d19cc6cf4ff421225ea53c81b94ab3add7194818"
MODEL_SHA = "8921afb65e87132a4ccc39eee2869423a1dc29cc2d27bdb1e8287ca98adb6c47"
OUT = Path("ura_haruq_independent/diagnostics/live_20261026")
FIELDS = ["race_id","venue","race_number","start_time","data_status","reason",
          "line_pattern","score_gap","predicted_bikes","ranked_scores",
          "v1_result","tickets","points","stake_yen","skill_spread",
          "third_skill_mean","research_selector","decision_at_utc"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_num(n):
    try:
        x = float(n)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    args=ap.parse_args()
    rule=Path("ura_haruq_independent/frozen_ura_haru_v1/URA_HARU_V1_FROZEN_RULE.py")
    if sha(rule)!=FROZEN_SHA or sha(args.model)!=MODEL_SHA:
        raise RuntimeError("IMMUTABLE_MODEL_OR_V1_HASH_NOT_MATCHING")
    OUT.mkdir(parents=True,exist_ok=True)
    model=LGBMRanker()
    model.load(args.model)
    builder=FeatureBuilder()
    if len(model.feature_names)!=44 or set(model.feature_names)!=set(builder.feature_names):
        raise RuntimeError("44_FEATURE_SCHEMA_MISMATCH")
    scraper=KeirinScraper()
    conn=get_connection()
    # Safety barrier: disallow race results on or after the decision day. Temporary DB only.
    bad=conn.execute(
        "SELECT count(*) FROM race_results rr JOIN races r ON rr.race_id=r.race_id WHERE r.date>=?",
        (DATE_SQL,)).fetchone()[0]
    if bad:
        raise RuntimeError("FORBIDDEN_CURRENT_OR_FUTURE_RESULT_ROWS_IN_HISTORY")
    rows=[]
    for jyo,(venue,count) in TRACKS.items():
        for n in range(1,count+1):
            rid=f"{DATE}{jyo}{n:02d}"
            record=dict.fromkeys(FIELDS,"")
            record.update(race_id=rid,venue=venue,race_number=n,points=0,
                          stake_yen=0,decision_at_utc=datetime.now(timezone.utc).isoformat())
            try:
                card=scraper.scrape_race_entry(rid)
                entries=card.get("entries",[])
                record["start_time"]=str(card.get("start_time") or "")
                if len(entries)!=7 or len({e.get("bike_number") for e in entries})!=7:
                    record.update(data_status="SKIP",reason=f"NON_7_OR_INCOMPLETE_ENTRIES_{len(entries)}")
                    rows.append(record)
                    continue
                if all(not e.get("class") for e in entries) or any(
                    str(e.get("class") or "").upper().startswith("L") for e in entries):
                    record.update(data_status="SKIP",reason="GIRLS")
                    rows.append(record)
                    continue
                lines={int(e["bike_number"]):str(e.get("line_group") or "").strip() for e in entries}
                sizes=sorted(Counter(lines.values()).values(),reverse=True)
                record["line_pattern"]="-".join(map(str,sizes))
                if not all(lines.values()):
                    record.update(data_status="SKIP",reason="LINE_UNKNOWN")
                    rows.append(record)
                    continue
                if sizes!=[2,2,2,1]:
                    record.update(data_status="SKIP",reason="NOT_2221")
                    rows.append(record)
                    continue
                insert_race(conn,{
                    "race_id":rid,"date":DATE_SQL,"velodrome":card.get("velodrome") or venue,
                    "race_number":n,"race_name":card.get("race_name"),
                    "grade":card.get("grade"),"rider_count":7,"start_time":card.get("start_time")})
                for e in entries:
                    if not e.get("rider_id"):
                        raise RuntimeError("MISSING_RIDER_ID")
                    insert_rider(conn,{"rider_id":e["rider_id"],"name":e.get("rider_name") or "?",
                              "class":e.get("class"),"prefecture":e.get("prefecture")})
                    insert_entry(conn,e)
                conn.commit()
                feats=builder.build_race_features(rid,DATE_SQL,conn)
                if len(feats)!=7 or set(feats.rider_id)!={e["rider_id"] for e in entries}:
                    raise RuntimeError("INCOMPLETE_44_FEATURE_ROWS")
                x=feats[model.feature_names].apply(pd.to_numeric,errors="coerce").fillna(0)
                scores=model.predict(x)
                feats["score"]=scores
                if not np.isfinite(feats["score"]).all():
                    raise RuntimeError("NONFINITE_FORECAST_SCORE")
                bikes_by_id={e["rider_id"]:int(e["bike_number"]) for e in entries}
                feats["bike"]=feats.rider_id.map(bikes_by_id)
                feats=feats.sort_values(["score","bike"],ascending=[False,True],kind="stable")
                bikes=[int(b) for b in feats.bike]
                gap=float(feats.iloc[0].score-feats.iloc[1].score)
                record["score_gap"]=round(gap,8)
                record["predicted_bikes"]="-".join(map(str,bikes))
                record["ranked_scores"]=";".join(f"{v:.8f}" for v in feats.score)
                bets=formation(bikes,lines,gap)
                if not bets:
                    record.update(data_status="OK",v1_result="SKIP",
                                  reason="NOT_LOW" if gap>=1 else "TOP_PAIR_NOT_SAME_LINE",
                                  research_selector="NO_BET")
                    rows.append(record)
                    continue
                if len(bets)!=6:
                    raise RuntimeError("UNEXPECTED_V1_TICKET_COUNT")
                third_bikes={t[2] for t in bets}
                third_skills=[safe_num(e.get("avg_competition_score")) for e in entries
                              if int(e["bike_number"]) in third_bikes]
                skill_list=[safe_num(e.get("avg_competition_score")) for e in entries]
                if any(v is None for v in skill_list+third_skills) or len(third_skills)!=3:
                    record.update(data_status="ERROR",reason="MISSING_COMPETITION_SCORES")
                    rows.append(record)
                    continue
                skill_spread=max(skill_list)-min(skill_list)
                third_mean=sum(third_skills)/len(third_skills)
                selected=(skill_spread<=7.1 and third_mean>95.8767)
                record.update(data_status="OK",v1_result="BET",reason="V1_QUALIFIED",
                              tickets=";".join("-".join(map(str,t)) for t in bets),
                              points=6,stake_yen=600,
                              skill_spread=round(skill_spread,5),
                              third_skill_mean=round(third_mean,5),
                              research_selector="BET" if selected else "NO_BET")
            except Exception as exc:
                record.update(data_status="ERROR",reason=f"{type(exc).__name__}:{str(exc)[:180]}")
            rows.append(record)
            print("LIVE_V1",json.dumps(record,ensure_ascii=False),flush=True)
    with (OUT/"every_race.csv").open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report={"date":DATE,"asof_utc":datetime.now(timezone.utc).isoformat(),
            "model_sha256":MODEL_SHA,"policy_id":POLICY_ID,"frozen_v1_sha256":FROZEN_SHA,
            "selector":"skill_spread<=7.1 & third_skill_mean>95.8767 (research-only)",
            "totals":dict(Counter(r["data_status"] for r in rows)),
            "v1_eligible":sum(r["v1_result"]=="BET" for r in rows),
            "selector_eligible":sum(r["research_selector"]=="BET" for r in rows),
            "source_races":len(rows),"historical_snapshot_only_through":"2026-09-30",
            "warning":"No result or payoff used in decision. May be retrospective for already-started events. No live odds confirmation. Production model unchanged."}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    conn.close()
    print("LIVE_V1_REPORT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
