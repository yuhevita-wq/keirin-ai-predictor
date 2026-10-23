#!/usr/bin/env python3
"""Frozen 44-feature LambdaRank exacta F1 payout backtest.

Prespecified original 2車単 strategy: top1 -> predicted rank2..5, four 500-yen
tickets, require score-gap >= 1.5, exclude F2/G2/G3. No threshold tuning.
Payout records are from public KDreams post-race tables, matched on date,
venue, race number and *both* winning bike numbers. Unknown/unmatched races
are excluded, not silently counted as losers.
No 2車単 pre-deadline odds: torigami skip is NOT reproduced.
"""
import csv
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config import CLASS_MAP, DB_PATH, DEFAULT_GEAR_RATIO, get_bet_category
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
log=logging.getLogger("exacta_frozen")
OUT=Path("exacta_outputs");OUT.mkdir(exist_ok=True)
LOW="2025-07-01"
END="2026-04-01"
FROZEN_MODEL=Path("fixed_model/model_lgbm_44_retrained.pkl")
EXCLUDED={"F2","G2","G3"}

def fmt(key):
    return f"{key[0]}-{key[1]}-{key[2]:02d}"

def payout_source():
    """Read official 2車単 payout labels. Do not read off-line historical finish
    as odds or predictors; these are outcome labels for bet settlement ONLY."""
    files=sorted(Path("payouts").glob("*.csv"))
    out=defaultdict(set)
    for f in files:
        count=0
        with f.open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle)
            assert {"race_id","race_date","race_no","ticket_type","combination","payout_yen","status"} <= set(reader.fieldnames)
            for row in reader:
                if row["ticket_type"] != "2車単" or row["status"] != "paid": continue
                try:
                    rid=row["race_id"]
                    d=row["race_date"];code=rid[:2];no=int(row["race_no"])
                    cars=tuple(int(v) for v in re.findall(r"\d+",row["combination"]))
                    yen=int(float(row["payout_yen"]))
                    if len(cars)!=2 or len(set(cars))!=2 or not 100<=yen<=10_000_000:continue
                    if not LOW<=d<END:continue
                    out[(d,code,no)].add((cars[0],cars[1],yen))
                    count+=1
                except (ValueError,TypeError,KeyError):
                    continue
        log.info("SOURCE %s: paid_exacta_rows=%d",f.name,count)
    return out

def prepare_valid(conn):
    return {r[0] for r in conn.execute("""
        SELECT r.race_id FROM races r
        JOIN race_results rr ON r.race_id=rr.race_id
        GROUP BY r.race_id HAVING
          COUNT(*)=r.rider_count AND SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
          AND COUNT(DISTINCT rr.finish_position)=COUNT(*)
    """)}

def block_report(rows,name):
    stake=sum(r["stake"] for r in rows);paid=sum(r["paid"] for r in rows)
    races=len(rows);hits=sum(r["hit"] for r in rows)
    return {"group":name,"races":races,"bets":4*races,"hits":hits,
            "stake_yen":stake,"return_yen":paid,
            "profit_yen":paid-stake,
            "roi_percent":round(100*paid/stake,2) if stake else None,
            "hit_rate_percent":round(100*hits/races,2) if races else None,
            "max_payout_yen":max((r["paid"] for r in rows),default=0)}

def main():
    assert FROZEN_MODEL.is_file()
    conn=sqlite3.connect(f"file:{DB_PATH}?mode=ro",uri=True)
    conn.execute("PRAGMA query_only=ON")
    valid=prepare_valid(conn)
    gear=pd.read_sql_query("SELECT race_id,rider_id,gear_ratio AS gear_entry FROM entries",conn)
    meta=pd.read_sql_query("SELECT race_id,grade,date FROM races",conn)
    conn.close()
    log.info("RACE VALID %d",len(valid))
    model=LGBMRanker();model.load(str(FROZEN_MODEL))
    builder=FeatureBuilder();cols=builder.feature_names
    assert len(cols)==44 and model.feature_names==cols
    frame=builder.build_dataset(2022,2026)
    frame=frame[frame["race_id"].isin(valid)&frame["race_date"].between(LOW,END,inclusive="left")].copy()
    frame["rider_class_num"]=frame["class"].map(CLASS_MAP).fillna(6).astype(float)
    frame["class_rank_in_race"]=frame.groupby("race_id")["rider_class_num"].rank(method="min")
    frame["rider_class_x_jiku"]=frame["rider_class_num"]*frame["line_is_jiku"]
    frame=frame.merge(gear,on=["race_id","rider_id"],how="left",validate="many_to_one")
    frame["entry_gear_ratio"]=pd.to_numeric(frame["gear_entry"],errors="coerce").fillna(DEFAULT_GEAR_RATIO)
    frame.drop(columns=["gear_entry"],inplace=True)
    frame=frame.merge(meta,on="race_id",how="left",validate="many_to_one")
    frame["score"]=model.predict(frame[cols].apply(pd.to_numeric,errors="coerce").fillna(0)).values
    log.info("FROZEN_HOLDOUT %d rows %d races",len(frame),frame["race_id"].nunique())
    assert frame["race_id"].nunique()==3973

    payoff=payout_source()
    log.info("OFFICIAL_EXACTA_RACES %d",len(payoff))
    results=[]; audits=Counter()
    for rid,g in frame.groupby("race_id",sort=False):
        grade=str(g["grade"].iloc[0]);day=str(g["race_date"].iloc[0])
        site=rid[8:10];no=int(rid[10:12]);key=(day,site,no)
        if grade not in EXCLUDED and grade!="F1":
            audits["other_nonexcluded_grade"]+=1
        if grade!="F1":
            continue
        audits["base_F1"]+=1
        if len(g)<5 or g["bike_number"].isna().any():
            audits["malformed_base"]+=1;continue
        actual=g.sort_values("finish_position",kind="stable")
        a1,a2=int(actual["bike_number"].iloc[0]),int(actual["bike_number"].iloc[1])
        pay=payoff.get(key)
        if not pay:
            audits["no_official_payout_match"]+=1;continue
        winner_set={(x[0],x[1]) for x in pay}
        if len(pay)!=1 or winner_set!={(a1,a2)}:
            audits["conflicting_or_unmatched_payout"]+=1;continue
        odds_cars,odds_yen=next(iter(pay))[:2],next(iter(pay))[2]
        ordered=g.sort_values("score",ascending=False,kind="stable")
        bikes=[int(x) for x in ordered["bike_number"].tolist()]
        scores=[float(x) for x in ordered["score"].tolist()]
        gap=scores[0]-scores[1]
        label,_,yen=get_bet_category(gap,grade)
        purchased=(yen>0)
        tickets=[(bikes[0],b) for b in bikes[1:5]]
        assert len(tickets)==4 and len(set(tickets))==4
        hit=int((a1,a2) in tickets)
        if not purchased:
            audits["eligible_but_below_threshold"]+=1
        else:
            audits["bet_eligible"]+=1
        results.append({
            "race_id":rid,"race_date":day,"grade":grade,"score_gap":round(gap,5),
            "prediction_top5":"-".join(map(str,bikes[:5])),
            "prediction_all":"-".join(map(str,bikes)),
            "score_vector_json":json.dumps(scores,separators=(",",":")),
            "exacta_four":";".join(f"{x}-{y}" for x,y in tickets),
            "actual_exacta":f"{a1}-{a2}","payout_yen_100":odds_yen,
            "qualifies":int(purchased),"hit":hit if purchased else 0,
            "stake":4*yen if purchased else 0,
            "paid":int(odds_yen*yen/100) if purchased and hit else 0,
            "f1_top1_hit":int(bikes[0]==a1),
        })
    log.info("COVERAGE_AUDIT %s",dict(audits))
    assert len(results)>0,"No matched official 2車単 data"
    tab=pd.DataFrame(results).sort_values(["race_date","race_id"])
    tab.to_csv(OUT/"matched_2shatan_all_f1.csv",index=False)
    selections=[r for r in results if r["qualifies"]]
    overall=block_report(selections,"selected_1.5_and_F1")
    quarterly={}
    for k in ("2025Q3","2025Q4","2026Q1"):
        q=[r for r in selections if (r["race_date"]<"2025-10-01" if k=="2025Q3" else
          "2025-10-01"<=r["race_date"]<"2026-01-01" if k=="2025Q4" else
          "2026-01-01"<=r["race_date"])]
        quarterly[k]=block_report(q,k)
    monthly={}
    for month in sorted({r["race_date"][:7] for r in results}):
        monthly[month]=block_report([r for r in selections if r["race_date"][:7]==month],month)

    # Fixed rule compared with no confidence filter, a diagnostic only.
    all_f1=[]
    for r in results:
        pay=r["payout_yen_100"];actual=r["actual_exacta"]
        tickets=r["exacta_four"].split(";")
        all_f1.append({**r,"hit":int(actual in tickets),
            "stake":2000,"paid":int(pay*5) if actual in tickets else 0})
    full=block_report(all_f1,"all_F1_same_4_point_tickets")
    # Ex post coverage: outcomes/settlements not available for uncovered races.
    report={
      "model":"frozen 44-feature LambdaRank from 39911266752, no retraining",
      "test_interval":[LOW,END],"ranking":"model-only, no live line-bonus",
      "original_betting_rule":"F1 only in available payout set; score_gap>=1.5; 2車単 ◎->rank2..5, 4 x 500 yen",
      "eligible_universe_excluded_grade":{"F2":True,"G2":True,"G3":True},
      "base_holdout_races":int(frame["race_id"].nunique()),
      "matched_official_F1_payout_races":len(results),
      "audit":dict(audits),"overall":overall,"quarterly":quarterly,
      "monthly":monthly,"all_F1_4point_diagnostic":full,
      "limitations":[
       "Post-race official 2車単 winning payouts are used; these are NOT pre-race odds.",
       "The live torigami filter (all four odds <4.0) cannot be applied without four pre-race exacta prices.",
       "Historical payout information is used only to settle predefined tickets, never to select them.",
       "No independently confirmed decision-time odds; not a validated executable return.",
       "The offline fixed model rankings do not include the original live line bonus step.",
       "Only S-class F1 race payouts in existing market-data files are covered; other grades and missing race IDs are not assigned zero return.",
       "Selection threshold was fixed from original config.py before inspecting these results.",
      ],
    }
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    log.info("EXACTA RESULTS %s",json.dumps({k:report[k] for k in ("matched_official_F1_payout_races","audit","overall","quarterly","monthly","all_F1_4point_diagnostic")},ensure_ascii=False))
    log.info("EXACTA COMPLETED %s",OUT)

if __name__=="__main__":
    main()
