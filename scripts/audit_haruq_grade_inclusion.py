#!/usr/bin/env python3
"""Independent frozen 3/21 purchase-ledger audit and F2-inclusive extension.

Uses the saved pretrained model, and historical 44-feature code; NO training.
Checks original release ledger row-by-row against freshly reconstructed results,
checks reported ROI arithmetic, and computes F2-only and all-grade strategy ROI.
Unverifiable payouts are counted but NOT misreported as losses or wins.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from scripts.flagship_live_predict import MODEL_SHA256
from scripts.reproduce_haruqube_20260321 import (
    FIELDS, parse_combo, settle, verify_payout, simulate, verify_sources,
)
import scripts.reproduce_haruqube_20260321 as repro
from scripts.reproduce_haruqube_fresh import (
    dataset, split, TRAIN_END, VALID_END, TEST_END,
)
from models.lgbm_ranker import LGBMRanker
from config import DB_PATH, get_bet_category, get_trifecta_category


def summarise(rows):
    good=[r for r in rows if bool(r["payout_verified"])]
    stake=sum(int(r["invested_yen"]) for r in good)
    returned=sum(int(r["returned_yen"]) for r in good)
    return {
        "attempted":len(rows),
        "verified":len(good),
        "unverified":len(rows)-len(good),
        "hits":sum(int(r["hit"]) for r in good),
        "investment_yen":stake,
        "return_yen":returned,
        "profit_yen":returned-stake,
        "roi_percent":round(100.0*returned/stake,3) if stake else None,
    }


def f2_and_grade_segments(test,model,features):
    scores=model.predict(test[features].apply(pd.to_numeric,errors="coerce").fillna(0.0)).values
    frame=test.copy()
    frame["audit_score"]=scores
    all_rows=[]
    target_count=Counter()
    for rid,group in frame.groupby("race_id",sort=False):
        rank=group.sort_values(["audit_score","bike_number"],ascending=[False,True],kind="stable")
        if len(rank)<5:
            continue
        grade=str(rank.iloc[0]["grade"])
        target_count[grade]+=1
        # Only the original March 21 F2 exclusion is bypassed; confidence and
        # all other 03/21 parameters remain exactly unchanged.
        if grade!="F2":
            continue
        real=group.sort_values("finish_position",kind="stable")
        actual2=tuple(int(x) for x in real.head(2)["bike_number"])
        actual3=tuple(int(x) for x in real.head(3)["bike_number"])
        gap=float(rank.iloc[0]["audit_score"]-rank.iloc[1]["audit_score"])
        bikes=[int(x) for x in rank.head(5)["bike_number"]]
        if len(set(bikes))!=5:
            raise RuntimeError(f"F2 duplicate bike numbers {rid}")
        exacta=[(bikes[0],p) for p in bikes[1:5]]
        tri=[(bikes[0],b,c) for b,c in itertools.permutations(bikes[1:4],2)]
        ex_tier,_,ex_each=get_bet_category(gap,"")
        tri_tier,_,tri_each=get_trifecta_category(gap,"")
        if ex_each:
            all_rows.append(settle("F2_ADDED_exacta",rank,gap,grade,ex_tier,ex_each,exacta,actual2,"nisyatan"))
        if tri_each:
            all_rows.append(settle("F2_ADDED_trifecta",rank,gap,grade,tri_tier,tri_each,tri,actual3,"sanrentan"))
    return all_rows,target_count


def keyed(rows):
    return {(r["version"],r["race_id"]):r for r in rows}


def ledger_check(rebuilt_rows,archived_rows):
    a=keyed(archived_rows)
    b=keyed(rebuilt_rows)
    if len(a)!=len(archived_rows) or len(b)!=len(rebuilt_rows):
        raise RuntimeError("Duplicate row keys in original/rebuilt ledger")
    if a.keys()!=b.keys():
        raise RuntimeError("Rebuilt vs published ledger race sets differ: "
                           f"missing {len(a.keys()-b.keys())}, extra {len(b.keys()-a.keys())}")
    fields=("date","grade","tier","ticket_count","tickets","winning_combo",
            "stake_each_yen","invested_yen","returned_yen","hit","payout_verified")
    discrepancies=[]
    for key in a:
        for f in fields:
            if str(a[key][f])!=str(b[key][f]):
                discrepancies.append({"key":key,"field":f,
                                      "release":str(a[key][f]),"rebuilt":str(b[key][f])})
                if len(discrepancies)>=10:
                    break
        if abs(float(a[key]["confidence"])-float(b[key]["confidence"]))>1e-6:
            discrepancies.append({"key":key,"field":"confidence"})
        if discrepancies and len(discrepancies)>=10:
            break
    if discrepancies:
        raise RuntimeError("Original ledger reconstruction disagrees: "
                           +json.dumps(discrepancies,ensure_ascii=False))
    return {"rows":len(a),"race_sets_equal":True,"all_ticket_outcomes_equal":True}


def tier_breakdown(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["version"],r["tier"],r["grade"],r["date"][:4])].append(r)
    return [{"version":v,"tier":t,"grade":g,"year":y,**summarise(group)}
            for (v,t,g,y),group in sorted(groups.items())]


def stress(r):
    settled=sorted((x for x in r if x["payout_verified"]),
                   key=lambda x:int(x["returned_yen"]),reverse=True)
    output={}
    for n in (0,1,3,5,10,20):
        output[f"without_{n}_largest_returns"]=summarise(settled[n:])
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--archived-report",required=True,type=Path)
    p.add_argument("--archived-ledger",required=True,type=Path)
    p.add_argument("--out",default="haruq_financial_audit.json",type=Path)
    args=p.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise RuntimeError("Saved trained model digest mismatch")
    verify_sources()
    with sqlite3.connect(DB_PATH) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0]!="ok":
            raise RuntimeError("Historical DB integrity_check failed")
        source_count=conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        race_days=conn.execute("SELECT MIN(date),MAX(date) FROM races").fetchone()
        pay=pd.read_sql_query("SELECT race_id,sanrentan_combo,sanrentan_payout FROM race_payouts",conn)
    if pay["race_id"].duplicated().any():
        raise RuntimeError("Duplicate 3連単 payout race")
    frame,features,valid=dataset()
    frame=frame.merge(pay,on="race_id",how="left",validate="many_to_one")
    _train=frame[frame["race_date"]<TRAIN_END]
    _val=frame[(frame["race_date"]>=TRAIN_END)&(frame["race_date"]<VALID_END)]
    test=frame[(frame["race_date"]>=VALID_END)&(frame["race_date"]<TEST_END)]
    test,xt,yt,groups=split(test,features)
    model=LGBMRanker()
    model.load(str(args.model))
    if features!=model.feature_names:
        raise RuntimeError("Historical feature names do not match saved model")
    repro.OUT=Path("audit_rebuilt_tmp")
    repro.OUT.mkdir(exist_ok=True)
    orig=repro.simulate(test,model,features)
    rebuilt=list(csv.DictReader((repro.OUT/"holdout_all_selected_tickets.csv").open(encoding="utf-8")))
    archived=list(csv.DictReader(args.archived_ledger.open(encoding="utf-8")))
    assert len(archived)>10000
    compare=ledger_check(rebuilt,archived)
    released=json.loads(args.archived_report.read_text(encoding="utf-8"))
    if released.get("model_sha256")!=MODEL_SHA256:
        raise RuntimeError("Original report's trained model digest changed")
    for k,summary in orig.items():
        archived_summary=released["holdout_betting"][k]
        for a,b in (("payout_verified_races","payout_verified_races"),
                    ("invested_yen","invested_yen"),("returned_yen","returned_yen"),
                    ("hits","hits")):
            if summary[a]!=archived_summary[b]:
                raise RuntimeError(f"Archived release report mismatch {k} {a}")
        if "tiers" in summary:
            for tier,sm in summary["tiers"].items():
                other=archived_summary["tiers"][tier]
                for key in ("payout_verified_races","invested_yen","returned_yen","hits"):
                    if sm[key]!=other[key]:
                        raise RuntimeError(f"Tier {tier} {key} mismatch")
    f2_rows,all_grades=f2_and_grade_segments(test,model,features)
    full=rebuilt+f2_rows
    def where(version=None,tier=None,grade=None):
        return [r for r in full if (version is None or r["version"]==version)
                and (tier is None or r["tier"]==tier)
                and (grade is None or r["grade"]==grade)]
    old_tri=where("2026-03-21_trifecta")
    f2_tri=where("F2_ADDED_trifecta")
    old_ex=where("2026-03-21_exacta")
    f2_ex=where("F2_ADDED_exacta")
    all_high=[r for r in old_tri+f2_tri if r["tier"]=="S3-HIGH"]
    all_medplus=[r for r in old_tri+f2_tri if r["tier"]=="S3-MED+"]
    all_med=[r for r in old_tri+f2_tri if r["tier"]=="S3-MED"]
    # This newly authorised policy is not the original 03/21 3-band strategy.
    # 3連単 buys HIGH only, MED+ rejects, MED is user-review-only.
    to_buy=old_ex+f2_ex+all_high
    report={
        "status":"RECONSTRUCTED_FROZEN_MODEL_AUDIT",
        "model_sha256":MODEL_SHA256,
        "verification":{"archived_release_ledger":compare,
                        "archived_report_arithmetic":True,
                        "db_integrity_check":"ok",
                        "historic_feature_names_match":True},
        "source":{"db_races":source_count,"db_start":race_days[0],"db_end":race_days[1],
                  "valid_races":valid,"holdout_races":len(groups),
                  "holdout_date_from":"2025-07-01","holdout_date_to":"2026-09-30",
                  "model_source":"preserved frozen release"},
        "holdout_grade_counts":dict(all_grades),
        "historical_0321":{
            "trifecta":summarise(old_tri),"exacta":summarise(old_ex),
            "s3_high":summarise([x for x in old_tri if x["tier"]=="S3-HIGH"]),
            "s3_medplus":summarise([x for x in old_tri if x["tier"]=="S3-MED+"]),
            "s3_med":summarise([x for x in old_tri if x["tier"]=="S3-MED"]),
            "f2_selected_rows":sum(r["grade"]=="F2" for r in rebuilt),
        },
        "f2_added":{
            "trifecta":summarise(f2_tri),"exacta":summarise(f2_ex),
            "s3_high":summarise([x for x in f2_tri if x["tier"]=="S3-HIGH"]),
            "s3_medplus":summarise([x for x in f2_tri if x["tier"]=="S3-MED+"]),
            "s3_med":summarise([x for x in f2_tri if x["tier"]=="S3-MED"]),
        },
        "new_all_grade_policy":{
            "s3_high_buy":summarise(all_high),
            "s3_medplus_reject_diagnostic_only":summarise(all_medplus),
            "s3_med_manual_review_diagnostic_only":summarise(all_med),
            "exacta_original_tiers_with_f2_enabled":summarise(old_ex+f2_ex),
            "combined_definite_stakes":summarise(to_buy),
        },
        "tier_grade_year_breakdown":tier_breakdown(full),
        "s3_high_largest_payout_stress":stress(all_high),
        "limitations":[
            "This is historical holdout reconstruction, NOT live betting earnings.",
            "2025-07 through 2026-09 holdout was used to choose new S3 buy/reject policy; selection/overfit risk remains.",
            "Winning payout-only verification excludes races with unverified payout from both numerator and denominator; missingness could bias ROI.",
            "Historic live pre-race feature parity vs batch build_dataset is not independently established.",
            "No historical pre-race transaction logs; each selected ticket is simulated at final payout.",
            "F2 is newly included in posthoc analysis; this does not retroactively alter original 03/21 ROI.",
        ],
    }
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HARUQ_AUDIT_COMPLETE "+json.dumps({
        "model_match":True,"archived_rows_verified":compare["rows"],
        "original_high":report["historical_0321"]["s3_high"],
        "f2_high":report["f2_added"]["s3_high"],
        "all_grade_high":report["new_all_grade_policy"]["s3_high_buy"],
        "new_combined":report["new_all_grade_policy"]["combined_definite_stakes"],
        "total_holdout":len(groups)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
