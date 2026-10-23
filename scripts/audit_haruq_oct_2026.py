#!/usr/bin/env python3
"""Predeclared October 2026 after-selection test for immutable HaruQ 03/21 model.

No model fitting. Same original 44 features, score-gap thresholds, 4 exacta
and 6 trifecta tickets. F2 included by explicit 2026-10-23 user instruction.
Period: Oct 1-22, 2026, deliberately outside original Jul25-Sep26 ROI set.
"""
import argparse, hashlib, itertools, json
from collections import defaultdict, Counter
from pathlib import Path
import sqlite3

import pandas as pd

from scripts.flagship_live_predict import MODEL_SHA256
from scripts.reproduce_haruqube_20260321 import verify_sources, settle
from scripts.reproduce_haruqube_fresh import dataset, split
from config import DB_PATH, get_bet_category, get_trifecta_category
from models.lgbm_ranker import LGBMRanker


START="2026-10-01"
END="2026-10-22"

def aggregate(rows):
    verified=[r for r in rows if r["payout_verified"]]
    spent=sum(r["invested_yen"] for r in verified)
    got=sum(r["returned_yen"] for r in verified)
    return dict(attempted=len(rows),verified=len(verified),
                payout_unverifiable=len(rows)-len(verified),
                hits=sum(int(r["hit"]) for r in verified),
                investment_yen=spent,return_yen=got,profit_yen=got-spent,
                roi_percent=round(100*got/spent,3) if spent else None)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--model",type=Path,required=True)
    a.add_argument("--output",type=Path,default=Path("haruq_oct_2026_forward_test.json"))
    args=a.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise RuntimeError("Frozen model changed")
    verify_sources()
    with sqlite3.connect(DB_PATH) as c:
        if c.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise RuntimeError("DB corrupt")
        last=c.execute("SELECT MAX(date) FROM races").fetchone()[0]
        p=pd.read_sql_query("SELECT race_id,sanrentan_combo,sanrentan_payout FROM race_payouts",c)
    if last<END:raise RuntimeError("Current database does not cover Oct22")
    df,features,valid=dataset()
    df=df.merge(p,on="race_id",how="left",validate="many_to_one")
    focused=df[(df["race_date"]>=START)&(df["race_date"]<=END)]
    focused,x,y,groups=split(focused,features)
    if len(groups)<50:raise RuntimeError(f"Not enough October races: {len(groups)}")
    model=LGBMRanker()
    model.load(str(args.model))
    if features!=model.feature_names:raise RuntimeError("Feature names differ")
    scores=model.predict(x)
    focused=focused.copy()
    focused["score"]=scores
    rows=[]
    races=Counter()
    for rid,group in focused.groupby("race_id",sort=False):
        pred=group.sort_values(["score","bike_number"],ascending=[False,True],kind="stable")
        if len(pred)<5:continue
        actual=group.sort_values("finish_position",kind="stable")
        win2=tuple(int(n) for n in actual.head(2)["bike_number"])
        win3=tuple(int(n) for n in actual.head(3)["bike_number"])
        grade=str(pred.iloc[0]["grade"])
        races[grade]+=1
        score_gap=float(pred.iloc[0]["score"]-pred.iloc[1]["score"])
        top=[int(b) for b in pred.head(5)["bike_number"]]
        if len(set(top))!=5:raise RuntimeError("Duplicate bike numbers")
        ex=[(top[0],b) for b in top[1:5]]
        tri=[(top[0],b,c) for b,c in itertools.permutations(top[1:4],2)]
        # The historical 03/21 category function is unchanged, but HaruQ V2
        # explicitly bypasses the F2 grade gate only.
        eligibility_grade="" if grade=="F2" else grade
        ex_tier,_,ex_amount=get_bet_category(score_gap,eligibility_grade)
        tri_tier,_,tri_amount=get_trifecta_category(score_gap,eligibility_grade)
        if ex_amount:
            rows.append(settle("OCT_exacta",pred,score_gap,grade,ex_tier,
                               ex_amount,ex,win2,"nisyatan"))
        if tri_amount:
            rows.append(settle("OCT_trifecta",pred,score_gap,grade,tri_tier,
                               tri_amount,tri,win3,"sanrentan"))
    by=defaultdict(list)
    for r in rows:by[(r["version"],r["tier"],r["grade"])].append(r)
    def filt(version=None,tier=None,grade=None):
        return [r for r in rows if (version is None or r["version"]==version)
                and (tier is None or r["tier"]==tier)
                and (grade is None or r["grade"]==grade)]
    high=filt("OCT_trifecta","S3-HIGH")
    med=filt("OCT_trifecta","S3-MED")
    medp=filt("OCT_trifecta","S3-MED+")
    ex=filt("OCT_exacta")
    selected=high+ex
    high_by_month=sorted(high,key=lambda r:r["returned_yen"],reverse=True)
    stress={str(i):aggregate(high_by_month[i:]) for i in (0,1,3,5,10)}
    report={
      "status":"OCTOBER_AFTER_SELECTION_FROZEN_MODEL_TEST",
      "model_sha256":MODEL_SHA256,
      "period":{"start":START,"end":END,"outside_original_holdout":True},
      "source":{"latest_complete_day":last,"grade_race_counts":dict(races),
                "race_count":len(groups),"quality_valid_races_all_years":valid},
      "purchases":{
        "s3_high_all_grades":aggregate(high),
        "s3_high_f2":aggregate(filt("OCT_trifecta","S3-HIGH","F2")),
        "s3_high_non_f2":aggregate([r for r in high if r["grade"]!="F2"]),
        "two_car_exacta_all_grades":aggregate(ex),
        "combined_definite":aggregate(selected)},
      "diagnostic_not_purchased":{
        "s3_medplus":aggregate(medp),
        "s3_med":aggregate(med)},
      "breakdown":[{"version":v,"tier":t,"grade":g,**aggregate(r)}
                   for (v,t,g),r in sorted(by.items())],
      "high_payout_concentration":stress,
      "limits":[
        "This period follows the historical ROI holdout but is retrospective, not timestamped live staking.",
        "Original model trained only until 2024 and fixed, but historical feature-vs-live parity remains unaudited.",
        "A single 22-day holdout cannot establish a stable true edge.",
        "Ticket payouts simulated at recorded winning payout; no true-time odds or market-size constraints verified."
      ],
    }
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HARUQ_OCT_FORWARD_TEST_COMPLETE "+json.dumps({"dates":report["period"],
       "races":len(groups),"high":report["purchases"]["s3_high_all_grades"],
       "high_F2":report["purchases"]["s3_high_f2"],
       "combined":report["purchases"]["combined_definite"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
