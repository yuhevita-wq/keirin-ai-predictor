#!/usr/bin/env python3
"""Separate buying-policy learner for EACH OFFICIAL RACE TYPE, not one pooled tree.

The fixed ranker and published KDreams race_type/payoff labels are immutable.
Race-type keys are literal (division, race grade, official named race_type).
2025H1 dev -> 2025Q3 selects per-type candidate -> 2025Q4/2026Q1
unseen tests. All records retained, every unsupported class abstains rather
than being silently excluded from dataset. No historical losing-ticket odds.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.tree import DecisionTreeRegressor

OUT=Path("race_type_roi_outputs")
PAIRS=OUT/"all_exacta_ticket_outcomes.csv"
RACES=OUT/"matched_race_types.csv"
MODELS=OUT/"specialist_models"
MODELS.mkdir(parents=True,exist_ok=True)
DIMS=["division","grade","race_type"]
FEATURES=[
   "head_rank","second_rank","rider_count","gap12","gap23","gap34","gap45",
   "gap_1st_head","gap_1st_second","head_second_gap","score_spread",
   "head_next_gap","second_next_gap","normalized_pair_strength","score_std"
]

def metric(rows):
    n=len(rows)
    if not n:return {"bets":0,"races":0,"hits":0,"roi_pct":None,"stake_yen":0,"paid_yen":0,"profit_yen":0}
    paid=int(rows["payout_yen"].sum())
    return {"bets":int(n),"races":int(rows["race_id"].nunique()),"hits":int(rows["won"].sum()),
            "roi_pct":round(paid/n,2),"stake_yen":100*n,
            "paid_yen":paid,"profit_yen":paid-100*n}

def sample_ci(bets,n=1000):
    if bets.empty:return None
    z=bets.groupby("race_id").agg(bets=("payout_yen","size"),pay=("payout_yen","sum")).to_numpy()
    rng=np.random.default_rng(20261023)
    samples=[]
    for _ in range(n):
        idx=rng.integers(0,len(z),len(z))
        v=z[idx]
        samples.append(float(v[:,1].sum()/v[:,0].sum()))
    return [round(x,2) for x in np.quantile(samples,[.025,.975])]

def group_name(values):
    return "/".join(str(v) for v in values)

def group_selection(frame):
    m_dev=frame["phase"]=="dev"
    m_val=frame["phase"]=="val"
    m_q4=frame["phase"]=="test_q4"
    m_q1=frame["phase"]=="test_q1"
    records=[];outcomes=[];files=[]
    for group,keyed in frame.groupby(DIMS,sort=True):
        name=group_name(group)
        tr=keyed[keyed["phase"]=="dev"]
        va=keyed[keyed["phase"]=="val"]
        q4=keyed[keyed["phase"]=="test_q4"]
        q1=keyed[keyed["phase"]=="test_q1"]
        report={"label":name,
                "division":group[0],"grade":group[1],"race_type":group[2],
                "counts":{"development_races":int(tr["race_id"].nunique()),
                          "validation_races":int(va["race_id"].nunique()),
                          "test_2025Q4_races":int(q4["race_id"].nunique()),
                          "test_2026Q1_races":int(q1["race_id"].nunique())}}
        # Each group has explicitly different historical 2車単 results and
        # a different trained predictor, no pooling across unrelated race types.
        best=None;test_briefs=[]
        if report["counts"]["development_races"]>=35 and report["counts"]["validation_races"]>=20:
            x=tr[FEATURES].fillna(0)
            for depth,leaf,cap in [(2,100,100),(3,80,100)]:
                model=DecisionTreeRegressor(
                    max_depth=depth,min_samples_leaf=leaf,random_state=20261023)
                model.fit(x,tr["reward"].clip(upper=cap))
                for threshold in (.85,1.0,1.2):
                    tr_b=tr[model.predict(tr[FEATURES].fillna(0))>threshold]
                    va_b=va[model.predict(va[FEATURES].fillna(0))>threshold]
                    tm=metric(tr_b);vm=metric(va_b)
                    lv=(va_b["payout_yen"].sum()-va_b["payout_yen"].max())/len(va_b) if len(va_b) else 0.
                    # Predeclared minimum sample size, positive Q3 return and
                    # some resilience to single lucky jackpot.
                    selected=bool(
                        tm["bets"]>=40 and tm["roi_pct"] is not None and tm["roi_pct"]>=85
                        and vm["bets"]>=40 and vm["races"]>=15 and vm["hits"]>=2
                        and vm["roi_pct"]>105 and lv>=75
                    )
                    test_briefs.append({"depth":depth,"leaf":leaf,
                         "threshold":threshold,"dev":tm,"val":vm,
                         "val_roi_without_biggest_win":round(float(lv),2),
                         "passes_selection":selected})
                    if selected:
                        item={"model":model,"depth":depth,"leaf":leaf,
                            "threshold":threshold,"dev":tm,"val":vm,
                            "robust_val_roi":lv}
                        if best is None or (item["robust_val_roi"],item["val"]["roi_pct"]) > (
                           best["robust_val_roi"],best["val"]["roi_pct"]):
                            best=item
            report["models_examined"]=len(test_briefs)
            report["selection_passes"]=sum(t["passes_selection"] for t in test_briefs)
        else:
            report["insufficient_dev_or_val"]=True
        if best:
            model=best["model"]
            threshold=best["threshold"]
            report["specialist_selected"]=True
            report["model_def"]={"depth":best["depth"],"leaf":best["leaf"],
                                 "threshold":threshold}
            report["development"]=best["dev"]
            report["validation"]=best["val"]
            file=f"specialist_{len(files):03d}.joblib"
            dump({"model":model,"features":FEATURES,"group":group,
                  "threshold":threshold,"dev_period":"2025H1"},
                 MODELS/file)
            files.append({"name":file,"group":name})
            for phase,rows in (("2025Q4",q4),("2026Q1",q1)):
                bets=rows[model.predict(rows[FEATURES].fillna(0))>threshold] if len(rows) else rows
                report["test_"+phase]=metric(bets)
                # Record exact tickets for audit, not a generic "successful" tag
                if len(bets):
                    stub=bets[["race_id","date","head_rank","second_rank","won","payout_yen"]].copy()
                    stub["specialist_group"]=name
                    stub["test_phase"]=phase
                    outcomes.append(stub)
        else:
            report["specialist_selected"]=False
            report["no_supported_profitable_policy"]=True
        # All of the (also unsupported) classifications are tracked.
        for label,x in (("2025Q4",q4),("2026Q1",q1)):
            fixed=x[(x["head_rank"]==1)&x["second_rank"].between(2,5)]
            report["base_four_"+label]=metric(fixed)
            fixed=x[(x["head_rank"]==1)&(x["second_rank"]==2)]
            report["base_rank12_"+label]=metric(fixed)
        records.append(report)
    if outcomes:
        pick=pd.concat(outcomes,ignore_index=True)
    else:
        pick=frame.iloc[:0][["race_id","date","head_rank","second_rank","won","payout_yen"]].copy()
    pick.to_csv(OUT/"specialist_forward_tickets.csv",index=False)
    overall={}
    for phase in ("2025Q4","2026Q1"):
        overall[phase]=metric(pick[pick["test_phase"]==phase]) if len(pick) else metric(pick)
        overall[phase]["bootstrap95"]=sample_ci(pick[pick["test_phase"]==phase]) if len(pick) else None
    overall["combined"]=metric(pick)
    overall["combined"]["bootstrap95"]=sample_ci(pick)
    overall["types_with_specialist"]=len(files)
    overall["types_without_specialist"]=len(records)-len(files)
    overall["deployable"]=bool(
       len(files)>0 and all((overall[p]["roi_pct"] or 0)>100 for p in ("2025Q4","2026Q1")) and
       overall["combined"]["bootstrap95"] and overall["combined"]["bootstrap95"][0]>100)
    overall["live_deployed"]=False
    return records,files,overall

def main():
    all_pairs=pd.read_csv(PAIRS,dtype={"race_id":str})
    all_races=pd.read_csv(RACES,dtype={"race_id":str})
    # Re-normalize precise label directly from official name, never inference
    # from outcome. Retain all original 7018 races; no F1 or other filtering.
    assert len(all_races)==7018 and all_pairs["race_id"].nunique()==7018
    records,files,portfolio=group_selection(all_pairs)
    report={"all_races":len(all_races),"payout_matched_races":len(all_races),
            "unique_literal_types":len(records),
            "type_definition":"division/grade/official literal race_type",
            "source":"immutable corrected frozen base AI rank scores and KDreams 2車単 settlement archive",
            "training":"2025H1","validation":"2025Q3",
            "final_tests":["2025Q4","2026Q1"],
            "method":"Independently fitted decision tree per official race type, never pooled across distinct categories",
            "portfolio":portfolio,"models":files,
            "disaggregated_type_results":records,
            "limitations":[
                "Earnings reflect final published 100-yen winning payouts, not before-close odds.",
                "No unpriced-ticket market EV or torigami filter tested.",
                "Some types have small or absent evaluation samples.",
                "Any model trained on H1 is not production-approved without robust forward-quarter results.",
                "Race types without evidence are retained in reporting and produce no purchases.",
            ]}
    (OUT/"race_type_specialists_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2))
    print("SPECIALISTS_GLOBAL",json.dumps({
        "race_types":len(records),"portfolio":portfolio},ensure_ascii=False),flush=True)
    print("SPECIALISTS_SELECTED_TYPE_SUMMARIES",
          json.dumps([
              {"type":x["label"],"counts":x["counts"],
               "dev":x.get("development"),"val":x.get("validation"),
               "test_q4":x.get("test_2025Q4"),"test_q1":x.get("test_2026Q1")}
              for x in records if x["specialist_selected"]],ensure_ascii=False),flush=True)
    print("SPECIALISTS_ALL_TYPE_COUNTS",json.dumps(
       [{"type":x["label"],"counts":x["counts"],
          "selected":x["specialist_selected"],
          "q4_base_4pt_roi":x["base_four_2025Q4"]["roi_pct"],
          "q1_base_4pt_roi":x["base_four_2026Q1"]["roi_pct"]}
         for x in records],ensure_ascii=False),flush=True)
    print("SPECIALISTS_COMPLETE",flush=True)

if __name__=="__main__":main()
