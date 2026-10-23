#!/usr/bin/env python3
"""Learn which FROZEN base-AI ranking score shapes support 2車単 profit.

Uses only full pre-result score vectors from the preceding immutable model
evaluation and post-result published 2車単 settlement outcomes. No new
ranking model. No outcome/price enters features. Chronology:
  develop 2025Q3, model selection only 2025Q4, one final test 2026Q1.
All ordered rank pairs are available; no fixed favorite, formation or k limit.
No losing ticket price is ever imputed. With realized winning payout only,
this is a retrospective payout-conditioned policy-learning experiment;
it cannot determine live price-based EV.
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

ROOT=Path("exacta_outputs")
DATA=ROOT/"matched_2shatan_all_f1.csv"
FEATURES=[
  "riders","head_rank","second_rank","rank_sep",
  "gap12","gap23","gap34","gap45",
  "gap13","gap15","full_range","score_std",
  "head_gap_from_top","second_gap_from_top",
  "head_second_gap","pair_strength_sum",
  "head_gap_prev","second_gap_prev",
  "head_gap_next","second_gap_next"
]
TRAIN=("2025-07-01","2025-10-01")
VAL=("2025-10-01","2026-01-01")
TEST=("2026-01-01","2026-04-01")

def phase(d):
    if TRAIN[0]<=d<TRAIN[1]:return "develop_2025Q3"
    if VAL[0]<=d<VAL[1]:return "selection_2025Q4"
    if TEST[0]<=d<TEST[1]:return "test_2026Q1"
    return None

def is_valid(scores):
    return all(math.isfinite(s) for s in scores) and all(a>=b-1e-8 for a,b in zip(scores,scores[1:]))

def make_data():
    data=[]
    n_races=Counter();bad=Counter()
    with DATA.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        try:
            d=r["race_date"];split=phase(d)
            if split is None:continue
            nums=[int(q) for q in r["prediction_all"].split("-")]
            scores=[float(s) for s in json.loads(r["score_vector_json"])]
            if not (len(nums)==len(scores) and len(set(nums))==len(nums) and len(nums)>=5 and is_valid(scores)):
                bad["invalid_ranking"]+=1;continue
            res=tuple(map(int,r["actual_exacta"].split("-")))
            payout=int(r["payout_yen_100"])
            if not 100<=payout<=10000000:
                bad["invalid_payout"]+=1;continue
            s=np.array(scores,dtype=float)
            gap=lambda i,j: float(s[i]-s[j]) if j<len(s) else 0.0
            rr=r["race_id"]
            n_races[split]+=1
            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i==j:continue
                    # Information invariant to additive score shifts:
                    # score differences and score spread only.
                    v={
                      "riders":len(nums),
                      "head_rank":i+1,"second_rank":j+1,
                      "rank_sep":abs(i-j),
                      "gap12":gap(0,1),"gap23":gap(1,2),
                      "gap34":gap(2,3),"gap45":gap(3,4),
                      "gap13":gap(0,2),"gap15":gap(0,4),
                      "full_range":gap(0,len(s)-1),
                      "score_std":float(s.std()),
                      "head_gap_from_top":gap(0,i),
                      "second_gap_from_top":gap(0,j),
                      "head_second_gap":float(s[i]-s[j]),
                      "pair_strength_sum":float(2*s[0]-s[i]-s[j]),
                      "head_gap_prev":float(s[i-1]-s[i]) if i>0 else 0.,
                      "second_gap_prev":float(s[j-1]-s[j]) if j>0 else 0.,
                      "head_gap_next":float(s[i]-s[i+1]) if i+1<len(s) else 0.,
                      "second_gap_next":float(s[j]-s[j+1]) if j+1<len(s) else 0.,
                    }
                    hit=(nums[i],nums[j])==res
                    data.append({**v,"race_id":rr,"date":d,"phase":split,
                         "head_bike":nums[i],"second_bike":nums[j],
                         "won":int(hit),
                         "pay_yen":payout if hit else 0,
                         "reward_multiple":payout/100 if hit else 0.})
        except Exception as exc:
            bad[type(exc).__name__]+=1
    frame=pd.DataFrame(data)
    assert not frame.empty
    assert len(frame["race_id"].unique())==sum(n_races.values())
    frame.to_csv(ROOT/"all_ordered_rank_pair_rewards.csv",index=False)
    return frame,dict(n_races),dict(bad)

def metrics(d,label):
    if d.empty:
        return {"label":label,"bets":0,"races":0,"stake_yen":0,
                "returned_yen":0,"roi_percent":None}
    n=len(d)
    pay=float(d["pay_yen"].sum())
    hits=int(d["won"].sum())
    return {"label":label,"bets":n,"races":int(d["race_id"].nunique()),
            "hits":hits,"stake_yen":n*100,"returned_yen":int(pay),
            "profit_yen":int(pay-n*100),"roi_percent":round(pay/n,3),
            "hit_rate_pct":round(100*hits/n,3),
            "largest_single_payout":int(d["pay_yen"].max())}

def bootstrap_ci_rows(bets,seed=20261023,n_rep=1500):
    if bets.empty:return None
    grouped=bets.groupby("race_id").agg(paid=("pay_yen","sum"),bets=("pay_yen","size")).to_numpy()
    if len(grouped)<20:return None
    rng=np.random.default_rng(seed)
    outcome=[]
    for _ in range(n_rep):
        idx=rng.integers(0,len(grouped),size=len(grouped))
        pairs=grouped[idx]
        outcome.append(100*float(pairs[:,0].sum())/(100*float(pairs[:,1].sum())))
    return [round(x,2) for x in np.quantile(outcome,[.025,.975])]

def policy_tree_data(train,validation):
    # Train on actual-yen reward for all possible rank pairs, nonzero only
    # for observed winning 2車単. Prudent target clipping changes ONLY the
    # training loss; validation and test payouts remain complete, unmodified.
    policies={}
    for depth,leaf,cap in [(2,400,100),(3,350,100),(3,250,250)]:
        model=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=20261023)
        target=train["reward_multiple"].clip(upper=cap)
        model.fit(train[FEATURES],target)
        importance=dict(sorted(zip(FEATURES,model.feature_importances_),key=lambda x:-x[1]))
        policies[f"tree_d{depth}_l{leaf}_clip{cap}"]={
          "model":model,"importance":importance}
    return policies

def prepare_policy_candidates(frame,models):
    # Candidate portfolio gates are chosen on validation, never on 2026Q1.
    df=frame.copy()
    for name,item in models.items():
        df["prediction_"+name]=item["model"].predict(df[FEATURES])
    return df

def apply_strategy(d,name,threshold=None):
    if name=="original_gap_1p5":
        return d[(d["gap12"]>=1.5)&(d["head_rank"]==1)&d["second_rank"].between(2,5)]
    if name=="all_f1_four":
        return d[(d["head_rank"]==1)&d["second_rank"].between(2,5)]
    if name=="straight_rank12":
        return d[(d["head_rank"]==1)&(d["second_rank"]==2)]
    if name=="reversed_rank21":
        return d[(d["head_rank"]==2)&(d["second_rank"]==1)]
    if name.startswith("tree_"):
        return d[d["prediction_"+name]>threshold]
    raise ValueError(name)

def explain_group(frame):
    # Descriptive, train-anchored gap band ROI, NOT selected as a trading rule.
    q=frame[frame["phase"]=="develop_2025Q3"].drop_duplicates("race_id")["gap12"]
    cuts=sorted(set(float(z) for z in q.quantile([0,.25,.5,.75,1]).values))
    if len(cuts)<5:return {"available":False}
    # Remove floating equality ambiguity; highest bound infinity.
    bands={}
    for phase in ("develop_2025Q3","selection_2025Q4","test_2026Q1"):
        group=frame[frame["phase"]==phase]
        for k,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:]),start=1):
            # Q3 top quartile might not extend to Q4; last bucket is unbounded.
            high=float("inf") if k==4 else hi
            subset=group[(group["gap12"]>=lo if k>1 else group["gap12"]>=-float("inf"))&
                         (group["gap12"]<high)&(group["head_rank"]==1)&
                         group["second_rank"].between(2,5)]
            bands[f"{phase}_Q{k}"]=metrics(subset,f"{phase}_Q{k}")
    return {"train_quartile_edges":cuts,"bands":bands}

def main():
    frame, counts, bad=make_data()
    tr=frame[frame["phase"]=="develop_2025Q3"].copy()
    val=frame[frame["phase"]=="selection_2025Q4"].copy()
    test=frame[frame["phase"]=="test_2026Q1"].copy()
    assert len(tr)>500 and len(val)>500 and len(test)>100
    models=policy_tree_data(tr,val)
    for model_name, model_record in models.items():
        dump({'model':model_record['model'],'features':FEATURES,
              'training_period':TRAIN,'candidate_name':model_name},
             ROOT/(model_name+'.joblib'))
    scored=prepare_policy_candidates(frame,models)
    tr=scored[scored["phase"]=="develop_2025Q3"]
    val=scored[scored["phase"]=="selection_2025Q4"]
    test=scored[scored["phase"]=="test_2026Q1"]
    # Predeclared candidate selection: use the validation ROI, with a minimum
    # breadth of evidence, AND untrimmed profit positive in development.
    # Never select models or thresholds using Q1 returns.
    baseline=["original_gap_1p5","all_f1_four","straight_rank12","reversed_rank21"]
    candidates=[(n,float(t)) for n in models for t in (.9,1.,1.15,1.35)]
    scoreboard=[]
    for n,t in [(n,None) for n in baseline]+candidates:
        train_b=apply_strategy(tr,n,t)
        val_b=apply_strategy(val,n,t)
        metrics_tr=metrics(train_b,n)
        metrics_val=metrics(val_b,n)
        if len(val_b):
            biggest=val_b["pay_yen"].max()
            # remove top 1 profitable winning ticket for sensitivity,
            # compute unit return on other wagers, keep stake unchanged
            sensitivity=max(0.,float(val_b["pay_yen"].sum()-biggest))/len(val_b)
        else:
            sensitivity=0.
        scoreboard.append({
           "name":n,"threshold":t,"train":metrics_tr,
           "validation":metrics_val,
           "validation_without_largest_win_roi_percent":round(sensitivity,2),
           "eligible_for_validation_selection":bool(
                n.startswith("tree_") and
                metrics_tr["bets"]>=80 and metrics_val["bets"]>=80 and
                metrics_val["races"]>=30 and
                metrics_val["hits"]>=5 and
                metrics_val["roi_percent"]>100 and
                metrics_tr["roi_percent"]>100),
        })
    viable=[s for s in scoreboard if s["eligible_for_validation_selection"]]
    # Choose exactly ONE model/threshold before seeing Q1.
    chosen=sorted(viable,key=lambda s:(
        -s["validation_without_largest_win_roi_percent"],
        -s["validation"]["roi_percent"],s["name"],s["threshold"]))[0] if viable else None
    selected={}
    if chosen is not None:
        n=chosen["name"];t=chosen["threshold"]
        selected_b=apply_strategy(test,n,t)
        selected={"policy":n,"threshold":t,
                  "train":chosen["train"],"validation":chosen["validation"],
                  "test":metrics(selected_b,"final_unseen_2026Q1"),
                  "test_roi_bootstrap_ci95":bootstrap_ci_rows(selected_b)}
        interval=selected["test_roi_bootstrap_ci95"]
        selected["deployment_approved"]=bool(
            selected["test"]["roi_percent"] is not None and
            selected["test"]["roi_percent"]>100 and
            interval and interval[0]>100)
        selected["deployment_note"]="Not deployable unless reserved-period ROI and its bootstrap lower bound both exceed 100%."
        scored["selected_ticket"]=0
        idx=apply_strategy(scored,n,t).index
        scored.loc[idx,"selected_ticket"]=1
    else:
        selected={"policy":None,"status":"No candidate met minimum profitable training and validation evidence"}
        scored["selected_ticket"]=0

    comparators={}
    for n in baseline:
        comparators[n]={}
        for split,d in [("develop_2025Q3",tr),("selection_2025Q4",val),("test_2026Q1",test)]:
            b=apply_strategy(d,n)
            comparators[n][split]=metrics(b,n)
    # Top candidate results shown for audit, but not tested unless selected
    scoreboard.sort(key=lambda r:(
        -(r["validation"]["roi_percent"] or -float("inf")),
        -(r["validation"]["bets"])))
    with (ROOT/"validation_candidates.json").open("w") as handle:
        json.dump(scoreboard,handle,ensure_ascii=False,indent=2)
    policy_importance={k:v["importance"] for k,v in models.items()}
    (ROOT/"full_score_features_importance.json").write_text(
        json.dumps(policy_importance,ensure_ascii=False,indent=2),encoding="utf-8")
    scored[["race_id","date","phase","head_rank","second_rank","head_bike","second_bike",
            "gap12","gap23","gap34","gap45","reward_multiple","won","pay_yen",
            "selected_ticket"]].to_csv(ROOT/"all_ticket_policy_predictions.csv",index=False)
    report={
      "status":"complete","research_question":"Does use of full fixed-AI score geometry yield robust positive payout return?",
      "model_unchanged":True,"base_AI_overwrite":False,
      "purchase_model_files":[k+".joblib" for k in models],
      "source":"frozen 44-feature model + official exacta winning payouts",
      "race_counts_by_phase":counts,"candidate_ticket_rows_by_phase":{
        k:int((scored["phase"]==k).sum()) for k in counts},
      "parse_errors":bad,
      "evaluation_unit":"All possible ordered pairs in each race at hypothetical flat 100-yen ticket",
      "model_candidates":[{"name":k,"importance":v["importance"]} for k,v in models.items()],
      "selection":"Choose only profitable train+validation trees with >=80 tickets, >=30 validation races and >=5 validation hits. Order by leave-biggest-win-out validation ROI; test 2026Q1 exactly once.",
      "selected":selected,
      "baseline_results":comparators,
      "score_gap_quartiles":explain_group(scored),
      "best_validation_candidates":scoreboard[:10],
      "cautions":[
        "Winning payouts on settled tickets permit strategy backtest but NOT point-in-time price advantage identification.",
        "Historical market odds for losing exacta tickets are missing, so market-relative per-ticket EV cannot be established.",
        "Net live torigami filtering and line-score live corrections are absent.",
        "Only matched S-class F1 races represented; missing race data are excluded.",
        "Rare high payouts make fitted policies and ROI highly volatile.",
        "Out-of-sample performance MUST be interpreted from reserved test quarter, not the development or selection quarter.",
        "A test-period profitable outcome does not guarantee future profit.",
        "A failed gate implies no supported profitable purchase policy, not an obligation to place bets."
      ]}
    (ROOT/"score_shape_roi_learning_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("LEARNING SAMPLE COUNTS",json.dumps(counts),flush=True)
    print("LEARNING SELECTED",json.dumps(selected,ensure_ascii=False),flush=True)
    print("LEARNING BASELINES",json.dumps(comparators,ensure_ascii=False),flush=True)
    print("LEARNING TOP VALIDATION",json.dumps(scoreboard[:5],ensure_ascii=False),flush=True)
    print("LEARNING IMPORTANCE",json.dumps(policy_importance,ensure_ascii=False),flush=True)
    print("LEARNING COMPLETE",flush=True)

if __name__=="__main__":
    main()
