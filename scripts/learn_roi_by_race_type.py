#!/usr/bin/env python3
"""Race-type-aware wager-policy learning over the entire available *out-of-sample*
frozen base LambdaRank period (2025-01-01 through 2026-03-31).

DO NOT identify all keirin race types with F1. Match official post-race 2車単
outcomes by race ID and verify finish order. Keep grade, raw Japanese race type,
phase of race (prelim/semi/final etc), division (A3/A1-A2/S) and field size
separate as purchase-model inputs. Base prediction weights NEVER change.

Chronology: 2025H1 development, 2025Q3 validation, 2025Q4 + 2026Q1
frozen forward test. Period 2025H1 was held out from base fitting but used for
base model early stopping; development only. The strict return test starts Q4.
Loss is 100-yen settled-tick return. NOT ex-ante EV: losing ticket
decision-time odds unavailable. No selected strategy will be installed unless
independent test evidence passes robust gate.
"""
import csv
import io
import json
import math
import re
import sqlite3
import sys
import unicodedata
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config import CLASS_MAP,DB_PATH,DEFAULT_GEAR_RATIO
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
ROOT=Path("race_type_roi_outputs");ROOT.mkdir(exist_ok=True)
SRC=Path("race_type_payout_sources")
MODEL=Path("fixed_model/model_lgbm_44_retrained.pkl")
START="2025-01-01"
END="2026-04-01"
CATS=("division","grade","stage","race_type","rider_count","line_shape")
NUMS=["head_rank","second_rank","gap12","gap23","gap34","gap45",
      "gap_1st_head","gap_1st_second","head_second_gap","score_spread",
      "head_next_gap","second_next_gap","normalized_pair_strength",
      "score_std","rider_count"]
MIN_TEST_RACES=80

def normal(x):
    return unicodedata.normalize("NFKC",str(x or "")).replace(" ","").strip()

def stage_of(raw):
    s=normal(raw)
    checks=[
        ("準決勝",["準決"]),
        ("準々決勝",["準々決"]),
        ("決勝",["決勝","ファイナル"]),
        ("初日特選",["初日特選","初特選"]),
        ("特別選抜",["特別選抜"]),
        ("特別一般",["特一般"]),
        ("二次予選",["二次予選","二予選","二予"]),
        ("一次予選",["一次予選","一予選"]),
        ("特別予選",["特予選"]),
        ("予選",["予選"]),
        ("特別優秀",["特秀","特別優秀"]),
        ("特選",["特選"]),
        ("一般",["一般"]),
        ("選抜",["選抜"]),
        ("敗者戦",["敗者"]),
    ]
    # Specific labels must win over substring labels (準決勝 != 決勝).
    for key,parts in checks:
        if any(t in s for t in parts):
            return key
    return "その他"

def division_of(raw,source,grade):
    name=normal(raw)
    src=str(source)
    if "チャレンジ" in name or "a3_challenge" in src:return "A3"
    if name.startswith("A級") or "a_class_12" in src:return "A1_A2"
    if name.startswith("S級") or "s_class" in src:return "S"
    if grade in ("G1","G2","G3","GP"):return "S"
    return "unknown"

def read_archived_sources():
    """Index all uploaded CSV / ZIP member payout labels, check duplicates.
    No training data is selected by payout value."""
    records=defaultdict(set)
    provenance=Counter()
    broken=Counter()
    by_type=Counter()
    files=sorted(SRC.glob("*"))
    for path in files:
        if path.suffix==".csv":
            handles=[(path.name,path.open("r",encoding="utf-8-sig",newline=""))]
        elif path.suffix==".zip":
            handles=[]
            z=zipfile.ZipFile(path)
            candidates=[n for n in z.namelist() if n.lower().endswith(".csv") and ("payout" in n.lower() or "払戻" in n)]
            for member in candidates:
                try:
                    buf=z.read(member).decode("utf-8-sig")
                    handles.append((path.name+"/"+member,io.StringIO(buf)))
                except UnicodeDecodeError:
                    broken["zip_member_decode"]+=1
        else: continue
        count=0
        for label,handle in handles:
            try:
                reader=csv.DictReader(handle)
                if not reader.fieldnames:
                    broken["empty_member"]+=1;continue
                names=set(reader.fieldnames)
                if not {"race_id","ticket_type","combination","payout_yen"}<=names:
                    broken["different_schema"]+=1;continue
                for r in reader:
                    if normal(r.get("ticket_type"))!="2車単" or normal(r.get("status")) not in ("paid",""):
                        continue
                    rid=str(r.get("race_id",""))
                    if len(rid)!=16 or not rid.isdigit():
                        broken["bad_id"]+=1;continue
                    d=str(r.get("race_date",""))
                    if not START<=d<END:continue
                    try:
                        no=int(r.get("race_no") or int(rid[-4:]))
                        cars=tuple(int(x) for x in re.findall(r"\d+",str(r.get("combination",""))))
                        yen=int(float(r.get("payout_yen","0")))
                        if len(cars)!=2 or len(set(cars))!=2 or not 100<=yen<=10000000:continue
                        key=(d,rid[:2],no)
                        raw=normal(r.get("race_type",""))
                        records[key].add((cars[0],cars[1],yen,raw,label))
                        by_type[raw]+=1
                        provenance[label]+=1
                        count+=1
                    except (ValueError,TypeError):
                        broken["bad_payout_row"]+=1
            finally:
                handle.close()
        if path.suffix==".zip":z.close()
        print("PAY_FILE",path.name,"entries",count,flush=True)
    print("RAW_ARCHIVE_RECORDS",len(records),"ROWS",sum(provenance.values()),
          "FILES",len(provenance),"RAW_TYPES",len(by_type),flush=True)
    print("ARCHIVE_TYPE_COUNTS",json.dumps(by_type,ensure_ascii=False),flush=True)
    # Reconcile duplicated payouts across archives, with contradictory outcomes
    # strictly discarded. Race-type disagreements audited, never picked by payout.
    settled={}
    for key,items in records.items():
        paid={(a,b,y) for a,b,y,raw,label in items}
        if len(paid)!=1:
            broken["inconsistent_payout_for_id"]+=1;continue
        a,b,y=next(iter(paid))
        labels={raw for _,_,_,raw,_ in items if raw}
        if len(labels)>1:
            # Two distinct official labels => do not randomly assign a race type
            broken["inconsistent_race_type"]+=1;continue
        raw=next(iter(labels)) if labels else ""
        if not raw:
            broken["missing_race_type"]+=1;continue
        sample=next(iter(items))
        source=next(iter(sorted(i[-1] for i in items)))
        settled[key]={"first":a,"second":b,"paid":y,"race_type":raw,
                      "source":source,"all_sources":len(items)}
    return settled,dict(provenance),dict(broken),dict(by_type)

def valid_ids(conn):
    return {x[0] for x in conn.execute("""
       SELECT r.race_id FROM races r
       JOIN race_results rr ON rr.race_id=r.race_id
       GROUP BY r.race_id
       HAVING COUNT(*)=r.rider_count AND
          SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
          AND COUNT(DISTINCT rr.finish_position)=COUNT(*)
    """)}

def line_shape(g):
    a=g["line_group"].fillna("").astype(str)
    sizes=sorted([int(v) for v in a[a!=""].value_counts().values]+[1]*int((a=="").sum()),reverse=True)
    return "-".join(map(str,sizes)) if sizes else "unknown"

def phase(d):
    if d<"2025-07-01":return "dev"
    if d<"2025-10-01":return "val"
    if d<"2026-01-01":return "test_q4"
    return "test_q1"

def build_scores_and_join(archive):
    con=sqlite3.connect(f"file:{DB_PATH}?mode=ro",uri=True)
    con.execute("PRAGMA query_only=ON")
    valid=valid_ids(con)
    gear=pd.read_sql_query("SELECT race_id,rider_id,gear_ratio AS entry_gear FROM entries",con)
    meta=pd.read_sql_query("SELECT race_id,grade,date,rider_count FROM races",con)
    con.close()
    base=LGBMRanker();base.load(str(MODEL))
    b=FeatureBuilder(); cols=b.feature_names
    assert len(cols)==44 and cols==base.feature_names
    frame=b.build_dataset(2022,2026)
    frame=frame[frame["race_id"].isin(valid) & frame["race_date"].between(START,END,inclusive="left")].copy()
    frame["rider_class_num"]=frame["class"].map(CLASS_MAP).fillna(6).astype(float)
    frame["class_rank_in_race"]=frame.groupby("race_id")["rider_class_num"].rank(method="min")
    frame["rider_class_x_jiku"]=frame["rider_class_num"]*frame["line_is_jiku"]
    frame=frame.merge(gear,on=["race_id","rider_id"],how="left",validate="many_to_one")
    frame["entry_gear_ratio"]=pd.to_numeric(frame["entry_gear"],errors="coerce").fillna(DEFAULT_GEAR_RATIO)
    frame=frame.merge(meta,on="race_id",how="left",validate="many_to_one")
    frame["model_score"]=base.predict(frame[cols].apply(pd.to_numeric,errors="coerce").fillna(0.0)).values
    print("BASE_OUT_OF_FIT_RACES",frame["race_id"].nunique(),flush=True)
    races=[];tickets=[];audit=Counter(); type_count=Counter()
    for rid,g in frame.groupby("race_id",sort=False):
        d=str(g["race_date"].iloc[0]);grp=phase(d)
        grade=str(g["grade"].iloc[0])
        key=(d,rid[8:10],int(rid[10:12]))
        audit["all_base_future_races"]+=1
        official=archive.get(key)
        if not official:
            audit["no_archived_2exacta"]+=1;continue
        actual=g.sort_values("finish_position",kind="stable")
        a,bike=int(actual["bike_number"].iloc[0]),int(actual["bike_number"].iloc[1])
        if (official["first"],official["second"])!=(a,bike):
            audit["result_mismatch"]+=1;continue
        order=g.sort_values("model_score",ascending=False,kind="stable")
        if order["bike_number"].isna().any() or len(order)<5:
            audit["bad_ranking"]+=1;continue
        rank_bikes=[int(x) for x in order["bike_number"]]
        scores=[float(x) for x in order["model_score"]]
        if len(set(rank_bikes))!=len(rank_bikes) or any(not math.isfinite(s) for s in scores):
            audit["bad_scores"]+=1;continue
        raw=official["race_type"];stage=stage_of(raw)
        division=division_of(raw,official["source"],grade)
        if division=="unknown":
            audit["division_unknown"]+=1;continue
        n=len(order);shape=line_shape(g)
        # race type is sourced from published race descriptor, never finish outcome
        # or payout amount. Payout value used only as retrospective label.
        event={"race_id":rid,"date":d,"phase":grp,"grade":grade,
           "division":division,"stage":stage,"race_type":raw,
           "rider_count":n,"line_shape":shape,
           "winning_exacta":f"{a}-{bike}",
           "payout_100":official["paid"],
           "ranked_bikes":"-".join(map(str,rank_bikes)),
           "score_vector":json.dumps(scores,separators=(",",":"))}
        races.append(event)
        type_count[(division,grade,stage,raw,grp)]+=1
        audit["matched_races"]+=1
        for i in range(n):
            for j in range(n):
                if i==j:continue
                p={"race_id":rid,"date":d,"phase":grp,"grade":grade,
                   "division":division,"stage":stage,"race_type":raw,
                   "rider_count":n,"line_shape":shape,
                   "head_rank":i+1,"second_rank":j+1,
                   "gap12":scores[0]-scores[1],
                   "gap23":scores[1]-scores[2],
                   "gap34":scores[2]-scores[3],
                   "gap45":scores[3]-scores[4],
                   "gap_1st_head":scores[0]-scores[i],
                   "gap_1st_second":scores[0]-scores[j],
                   "head_second_gap":scores[i]-scores[j],
                   "score_spread":scores[0]-scores[-1],
                   "head_next_gap":scores[i]-scores[i+1] if i<n-1 else 0.,
                   "second_next_gap":scores[j]-scores[j+1] if j<n-1 else 0.,
                   "normalized_pair_strength":(2*scores[0]-scores[i]-scores[j])/(scores[0]-scores[-1]+1e-5),
                   "score_std":float(np.std(scores)),
                   "won":int((rank_bikes[i],rank_bikes[j])==(a,bike)),
                   "reward":official["paid"]/100 if (rank_bikes[i],rank_bikes[j])==(a,bike) else 0.,
                   "payout_yen":official["paid"] if (rank_bikes[i],rank_bikes[j])==(a,bike) else 0}
                tickets.append(p)
    raceframe=pd.DataFrame(races)
    pairframe=pd.DataFrame(tickets)
    if len(raceframe)==0: raise RuntimeError("ZERO MATCHED RACES")
    raceframe.to_csv(ROOT/"matched_race_types.csv",index=False)
    pairframe.to_csv(ROOT/"all_exacta_ticket_outcomes.csv",index=False)
    counts={str(k):v for k,v in sorted(type_count.items(),key=lambda p:-p[1])}
    print("MATCHED_COUNTS",json.dumps(dict(audit)),flush=True)
    print("MATCHED_BY_DIVISION",json.dumps(raceframe.groupby(["division","grade","stage","phase"]).size().to_dict().__repr__()),flush=True)
    return pairframe,raceframe,dict(audit),counts

def design(frame,typeaware=True):
    numerical=frame[NUMS].apply(pd.to_numeric,errors="coerce").fillna(0.0).reset_index(drop=True)
    if not typeaware:return numerical
    # pre-register categories using whole schema (not labels/returns), and
    # ensure same one-hot columns in all periods.
    category=frame[["grade","division","stage","race_type","line_shape"]].astype(str).reset_index(drop=True)
    return pd.concat([numerical,pd.get_dummies(category,prefix_sep="=",dtype=np.int8)],axis=1)

def metric(rows,tag):
    n=len(rows)
    if n==0:return {"tag":tag,"bets":0,"races":0,"roi_pct":None}
    n_races=rows["race_id"].nunique()
    gain=float(rows["payout_yen"].sum())
    invested=100*n
    return {"tag":tag,"bets":n,"races":int(n_races),"hits":int(rows["won"].sum()),
        "paid_yen":round(gain),"stake_yen":invested,
        "profit_yen":int(round(gain-invested)),
        "roi_pct":round(100*gain/invested,3),
        "largest_one_payout":int(rows["payout_yen"].max())}

def filter_strategy(table,policy,threshold):
    if policy=="rank12":
        return table[(table["head_rank"]==1)&(table["second_rank"]==2)]
    if policy=="rank21":
        return table[(table["head_rank"]==2)&(table["second_rank"]==1)]
    if policy=="original_four":
        return table[(table["head_rank"]==1)&table["second_rank"].between(2,5)&(table["gap12"]>=1.5)]
    if policy=="all_four":
        return table[(table["head_rank"]==1)&table["second_rank"].between(2,5)]
    return table[table[policy]>threshold]

def robust_lower_bound(rows,n_iter=750):
    if len(rows)==0:return None
    by=rows.groupby("race_id").agg(n=("payout_yen","size"),pay=("payout_yen","sum")).to_numpy(float)
    rng=np.random.default_rng(20261023)
    values=[]
    for _ in range(n_iter):
        rs=by[rng.integers(0,len(by),size=len(by))]
        values.append(100*rs[:,1].sum()/(100*rs[:,0].sum()))
    return [round(float(x),2) for x in np.quantile(values,[.025,.975])]

def learn(frame,raceframe,audit,types,provenance,problems):
    # All races, all official race types; no selection by grade or payout.
    x_wide=design(frame,True)
    x_pooled=design(frame,False)
    train_mask=(frame["phase"]=="dev").to_numpy()
    val_mask=(frame["phase"]=="val").to_numpy()
    test_mask=(frame["phase"].str.startswith("test")).to_numpy()
    featuremat={"pooled":x_pooled,"race_type_aware":x_wide}
    models={}
    scores=frame.copy()
    grid=[
       ("pool_d4_l500","pooled",4,500,100),
       ("pool_d5_l700","pooled",5,700,150),
       ("type_d4_l500","race_type_aware",4,500,100),
       ("type_d5_l700","race_type_aware",5,700,150),
       ("type_d6_l400","race_type_aware",6,400,100),
    ]
    for name,key,depth,leaf,cap in grid:
        model=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=20261023)
        features=featuremat[key]
        model.fit(features.loc[train_mask],
                  frame.loc[train_mask,"reward"].clip(upper=cap))
        scores[name]=model.predict(features)
        models[name]={"uses_race_type":key=="race_type_aware",
                      "features":features.columns.tolist(),
                      "importance":dict(sorted(zip(features.columns.tolist(),model.feature_importances_.tolist()),
                                               key=lambda x:-x[1])[:18]),
                      "nodes":int(model.tree_.node_count)}
    candidates=[]
    for name,*_ in grid:
        for th in [0.8,0.95,1.1,1.3]:
            train=filter_strategy(scores[scores["phase"]=="dev"],name,th)
            val=filter_strategy(scores[scores["phase"]=="val"],name,th)
            tr=metric(train,"dev");va=metric(val,"val")
            if len(val):
                leave_largest=(val["payout_yen"].sum()-val["payout_yen"].max())/len(val)
            else: leave_largest=0.
            eligible=bool(tr["bets"]>=150 and va["bets"]>=150 and
               va["races"]>=60 and va.get("hits",0)>=8 and
               (tr.get("roi_pct") or 0)>=90 and
               (va.get("roi_pct") or 0)>100 and
               leave_largest>100)
            candidates.append({"name":name,"threshold":th,"train":tr,"validation":va,
                              "validation_leave_largest_roi":round(float(leave_largest),2),
                              "eligible":eligible})
    accepted=[c for c in candidates if c["eligible"]]
    # choose on Q3 (val) before looking at Q4 or 2026Q1
    chosen=sorted(accepted,key=lambda c:(-c["validation_leave_largest_roi"],
                      -c["validation"]["roi_pct"],c["name"],c["threshold"]))[0] if accepted else None
    comparisons={}
    for phase in ("dev","val","test_q4","test_q1"):
        a=scores[scores["phase"]==phase]
        comparisons[phase]={p:metric(filter_strategy(a,p,None),p)
                            for p in ["rank12","rank21","all_four","original_four"]}
        comparisons[phase]["base_races_in_phase"]=int(raceframe[raceframe["phase"]==phase].shape[0])
    learned=None
    if chosen:
        name,th=chosen["name"],chosen["threshold"]
        learned={"strategy":name,"predicted_return_threshold":th,"model_detail":models[name],
                 "develop":chosen["train"],"validation":chosen["validation"]}
        for phase in ("test_q4","test_q1"):
            b=filter_strategy(scores[scores["phase"]==phase],name,th)
            learned[phase]=metric(b,phase)
            learned[phase+"_bootstrap95"]=robust_lower_bound(b)
        b=filter_strategy(scores[scores["phase"].str.startswith("test")],name,th)
        learned["combined_forward"]=metric(b,"combined_forward")
        learned["combined_forward_bootstrap95"]=robust_lower_bound(b)
        learned["deployable"]=bool(
            all((learned[p]["roi_pct"] or 0)>100 for p in ("test_q4","test_q1")) and
            learned["combined_forward_bootstrap95"] is not None and
            learned["combined_forward_bootstrap95"][0]>100)
        # Explicitly no live integration even if pass: need pre-deadline exacta odds.
        learned["live_deployed"]=False
    type_reports={}
    for phase in ("dev","val","test_q4","test_q1"):
        frame_phase=scores[scores["phase"]==phase]
        for (div,grade,stage),group in frame_phase.groupby(["division","grade","stage"]):
            k=f"{phase}/{div}/{grade}/{stage}"
            type_reports[k]={
                "matched_races":int(group["race_id"].nunique()),
                "all_four":metric(filter_strategy(group,"all_four",None),"all_four"),
                "rank12":metric(filter_strategy(group,"rank12",None),"rank12"),
            }
            if chosen:
                typ_b=filter_strategy(group,chosen["name"],chosen["threshold"])
                type_reports[k]["learned"]=metric(typ_b,"learned")
    report={
       "status":"completed","model_fixed":True,
       "user_constraint":"All available categories, distinct race-level, event grade and official race type. No F1-only filter.",
       "base_ai":"44-feature retrained LambdaRank, existing frozen model",
       "training":"2025H1, used as base early-stopping validation; purchase model development only",
       "validation":"2025Q3",
       "forward_tests":["2025Q4","2026Q1"],
       "matched_races":len(raceframe),"ticket_rows":len(frame),
       "unmatched_audit":audit,
       "archives":provenance,"archive_issues":problems,
       "grade_counts":raceframe.groupby("grade").size().to_dict(),
       "division_counts":raceframe.groupby("division").size().to_dict(),
       "stage_counts":raceframe.groupby("stage").size().to_dict(),
       "phase_counts":raceframe.groupby("phase").size().to_dict(),
       "official_type_counts":raceframe.groupby(["division","grade","race_type"]).size().to_dict().__repr__(),
       "matched_by_type_phase":types,
       "models":models,
       "chosen_policy":learned,
       "number_of_candidates":len(candidates),
       "number_passing_train_validation":len(accepted),
       "best_validation_candidates":sorted(candidates,key=lambda c:-(c["validation"]["roi_pct"] or 0))[:8],
       "base_comparisons":comparisons,
       "types_report_file":"per_race_type_policy_metrics.csv",
       "market_caveat":"Only winning exacta payouts known. No complete before-bet two-way exacta odds. Cannot infer price-conditioned expected value, nor fully replicate original torigami skip.",
       "limits":["The historical feature builder may include future information unless audited feature by feature.",
                 "Source archive coverage differs by grade and race type.",
                 "Sparse race types should not be claimed profitable based on tiny subsets.",
                 "The purchase policy was selected using Q3 results, and Q4/Q1 are not used for selection.",
                 "A profitable historical slice is not a guarantee of a live edge.",
                 "Any model failing forward test remains research only. No live purchase policy deployed."]}
    # no JSON tuples and protected serialization
    (ROOT/"race_type_learning_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    pd.DataFrame(candidates).to_json(ROOT/"candidate_validation.json",orient="records",force_ascii=False,indent=2)
    pd.DataFrame([{"key":k,**v} for k,v in type_reports.items()]).to_json(
       ROOT/"per_race_type_policy_metrics.json",orient="records",force_ascii=False,indent=2)
    # Compact sale summary, no outcome leakage in inference features
    cols=["race_id","date","phase","grade","division","stage","race_type","head_rank","second_rank","won","payout_yen"]
    if chosen:cols+= [chosen["name"]]
    scores[cols].to_csv(ROOT/"scored_ticket_evidence.csv",index=False)
    print("RACE_TYPE_LEARNING_COUNTS",json.dumps({
      "total":len(raceframe),"by_grade":report["grade_counts"],
      "by_division":report["division_counts"],"by_phase":report["phase_counts"],
      "by_stage":report["stage_counts"]},ensure_ascii=False),flush=True)
    print("RACE_TYPE_LEARNING_RESULT",json.dumps({
       "qualified_policy":learned,
       "candidate_count":len(candidates),"validation_pass":len(accepted),
       "base_comparisons":comparisons},ensure_ascii=False),flush=True)
    print("RACE_TYPE_LEARNING_COMPLETE",flush=True)

def main():
    if not MODEL.is_file():raise RuntimeError("Missing frozen original retrained model")
    a,p,b,t=read_archived_sources()
    frame,raceframe,audit,type_counts=build_scores_and_join(a)
    learn(frame,raceframe,audit,type_counts,p,b)

if __name__=="__main__":main()
