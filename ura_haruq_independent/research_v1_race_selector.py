#!/usr/bin/env python3
"""Read-only, chronological study of the exact frozen URA-HARU V1 formation.

2024 = train selector; 2025 = choose selector; 2026 Jan-Sep = holdout.
Predictions are from previously generated, out-of-year independent rankers.
No historical result or payout is allowed into the selection feature set.
This is NOT a production betting policy or live odds EV estimator.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ura_haruq_independent.frozen_ura_haru_v1.URA_HARU_V1_FROZEN_RULE import (
    BET_YEN_PER_TICKET, POLICY_ID, formation,
)

EXPECTED_RULE_SHA = "b0891724787f3c300a2b58d0d19cc6cf4ff421225ea53c81b94ab3add7194818"
EXPECTED_ARTIFACT_RUN = 40076792952
OUT_COLUMNS = [
    "race_id","race_date","forecast_year","grade","class_group",
    "score_gap","score_spread","skill_spread","skill_std",
    "pair_skill_gap","third_skill_mean","first_pair_skill_mean",
    "line_pattern","tickets","points","cost_yen","winning_combo",
    "official_payout_yen","winning_final_odds_proxy","hit",
    "return_yen","profit_yen","selector_selected",
]
NUMERIC = ("score_gap","score_spread","skill_spread",
           "skill_std","pair_skill_gap","third_skill_mean",
           "first_pair_skill_mean")
TRAIN_YEAR, VALID_YEAR, HOLDOUT_YEAR = "2024", "2025", "2026"


def assert_rule() -> None:
    p = Path(__file__).resolve().parent / "frozen_ura_haru_v1" / "URA_HARU_V1_FROZEN_RULE.py"
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != EXPECTED_RULE_SHA:
        raise RuntimeError("FROZEN_V1_RULE_SHA_CHANGED: "+actual)


def combo(text):
    value = str(text).strip()
    nums = [int(s) for s in re.findall(r"\d+", value)]
    return tuple(nums) if len(nums) == 3 and len(set(nums)) == 3 else None


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def clean_line(val):
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"nan","none","null","unknown","?"} else s


def read_predictions(path: Path):
    with (gzip.open(path,"rt",encoding="utf-8-sig",newline="") if
          str(path).endswith(".gz") else path.open(encoding="utf-8-sig",newline="")) as fh:
        data = list(csv.DictReader(fh))
    required = {"race_id","race_date","forecast_year","field_size","predicted_bikes_by_rank",
                "predicted_scores_by_rank","score_gap","trifecta_combo","trifecta_payout"}
    if not data or not required <= set(data[0]):
        raise RuntimeError("INCOMPATIBLE_WALKFORWARD_PREDICTION_ARTIFACT")
    ids = [r["race_id"] for r in data]
    if len(ids) != len(set(ids)):
        raise RuntimeError("DUPLICATE_FORECAST_RACE_ID")
    years = Counter(str(x["forecast_year"]) for x in data)
    if not all(years[y] > 1000 for y in (TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR)):
        raise RuntimeError("INCOMPLETE_2024_2026_WALKFORWARD_FORECASTS "+str(years))
    return data,years


def load_entries(path: Path):
    with sqlite3.connect(path) as conn:
        tables = {x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"races","entries","race_results","race_payouts"} <= tables:
            raise RuntimeError("MISSING_HISTORICAL_TABLES")
        audit=conn.execute(
            "SELECT substr(date,1,4),COUNT(*) FROM races WHERE date >= '2023-01-01' "
            "GROUP BY substr(date,1,4)").fetchall()
        rows=conn.execute("""
          SELECT e.race_id, e.bike_number, e.line_group, e.avg_competition_score,
                 e.class, r.race_name, r.rider_count
          FROM entries e JOIN races r ON e.race_id=r.race_id
          WHERE r.date >= '2024-01-01' AND r.date < '2026-10-01'
          ORDER BY e.race_id
        """)
        entries=defaultdict(list)
        for rid,bike,line,skill,cls,rname,rcount in rows:
            entries[str(rid)].append((bike,line,skill,cls,rname,rcount))
        # Independent audit: one official payout per event, not a manufactured
        # return calculated from winning odds.
        pays={}
        for rid,winning,payout in conn.execute(
                "SELECT race_id,sanrentan_combo,sanrentan_payout FROM race_payouts"):
            pays[str(rid)]=(combo(winning),number(payout))
    return entries,pays,dict(audit)


def class_group(classes):
    xs=[str(x).upper() for x in classes]
    if any(re.match(r"^L\d*",x) for x in xs):
        return "GIRLS"
    if not xs or any(not x for x in xs):
        return "UNKNOWN"
    if all(x.startswith("S") for x in xs):
        return "S"
    if all(x.startswith("A") for x in xs):
        return "A"
    return "MIXED"


def evaluate_forecasts(predictions,entries,pays):
    counters=Counter()
    per_year=Counter()
    records=[]
    for pred in predictions:
        year=str(pred["forecast_year"])
        per_year[year]+=1
        rid=str(pred["race_id"])
        if year not in {TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR}:
            counters["out_of_scope_forecasts"]+=1
            continue
        if len(rid)!=12 or pred["race_date"][:4]!=year or rid[:4]!=year:
            counters["invalid_date"]+=1;continue
        if int(pred["field_size"])!=7:
            counters["not_7_riders"]+=1;continue
        saved=entries.get(rid)
        if not saved:
            counters["missing_asof_entries"]+=1;continue
        if len(saved)!=7 or len(set(x[0] for x in saved))!=7:
            counters["entry_count_or_bike_mismatch"]+=1;continue
        b=list(map(int,pred["predicted_bikes_by_rank"].split("-")))
        if len(b)!=7 or set(b)!={int(x[0]) for x in saved}:
            counters["ranking_entry_mismatch"]+=1;continue
        if any(re.search(r"ガールズ|ガ決|ガ般|Ｌ級|L級",str(x[4])) for x in saved):
            counters["girls_excluded"]+=1;continue
        category=class_group([x[3] for x in saved])
        if category=="GIRLS":
            counters["girls_excluded"]+=1;continue
        lines={int(x[0]):clean_line(x[1]) for x in saved}
        if any(not lines[k] for k in b):
            counters["historical_line_missing"]+=1;continue
        sizes=tuple(sorted(Counter(lines.values()).values(),reverse=True))
        if sizes!=(2,2,2,1):
            counters["not_2221"]+=1;continue
        gap=number(pred["score_gap"])
        if gap is None or not 0<=gap<1:
            counters["not_low"]+=1;continue
        s=[number(v) for v in pred["predicted_scores_by_rank"].split(";")]
        if len(s)!=7 or any(v is None for v in s) or abs((s[0]-s[1])-gap)>1e-4:
            counters["invalid_scores"]+=1;continue
        tickets=formation(b,lines,gap) # THE UNMODIFIED, FROZEN V1 FUNCTION
        if not tickets:
            counters["top_pair_not_same_line"]+=1;continue
        if len(tickets)!=6:
            raise RuntimeError("UNEXPECTED_FROZEN_V1_TICKET_COUNT_"+rid)
        saved_combo=combo(pred["trifecta_combo"])
        saved_pay=number(pred["trifecta_payout"])
        db_combo,db_pay=pays.get(rid,(None,None))
        if (saved_combo is None or db_combo!=saved_combo or saved_pay is None
                or db_pay is None or int(db_pay)!=int(saved_pay)
                or saved_pay<100 or not float(saved_pay).is_integer()):
            counters["unverified_official_payout"]+=1;continue
        skill={int(x[0]):number(x[2]) for x in saved}
        vals=[v for v in skill.values() if v is not None]
        rank_pair=[b[i] for i in range(3) if any(b[i] in t[:2] for t in tickets)]
        # There must be two selected riders (the pair); one distinct line.
        if len(rank_pair)!=2:
            raise RuntimeError("RANK_PAIR_DETECTION_FAILED_"+rid)
        third=sorted({t[2] for t in tickets})
        def spread(numbers):
            return max(numbers)-min(numbers) if len(numbers)>1 else None
        pair_values=[skill[k] for k in rank_pair]
        third_values=[skill[k] for k in third]
        full_skill=all(v is not None for v in vals) and len(vals)==7
        cost=BET_YEN_PER_TICKET*len(tickets)
        hit=int(saved_combo in tickets)
        ret=int(saved_pay) if hit else 0
        records.append({
            "race_id":rid, "race_date":pred["race_date"],"forecast_year":year,
            "grade":str(pred["grade"]), "class_group":category,
            "score_gap":round(gap,9),"score_spread":round(s[0]-s[-1],8),
            "skill_spread":round(spread(vals),4) if full_skill else None,
            "skill_std":round(float(np.std(vals)),4) if full_skill else None,
            "pair_skill_gap":round(spread(pair_values),4)
                       if all(x is not None for x in pair_values) else None,
            "third_skill_mean":round(float(np.mean(third_values)),4)
                       if all(x is not None for x in third_values) else None,
            "first_pair_skill_mean":round(float(np.mean(pair_values)),4)
                       if all(x is not None for x in pair_values) else None,
            "line_pattern":"2-2-2-1",
            "tickets":";".join("-".join(map(str,t)) for t in tickets),
            "points":len(tickets),"cost_yen":cost,
            "winning_combo":"-".join(map(str,saved_combo)),
            "official_payout_yen":int(saved_pay),
            "winning_final_odds_proxy":round(saved_pay/100,2),
            "hit":hit,"return_yen":ret,"profit_yen":ret-cost,
            "selector_selected":0,
        })
        counters["settled_v1_races"]+=1
    return records,dict(counters),dict(per_year)


def stat(rows):
    amount=sum(int(x["cost_yen"]) for x in rows)
    ret=sum(int(x["return_yen"]) for x in rows)
    days=len(set(x["race_date"] for x in rows))
    wins=[x["return_yen"] for x in rows if x["return_yen"]>0]
    return {
        "races":len(rows),"distinct_days":days,
        "hits":len(wins),
        "hit_rate_pct":round(100*len(wins)/len(rows),4) if rows else None,
        "investment_yen":amount,"return_yen":ret,"profit_yen":ret-amount,
        "roi_pct":round(100*ret/amount,4) if amount else None,
        "max_hit_return_share":round(max(wins)/ret,5) if ret else None,
    }


def describe_win_loss(train):
    positive=[r for r in train if r["profit_yen"]>0]
    negative=[r for r in train if r["profit_yen"]<=0]
    diffs={}
    for feature in NUMERIC:
        a=[r[feature] for r in positive if r[feature] is not None]
        b=[r[feature] for r in negative if r[feature] is not None]
        diffs[feature]={
            "profitable_races":len(a),"losing_races":len(b),
            "median_profitable":round(float(np.median(a)),5) if a else None,
            "median_losing":round(float(np.median(b)),5) if b else None,
        }
    for name in ("grade","class_group"):
        diffs[name]={
            key:{"profitable":sum(r[name]==key for r in positive),
                 "losing":sum(r[name]==key for r in negative)}
            for key in sorted({r[name] for r in train})
        }
    # Price of realized winning outcome is post-event data, never a selector.
    diffs["winning_final_odds_proxy"]={
        "label":"POST_EVENT_DIAGNOSTIC_ONLY",
        "profitable_races_median":round(float(np.median(
            [r["winning_final_odds_proxy"] for r in positive])),3) if positive else None,
        "losing_races_median":round(float(np.median(
            [r["winning_final_odds_proxy"] for r in negative])),3) if negative else None,
    }
    return diffs


def conditions(train):
    conditions_list=[("all_v1",lambda r: True)]
    predicates=[]
    for feat in NUMERIC:
        vals=[r[feat] for r in train if r[feat] is not None]
        if len(vals)<50:
            continue
        for q in (.25,.5,.75):
            t=round(float(np.quantile(vals,q)),5)
            predicates.extend([
                (f"{feat}<={t:g}",lambda r,k=feat,v=t:r[k] is not None and r[k]<=v),
                (f"{feat}>{t:g}",lambda r,k=feat,v=t:r[k] is not None and r[k]>v)
            ])
    for grade in sorted({r["class_group"] for r in train}):
        if grade not in ("A","S"):
            continue
        predicates.append((f"class_group={grade}",lambda r,v=grade:r["class_group"]==v))
    # One and two dimensional selectors, no outcome or odds proxy features.
    conditions_list.extend(predicates)
    for (a,pa),(b,pb) in itertools.combinations(predicates,2):
        if a.split("<")[0].split(">")[0] == b.split("<")[0].split(">")[0]:
            continue
        conditions_list.append((a+" & "+b,lambda r,p=pa,q=pb:p(r) and q(r)))
    return conditions_list


def fit_selector(train,valid):
    tested=[]
    for name,predicate in conditions(train):
        a=[r for r in train if predicate(r)]
        b=[r for r in valid if predicate(r)]
        sa,sb=stat(a),stat(b)
        acceptable=(
            sa["races"]>=40 and sb["races"]>=40 and
            sa["distinct_days"]>=20 and sb["distinct_days"]>=20 and
            sa["hits"]>=4 and sb["hits"]>=4 and
            sa["roi_pct"] is not None and sb["roi_pct"] is not None and
            sa["roi_pct"]>100 and sb["roi_pct"]>100 and
            sa["max_hit_return_share"]<.5 and sb["max_hit_return_share"]<.5
        )
        tested.append({"name":name,"train":sa,"validation":sb,
                       "passes_support_and_profit":bool(acceptable),
                       "predicate":predicate})
    passing=[c for c in tested if c["passes_support_and_profit"]]
    # Select ONLY from 2024/2025. Never consult 2026 for model selection.
    passing.sort(key=lambda c:(min(c["train"]["roi_pct"],c["validation"]["roi_pct"]),
                               c["validation"]["profit_yen"],c["name"]),reverse=True)
    chosen=passing[0] if passing else None
    sanitized=[{k:v for k,v in t.items() if k!="predicate"} for t in tested]
    return chosen,sanitized,len(passing)


def daily_bootstrap(rows,rep=3000):
    if not rows:return None
    per_day=defaultdict(lambda:[0,0])
    for r in rows:
        v=per_day[r["race_date"]]
        v[0]+=int(r["cost_yen"])
        v[1]+=int(r["return_yen"])
    d=np.array(list(per_day.values()),dtype=float)
    rng=np.random.default_rng(20261026)
    indexes=rng.integers(0,len(d),size=(rep,len(d)))
    vals=d[indexes].sum(axis=1)
    rois=np.where(vals[:,0]>0,100*vals[:,1]/vals[:,0],np.nan)
    valid=rois[np.isfinite(rois)]
    return {"method":"calendar_day_cluster_bootstrap",
            "replicates":rep,
            "95pct_interval":list(map(lambda x:round(float(x),2),
                          np.quantile(valid,[.025,.975]))) if len(valid) else None}


def save_csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--db",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-run",type=int,default=EXPECTED_ARTIFACT_RUN)
    a=p.parse_args()
    assert_rule()
    if a.source_run!=EXPECTED_ARTIFACT_RUN:
        raise RuntimeError("UNREVIEWED_FORECAST_SOURCE_RUN")
    preds,coverage=read_predictions(a.predictions)
    entries,pays,db_years=load_entries(a.db)
    rows,reasons,counts=evaluate_forecasts(preds,entries,pays)
    train=[r for r in rows if r["forecast_year"]==TRAIN_YEAR]
    valid=[r for r in rows if r["forecast_year"]==VALID_YEAR]
    holdout=[r for r in rows if r["forecast_year"]==HOLDOUT_YEAR]
    if min(len(train),len(valid),len(holdout)) < 100:
        raise RuntimeError("INSUFFICIENT_VERIFIED_V1_RACES "+str(Counter(r["forecast_year"] for r in rows)))
    chosen,examined,passes=fit_selector(train,valid)
    # No alternative rules are created if validation found no stable profitable candidate.
    chosen_cond=chosen["predicate"] if chosen is not None else (lambda r:False)
    for row in rows:
        row["selector_selected"]=int(chosen_cond(row))
    selected={y:[r for r in rows if r["forecast_year"]==y and r["selector_selected"]]
              for y in (TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR)}
    baseline={y:[r for r in rows if r["forecast_year"]==y]
              for y in (TRAIN_YEAR,VALID_YEAR,HOLDOUT_YEAR)}
    report={
        "research_id":"URA_HARU_V1_CHRONOLOGICAL_RACE_SELECTOR_20261026",
        "production_changes":False,"research_only":True,
        "frozen_policy_id":POLICY_ID,"frozen_rule_sha256":EXPECTED_RULE_SHA,
        "source_chronological_forecast_run":a.source_run,
        "source_model_semantics":"Separate LightGBM fit on earlier years for each forecast year; raw score scales may differ",
        "train":"2024-01-01 to 2024-12-31",
        "selector_validation":"2025-01-01 to 2025-12-31",
        "heldout_scoring":"2026-01-01 to 2026-09-30",
        "asof_feature_line_audit_completed":False,
        "true_unseen_researcher_blind":False,
        "pre_race_3rentan_odds_complete":False,
        "winning_final_odds_proxy_is_post_event_only":True,
        "full_database_races_by_year":db_years,
        "prediction_artifact_races_by_year":coverage,
        "scoring_exclusions":reasons,
        "eligible_settled_v1_by_year":{y:len(baseline[y]) for y in baseline},
        "train_profit_vs_loss_diagnostics":describe_win_loss(train),
        "baseline_v1":{y:stat(baseline[y]) for y in baseline},
        "selector":{
            "chosen_condition":chosen["name"] if chosen else "NO_BET",
            "candidates_evaluated":len(examined),
            "passing_train_validation":passes,
            "minimum_support":"40 races, 20 days, 4 hits each train and validation; both ROI >100%; largest payout contribution <50%",
            "train_val_only_ranking":"Maximize lower of 2024 and 2025 observed ROI; 2026 excluded",
            "period_results":{y:stat(selected[y]) for y in selected},
            "holdout_calendar_day_bootstrap_95pct":daily_bootstrap(selected[HOLDOUT_YEAR]),
            "holdout_base_bootstrap_95pct":daily_bootstrap(baseline[HOLDOUT_YEAR]),
        },
        "limitations":[
            "2013/2023-era predictions not supplied; 2023 only appears in database coverage and model training, never as falsely out-of-sample tested bets.",
            "Existing 2024-2026 predictions are chronologically trained but underlying feature as-of/odds leakage is not fully audited.",
            "Independent 2024, 2025 and 2026 rankers have different raw score scales, so V1 LOW coverage may differ by year.",
            "Historical entries' line source timing has not been independently proved.",
            "Official payout checked against both historical database and chronological research artifact.",
            "Historical 3連単 odds per ticket at a purchasable timestamp are not available to this run; retrospective winning final odds cannot select races.",
            "Selection uses 2024 and 2025 only, but 2026 has been inspected in other historical research and is not fully researcher-blinded.",
            "This is a new optional research-only selection layer, not a modification or deployment of the frozen V1 model or web.",
        ]
    }
    a.out.mkdir(parents=True,exist_ok=True)
    save_csv(a.out/"every_verified_v1_race.csv",rows,OUT_COLUMNS)
    save_csv(a.out/"selector_candidates_2024_2025.csv",
             [{"condition":r["name"],"support_pass":int(r["passes_support_and_profit"]),
               **{f"{year}_{field}":r[k][field]
                  for k,year in (("train","2024"),("validation","2025"))
                  for field in ("races","hits","roi_pct","profit_yen","max_hit_return_share")}}
              for r in examined],
             ["condition","support_pass"]+[
                 f"{year}_{field}" for year in ("2024","2025")
                 for field in ("races","hits","roi_pct","profit_yen","max_hit_return_share")])
    (a.out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("V1_RESEARCH_SUMMARY",json.dumps({
        "base_2024":report["baseline_v1"][TRAIN_YEAR],
        "base_2025":report["baseline_v1"][VALID_YEAR],
        "base_2026":report["baseline_v1"][HOLDOUT_YEAR],
        "selected":report["selector"]["chosen_condition"],
        "holdout":report["selector"]["period_results"][HOLDOUT_YEAR],
        "exclusions":reasons,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
