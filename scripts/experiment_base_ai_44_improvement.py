#!/usr/bin/env python3
"""Prespecified improvement experiment for the original 44-feature LambdaRank.

No auxiliary model, new betting rules, or overwrite of released assets.
All variant selection is on the pre-holdout validation period.
"""
import hashlib
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_MAP, DB_PATH, DEFAULT_GEAR_RATIO, LGBM_PARAMS
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from models.trainer import evaluate_test_set
from scripts.train_base_ai_44 import prepare_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("base_ai_improvement")
OUT = Path("improvement_outputs")
TRAIN_END = "2025-01-01"
VAL_END = "2025-07-01"
TEST_END = "2026-04-01"
DB_HASH = "188a04241809a41ae9f7bfac3861225666361c6d3cb81e3387ddf4815e6b14e0"
ORIGINAL_MODEL_HASH = "5a6bfe4028e2d82c41d396e013763c825031862a074faf4b0dee2b542eea2a34"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def get_valid_races(conn):
    return {x[0] for x in conn.execute("""
        SELECT r.race_id
        FROM races r JOIN race_results rr ON rr.race_id=r.race_id
        GROUP BY r.race_id
        HAVING COUNT(*)=r.rider_count
          AND SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
          AND COUNT(DISTINCT rr.finish_position)=COUNT(*)
    """)}

def load_payouts(conn):
    exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='race_payouts'"
    ).fetchone()[0]
    if not exists:
        return pd.DataFrame(columns=["race_id"])
    cols = {r[1] for r in conn.execute("PRAGMA table_info(race_payouts)")}
    available = ["race_id"] + [
        c for c in ["nisyatan_combo","nisyatan_payout",
                    "sanrentan_combo","sanrentan_payout"] if c in cols
    ]
    return pd.read_sql_query("SELECT " + ",".join(available) + " FROM race_payouts", conn)

def metric_silent(frame, model, cols):
    # Existing evaluator, original ranking objective; no calibration.
    return evaluate_test_set(frame, model, cols)

def race_predictions(frame, model, cols):
    values = model.predict(frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)).to_numpy()
    work = frame[["race_id","race_date","finish_position","bike_number"]].copy()
    work["score"] = values
    rows = []
    for rid, g in work.groupby("race_id", sort=False):
        pred = g.sort_values("score", ascending=False, kind="stable")
        actual = g.sort_values("finish_position", kind="stable")
        pred_b = tuple(int(x) for x in pred["bike_number"].iloc[:3])
        actual_b = tuple(int(x) for x in actual["bike_number"].iloc[:3])
        rows.append({
            "race_id": rid, "date": str(g["race_date"].iloc[0]),
            "pred1": pred_b[0], "pred2": pred_b[1], "pred3": pred_b[2],
            "actual1": actual_b[0], "actual2": actual_b[1], "actual3": actual_b[2],
            "top1_hit": int(pred_b[0] == actual_b[0]),
            "exacta_hit": int(pred_b[:2] == actual_b[:2]),
            "trifecta_hit": int(pred_b == actual_b),
            "top3_overlap": len(set(pred_b) & set(actual_b)),
        })
    return pd.DataFrame(rows)

def attach_payouts(pred, payouts):
    frame = pred.merge(payouts, on="race_id", how="left", validate="one_to_one")
    for kind, n, hit_col in [
        ("nisyatan", 2, "exacta_hit"),
        ("sanrentan", 3, "trifecta_hit"),
    ]:
        combo_col, payout_col = kind+"_combo", kind+"_payout"
        valid_col, return_col = kind+"_payout_valid", kind+"_return_yen"
        if combo_col not in frame or payout_col not in frame:
            frame[valid_col] = False
            frame[return_col] = np.nan
            continue
        def combo_matches(row):
            text = str(row[combo_col])
            nums = re.findall(r"\d+", text)
            if len(nums) != n or len(set(nums)) != n:
                return False
            try:
                return tuple(int(x) for x in nums) == tuple(
                    int(row["actual"+str(i)]) for i in range(1, n+1)
                )
            except (ValueError, TypeError):
                return False
        valid = frame.apply(combo_matches, axis=1)
        amounts = pd.to_numeric(frame[payout_col], errors="coerce")
        valid &= amounts.between(100, 100000000)
        frame[valid_col] = valid
        frame[return_col] = np.where(
            valid, frame[hit_col].to_numpy() * amounts.to_numpy(), np.nan
        )
    return frame

def roi_probe(frame, prefix):
    valid_col = prefix+"_payout_valid"
    return_col = prefix+"_return_yen"
    part = frame[frame[valid_col]].copy()
    if part.empty:
        return {"n":0,"roi":None,"exact_hits":0,"mean_payout":None}
    raw = part[return_col].astype(float)
    return {
        "n":len(part),
        "roi":float(raw.mean()/100.0),
        "exact_hits":int((raw>0).sum()),
        "return_total_yen":float(raw.sum()),
        "stake_total_yen":int(100*len(part)),
        "mean_payout":float(pd.to_numeric(part[prefix+"_payout"]).mean()),
        "periods": {
            str(k): {
                "n":len(v),
                "roi":float(v[return_col].mean()/100.0),
                "hits":int((v[return_col]>0).sum())
            } for k,v in part.groupby(part["date"].str[:7])
        },
    }

def paired_bootstrap(base, candidate, metric, rng, scale=1.0):
    a=base.set_index("race_id")[metric]
    b=candidate.set_index("race_id")[metric]
    common=a.index.intersection(b.index)
    dif=(b.loc[common].to_numpy(dtype=float)-a.loc[common].to_numpy(dtype=float))
    boots=np.array([rng.choice(dif, size=len(dif), replace=True).mean() for _ in range(3000)])
    return {"n":len(dif), "difference":float(dif.mean()*scale),
            "ci95":[float(x*scale) for x in np.quantile(boots,[.025,.975])]}

def main():
    OUT.mkdir(exist_ok=True)
    original=Path("results/model_lgbm.pkl")
    prior=Path("prior_training/model_lgbm_44_retrained.pkl")
    assert DB_PATH.is_file() and original.is_file() and prior.is_file()
    assert sha256(DB_PATH)==DB_HASH
    assert sha256(original)==ORIGINAL_MODEL_HASH
    conn=sqlite3.connect("file:"+str(DB_PATH)+"?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    valid=get_valid_races(conn)
    entry_gear=pd.read_sql_query(
        "SELECT race_id,rider_id,gear_ratio AS gear_entry FROM entries", conn)
    payouts=load_payouts(conn)
    odds_summary=pd.read_sql_query("""
        SELECT COUNT(*) AS rows_total,
          SUM(CASE WHEN odds IS NOT NULL AND odds>0 THEN 1 ELSE 0 END) AS odds_positive,
          SUM(CASE WHEN gear_ratio IS NOT NULL AND gear_ratio>0 THEN 1 ELSE 0 END) AS gear_positive
        FROM race_results
    """,conn).iloc[0].to_dict()
    conn.close()
    log.info("Raw payout rows=%d, result odds coverage=%s",len(payouts),odds_summary)

    builder=FeatureBuilder()
    cols=builder.feature_names
    assert len(cols)==44
    df=builder.build_dataset(2022,2026)
    df=df[df["race_id"].isin(valid)].copy()
    df["rider_class_num"]=df["class"].map(CLASS_MAP).fillna(6).astype(float)
    df["class_rank_in_race"]=df.groupby("race_id")["rider_class_num"].rank(method="min")
    df["rider_class_x_jiku"]=df["rider_class_num"]*df["line_is_jiku"]
    df=df.merge(entry_gear,on=["race_id","rider_id"],how="left",validate="many_to_one")
    # Exact inputs of the previous successful retraining.
    df["entry_gear_ratio"]=pd.to_numeric(df["gear_entry"],errors="coerce").fillna(DEFAULT_GEAR_RATIO)
    df.drop(columns=["gear_entry"],inplace=True)

    train=df[df["race_date"]<TRAIN_END]
    val=df[(df["race_date"]>=TRAIN_END)&(df["race_date"]<VAL_END)]
    test=df[(df["race_date"]>=VAL_END)&(df["race_date"]<TEST_END)]
    train,xtr,ytr,gtr=prepare_groups(train,cols)
    val,xv,yv,gv=prepare_groups(val,cols)
    test,xt,yt,gt=prepare_groups(test,cols)
    assert len(gtr)==49728 and len(gv)==3168 and len(gt)==3973
    log.info("Chronological races training=%d validation=%d holdout=%d",len(gtr),len(gv),len(gt))

    base=LGBMRanker();base.load(str(prior))
    assert base.feature_names==cols
    base_val=metric_silent(val,base,cols)
    base_test=metric_silent(test,base,cols)
    log.info("Existing retrained 44 model validation=%s",json.dumps(base_val))
    log.info("Existing retrained 44 model holdout=%s",json.dumps(base_test))

    variants={
        "top1_early_stopping":{"ndcg_eval_at":[1,3]},
        "strong_regularization":{"num_leaves":23,"min_data_in_leaf":160,
                                 "lambda_l2":4.0},
        "higher_capacity":{"num_leaves":63,"min_data_in_leaf":48,
                           "lambda_l2":2.0},
        "less_subsampling":{"feature_fraction":.95,"bagging_fraction":.95,
                            "bagging_freq":1},
    }
    evaluations=[{"name":"existing_44","validation":base_val,"model":base}]
    for name,updates in variants.items():
        params=dict(LGBM_PARAMS);params.update(updates)
        model=LGBMRanker(params=params)
        model.train(xtr,ytr,gtr,xv,yv,gv)
        vm=metric_silent(val,model,cols)
        evaluations.append({"name":name,"validation":vm,"model":model})
        log.info("Candidate %s validation=%s trees=%d",name,json.dumps(vm),model.model.num_trees())

    # Select before looking at any candidate's holdout result.
    chosen=max(evaluations,key=lambda e:(
        e["validation"]["ndcg_at_3"],
        e["validation"]["top1_accuracy"]))
    log.info("VALIDATION-SELECTED %s",chosen["name"])
    improvement=chosen["model"]
    chosen_test=metric_silent(test,improvement,cols)
    log.info("Selected holdout %s",json.dumps(chosen_test))

    baseline_pred=race_predictions(test,base,cols)
    chosen_pred=race_predictions(test,improvement,cols)
    baseline_pay=attach_payouts(baseline_pred,payouts)
    chosen_pay=attach_payouts(chosen_pred,payouts)
    assert (baseline_pay["race_id"]==chosen_pay["race_id"]).all()
    # No bet recommendation or stake optimization: same hypothetical 100-yen
    # exact-order top-2/top-3 ticket in EVERY evaluable race, for diagnostics only.
    roi_base={k:roi_probe(baseline_pay,k) for k in ["nisyatan","sanrentan"]}
    roi_new={k:roi_probe(chosen_pay,k) for k in ["nisyatan","sanrentan"]}
    rng=np.random.default_rng(20261023)
    pair={
        "top1_accuracy_pp":paired_bootstrap(baseline_pred,chosen_pred,"top1_hit",rng,100),
        "top3_overlap_pp":paired_bootstrap(baseline_pred,chosen_pred,"top3_overlap",rng,100/3),
    }
    for kind in ["nisyatan","sanrentan"]:
        valid=baseline_pay[kind+"_payout_valid"] & chosen_pay[kind+"_payout_valid"]
        if valid.sum()>50:
            b=baseline_pay.loc[valid,["race_id",kind+"_return_yen"]]
            c=chosen_pay.loc[valid,["race_id",kind+"_return_yen"]]
            pair[kind+"_roi_pp"]=paired_bootstrap(b,c,kind+"_return_yen",rng,1.0)
            # Return amount is yen per 100-yen stake; mean difference in yen
            # numerically equals ROI percentage-point difference.

    chosen_path=OUT/"model_lgbm_44_validation_selected.pkl"
    improvement.save(str(chosen_path))
    baseline_pay.to_csv(OUT/"holdout_existing_predictions.csv",index=False)
    chosen_pay.to_csv(OUT/"holdout_selected_predictions.csv",index=False)
    scoreboard=[{"candidate":x["name"],**x["validation"]} for x in evaluations]
    pd.DataFrame(scoreboard).to_csv(OUT/"validation_candidates.csv",index=False)
    report={
        "purpose":"Original 44-feature LightGBM LambdaRank improvement; no auxiliary AI",
        "db_sha256":DB_HASH,
        "original_model_sha256":ORIGINAL_MODEL_HASH,
        "existing_model_sha256":sha256(prior),
        "selected_model_sha256":sha256(chosen_path),
        "split":{"train_races":len(gtr),"validation_races":len(gv),"holdout_races":len(gt)},
        "selection_metric":"validation NDCG@3, validation Top1 tiebreak",
        "candidate_validation":scoreboard,
        "selected_candidate":chosen["name"],
        "existing_holdout":base_test,
        "selected_holdout":chosen_test,
        "paired_holdout":pair,
        "odds_data_coverage":odds_summary,
        "payout_rows":len(payouts),
        "payout_probe_existing":roi_base,
        "payout_probe_selected":roi_new,
        "payout_probe_caveat":"Uniform one 100-yen ranked-order ticket per evaluable race is a common retrospective diagnostic, NOT a purchase strategy. Full-market EV or optimized ROI is not tested.",
        "limitations":[
            "Pre-existing holdout was previously inspected for baseline capability.",
            "Exact-order return depends on the completeness and integrity of historical payout records.",
            "The ranking model outputs scores, not calibrated event probabilities.",
            "One chronological holdout cannot prove future ROI improvement.",
            "The three historical entry-rate features remain unobserved in the training period."
        ],
    }
    (OUT/"improvement_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    # Compact decisive summary visible in GitHub Actions logs.
    summary={k:report[k] for k in [
        "selected_candidate","existing_holdout","selected_holdout",
        "paired_holdout","odds_data_coverage","payout_rows"]}
    summary["payout_probe_existing"]={
        k:{x:v[x] for x in ["n","roi","exact_hits"]}
        for k,v in roi_base.items()}
    summary["payout_probe_selected"]={
        k:{x:v[x] for x in ["n","roi","exact_hits"]}
        for k,v in roi_new.items()}
    log.info("IMPROVEMENT SUMMARY %s",json.dumps(summary,ensure_ascii=False))
    log.info("IMPROVEMENT COMPLETE %s",OUT)

if __name__=="__main__":
    main()
