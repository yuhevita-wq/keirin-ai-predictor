#!/usr/bin/env python3
"""Out-of-sample capability audit for the existing 44-feature LambdaRank.

No model is trained or modified. No betting logic is added.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

from config import CLASS_MAP, DB_PATH, DEFAULT_GEAR_RATIO
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("base_ai_capability")
OUT = Path("capability_outputs")
HOLDOUT_START = "2025-07-01"
HOLDOUT_END = "2026-04-01"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(hits, total):
    if total == 0:
        return [None, None]
    z = 1.95996398454
    p = hits / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    span = z * ((p*(1-p)/total + z*z/(4*total*total))**0.5) / den
    return [center-span, center+span]


def line_shape(group):
    labels = group["line_group"].fillna("").astype(str)
    present = labels[labels.str.strip() != ""]
    sizes = list(present.value_counts().astype(int).values)
    sizes.extend([1] * int((labels.str.strip() == "").sum()))
    return "-".join(map(str, sorted(sizes, reverse=True))) if sizes else "unknown"


def class_group(group):
    vals = set(group["class"].dropna().astype(str))
    if vals and vals <= {"SS", "S1", "S2"}:
        return "S"
    if vals and vals <= {"A1", "A2"}:
        return "A1-A2"
    if vals and vals <= {"A3"}:
        return "A3"
    return "mixed_or_missing"


def period_of(date):
    if date < "2025-10-01":
        return "2025Q3"
    if date < "2026-01-01":
        return "2025Q4"
    return "2026Q1"


def main():
    OUT.mkdir(exist_ok=True)
    baseline_path = Path("results/model_lgbm.pkl")
    retrained_path = Path("previous_training/model_lgbm_44_retrained.pkl")
    assert all(p.is_file() for p in [DB_PATH, baseline_path, retrained_path])
    assert sha256(DB_PATH) == "188a04241809a41ae9f7bfac3861225666361c6d3cb81e3387ddf4815e6b14e0"
    assert sha256(baseline_path) == "5a6bfe4028e2d82c41d396e013763c825031862a074faf4b0dee2b542eea2a34"

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    valid_races = {
        r[0] for r in conn.execute("""
            SELECT r.race_id
            FROM races r JOIN race_results rr ON r.race_id=rr.race_id
            GROUP BY r.race_id
            HAVING COUNT(*)=r.rider_count
              AND SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
              AND COUNT(DISTINCT rr.finish_position)=COUNT(*)
        """)
    }
    gear = pd.read_sql_query(
        "SELECT race_id,rider_id,gear_ratio AS gear_from_entry FROM entries", conn
    )
    meta = pd.read_sql_query(
        "SELECT race_id,grade,rider_count,velodrome FROM races", conn
    )
    conn.close()

    builder = FeatureBuilder()
    cols = builder.feature_names
    assert len(cols) == 44
    df = builder.build_dataset(2022, 2026)
    df = df[
        df["race_id"].isin(valid_races)
        & (df["race_date"] >= HOLDOUT_START)
        & (df["race_date"] < HOLDOUT_END)
    ].copy()
    df["rider_class_num"] = df["class"].map(CLASS_MAP).fillna(6).astype(float)
    df["class_rank_in_race"] = df.groupby("race_id")["rider_class_num"].rank(method="min")
    df["rider_class_x_jiku"] = df["rider_class_num"] * df["line_is_jiku"]
    df = df.merge(gear, on=["race_id", "rider_id"], how="left", validate="many_to_one")
    df["entry_gear_ratio"] = pd.to_numeric(
        df["gear_from_entry"], errors="coerce"
    ).fillna(DEFAULT_GEAR_RATIO)
    df.drop(columns=["gear_from_entry"], inplace=True)
    df = df.merge(meta, on="race_id", how="left", validate="many_to_one")
    df = df.sort_values(["race_date", "race_id", "entry_bike_number"], kind="stable")
    assert df["race_id"].nunique() == 3973
    assert len(df) == 28030

    old = LGBMRanker()
    old.load(str(baseline_path))
    new = LGBMRanker()
    new.load(str(retrained_path))
    assert old.feature_names == cols == new.feature_names
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["score_old"] = old.predict(x).values
    df["score_new"] = new.predict(x).values

    rows = []
    for race_id, group in df.groupby("race_id", sort=False):
        new_order = group.sort_values("score_new", ascending=False, kind="stable")
        old_order = group.sort_values("score_old", ascending=False, kind="stable")
        actual = group.loc[group["finish_position"] == 1, "rider_id"].iloc[0]
        actual_top3 = set(group.loc[group["finish_position"] <= 3, "rider_id"])
        new_ranking = new_order["rider_id"].tolist()
        old_ranking = old_order["rider_id"].tolist()
        comp_order = group.sort_values("rider_competition_score", ascending=False, kind="stable")
        win_order = group.sort_values("rider_win_rate_all", ascending=False, kind="stable")
        row = group.iloc[0]
        gap = float(new_order["score_new"].iloc[0] - new_order["score_new"].iloc[1])
        rows.append({
            "race_id": race_id,
            "date": str(row["race_date"]),
            "period": period_of(str(row["race_date"])),
            "grade": str(row["grade"]),
            "rider_count": int(len(group)),
            "race_class": class_group(group),
            "line_shape": line_shape(group),
            "median_history": float(group["rider_race_count"].median()),
            "score_gap": gap,
            "new_hit": int(new_ranking[0] == actual),
            "old_hit": int(old_ranking[0] == actual),
            "comp_score_hit": int(comp_order.iloc[0]["rider_id"] == actual),
            "past_win_rate_hit": int(win_order.iloc[0]["rider_id"] == actual),
            "new_winner_rank": int(new_ranking.index(actual) + 1),
            "new_top3_overlap": int(len(set(new_ranking[:3]) & actual_top3)),
            "old_top3_overlap": int(len(set(old_ranking[:3]) & actual_top3)),
        })
    races = pd.DataFrame(rows)
    assert int(races["new_hit"].sum()) == 1741
    assert int(races["old_hit"].sum()) == 1707
    races["gap_quintile"] = pd.qcut(
        races["score_gap"].rank(method="first"), 5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"]
    ).astype(str)
    races["history_band"] = pd.cut(
        races["median_history"], bins=[-1, 9, 49, float("inf")],
        labels=["under_10", "10_to_49", "50_plus"]
    ).astype(str)

    n = len(races)
    new_hits = int(races["new_hit"].sum())
    old_hits = int(races["old_hit"].sum())
    new_only = int(((races["new_hit"] == 1) & (races["old_hit"] == 0)).sum())
    old_only = int(((races["new_hit"] == 0) & (races["old_hit"] == 1)).sum())
    discordant = new_only + old_only
    mcnemar_p = float(binomtest(new_only, discordant, 0.5).pvalue) if discordant else 1.0

    rng = np.random.default_rng(42)
    paired_diffs = (races["new_hit"] - races["old_hit"]).to_numpy()
    boots = np.empty(5000)
    for i in range(5000):
        boots[i] = rng.choice(paired_diffs, size=n, replace=True).mean()
    diff_ci = np.quantile(boots, [0.025, 0.975]).tolist()

    total_rate = new_hits / n
    periods = races.groupby("period").agg(
        n=("new_hit", "size"), hits=("new_hit", "sum")
    )
    period_rates = (periods["hits"] / periods["n"]).to_dict()
    axes = ["period", "grade", "rider_count", "race_class",
            "line_shape", "history_band", "gap_quintile"]
    subgroup_rows = []
    for axis in axes:
        for label, part in races.groupby(axis, dropna=False):
            gn = len(part)
            if gn < 80:
                continue
            hits = int(part["new_hit"].sum())
            rest_hits = new_hits - hits
            rest_n = n - gn
            pvalue = float(fisher_exact(
                [[hits, gn-hits], [rest_hits, rest_n-rest_hits]]
            ).pvalue) if rest_n else 1.0
            consistency = []
            for period, ppart in part.groupby("period"):
                if len(ppart) >= 30:
                    sign = np.sign(ppart["new_hit"].mean() - period_rates[period])
                    consistency.append(int(sign))
            stable_direction = (
                len(consistency) >= 2
                and all(sign > 0 for sign in consistency)
            ) or (
                len(consistency) >= 2
                and all(sign < 0 for sign in consistency)
            )
            subgroup_rows.append({
                "axis": axis, "group": str(label), "n": gn,
                "new_hits": hits, "new_top1_rate": hits/gn,
                "old_top1_rate": float(part["old_hit"].mean()),
                "comp_score_top1_rate": float(part["comp_score_hit"].mean()),
                "past_win_rate_top1_rate": float(part["past_win_rate_hit"].mean()),
                "new_winner_top3_rate": float((part["new_winner_rank"] <= 3).mean()),
                "new_top3_overlap_rate": float(part["new_top3_overlap"].sum()/(3*gn)),
                "vs_overall_pp": 100*(hits/gn-total_rate),
                "wilson_low": wilson(hits,gn)[0],
                "wilson_high": wilson(hits,gn)[1],
                "pvalue": pvalue,
                "stable_direction": bool(stable_direction),
                "periods_tested": len(consistency)
            })
    groups = pd.DataFrame(subgroup_rows)
    m = len(groups)
    ordered = groups["pvalue"].sort_values().index.tolist()
    q_values = np.ones(m)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = ordered[rank-1]
        running = min(running, float(groups.loc[idx,"pvalue"]) * m / rank)
        q_values[idx] = running
    groups["fdr_qvalue"] = q_values
    groups["verdict"] = "not_established"
    confirmed = (groups["fdr_qvalue"] < 0.05) & groups["stable_direction"]
    groups.loc[confirmed & (groups["vs_overall_pp"] > 0), "verdict"] = "higher_signal"
    groups.loc[confirmed & (groups["vs_overall_pp"] < 0), "verdict"] = "lower_signal"
    groups = groups.sort_values(["axis","new_top1_rate"], ascending=[True,False])

    gain = new.feature_importance("gain")
    split = new.feature_importance("split")
    importance = pd.DataFrame({"feature": cols, "gain": gain, "split": split})
    importance["gain_share"] = importance["gain"] / max(1.0,float(importance["gain"].sum()))
    importance.sort_values("gain", ascending=False).to_csv(
        OUT/"feature_importance.csv", index=False
    )
    races.to_csv(OUT/"race_predictions.csv",index=False)
    groups.to_csv(OUT/"subgroup_results.csv",index=False)
    overall = {
        "model": "original 44-feature LightGBM LambdaRank, retrained without auxiliary model",
        "holdout_period": [HOLDOUT_START, HOLDOUT_END],
        "races": n,
        "rider_rows": len(df),
        "new_top1_hits": new_hits,
        "new_top1_rate": total_rate,
        "new_top1_wilson95": wilson(new_hits,n),
        "old_top1_hits": old_hits,
        "old_top1_rate": old_hits/n,
        "top1_paired_difference_pp": 100*(new_hits-old_hits)/n,
        "paired_bootstrap_difference_95pp": [100*x for x in diff_ci],
        "mcnemar_new_only": new_only,
        "mcnemar_old_only": old_only,
        "mcnemar_exact_p": mcnemar_p,
        "new_top3_winner_coverage": float((races["new_winner_rank"]<=3).mean()),
        "winner_rank_distribution": {
            "rank1": int((races["new_winner_rank"]==1).sum()),
            "rank2": int((races["new_winner_rank"]==2).sum()),
            "rank3": int((races["new_winner_rank"]==3).sum()),
            "rank4plus": int((races["new_winner_rank"]>=4).sum()),
        },
        "random_top1_expectation": float(np.mean(1/races["rider_count"])),
        "competition_score_heuristic_rate": float(races["comp_score_hit"].mean()),
        "past_win_rate_heuristic_rate": float(races["past_win_rate_hit"].mean()),
        "zero_gain_features": importance.loc[importance["gain"]==0,"feature"].tolist(),
        "subgroup_testing": "Fisher exact tests, Benjamini-Hochberg FDR, same-direction check across >=2 quarters",
        "limitations": [
            "One fixed chronological holdout does not prove an absolute learning ceiling.",
            "The released old model training cutoff is unknown, so old-vs-new comparison may not be independent.",
            "LambdaRank scores are not calibrated probabilities.",
            "Subgroup differences are associations, not causes."
        ],
    }
    with open(OUT/"capability_report.json","w",encoding="utf-8") as stream:
        json.dump(overall,stream,ensure_ascii=False,indent=2)
    log.info("Overall %s",json.dumps(overall,ensure_ascii=False))
    log.info("Confirmed subgroups %s",groups[groups["verdict"]!="not_established"][
        ["axis","group","n","new_top1_rate","vs_overall_pp","fdr_qvalue","verdict"]
    ].to_json(orient="records",force_ascii=False))
    log.info("Capability audit complete: %s",OUT)


if __name__ == "__main__":
    main()
