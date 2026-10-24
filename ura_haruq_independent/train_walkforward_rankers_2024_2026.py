#!/usr/bin/env python3
"""Research-only chronological, separately fitted HaruQ-like rankers.

Each forecast year has its *own* LightGBM LambdaRank fitted only on older
calendar years. Preserves the original 44 features and training parameters.
Frozen flagship, V3/V4 geometries, live decisions and purchase rules untouched.

2024: train Jan-Sep 2023, validate Oct-Dec 2023, predict 2024
2025: train 2023-Sep 2024, validate Oct-Dec 2024, predict 2025
2026: train 2023-Sep 2025, validate Oct-Dec 2025, predict Jan-Sep 2026
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import DB_PATH, LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING_ROUNDS
from models.lgbm_ranker import LGBMRanker
from scripts.reproduce_haruqube_fresh import dataset

OUT = ROOT / "ura_haruq_independent" / "diagnostics" / "walkforward_2024_2026"
FOLDS = (
    ("2024", "2023-10-01", "2024-01-01", "2025-01-01"),
    ("2025", "2024-10-01", "2025-01-01", "2026-01-01"),
    ("2026", "2025-10-01", "2026-01-01", "2026-10-01"),
)
FEATURE_COUNT = 44
COLUMN_NAMES = (
    "race_id", "race_date", "forecast_year", "grade", "field_size",
    "predicted_bikes_by_rank", "predicted_scores_by_rank", "score_gap",
    "low_quarter", "actual_top3", "predicted_top1_won",
    "predicted_top3_overlap", "trifecta_combo", "trifecta_payout",
)


def quarter(gap: float) -> str:
    if gap < 0.0 or not math.isfinite(gap):
        raise ValueError("Nonfinite or negative score gap")
    for hi, s in (
        (0.25, "LOW_0.00_0.25"), (0.50, "LOW_0.25_0.50"),
        (0.75, "LOW_0.50_0.75"), (1.00, "LOW_0.75_1.00"),
    ):
        if gap < hi:
            return s
    return "NOT_LOW"


def sorted_races(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["race_date", "race_id", "bike_number"], kind="stable").reset_index(drop=True)


def split_check(df: pd.DataFrame, year: str, val_start: str,
                pred_start: str, pred_end: str):
    train = df[(df.race_date >= "2023-01-01") & (df.race_date < val_start)].copy()
    valid = df[(df.race_date >= val_start) & (df.race_date < pred_start)].copy()
    predict = df[(df.race_date >= pred_start) & (df.race_date < pred_end)].copy()
    for piece in (train, valid, predict):
        if len(piece) == 0 or piece.race_id.nunique() < 1000:
            raise RuntimeError(f"INSUFFICIENT_RACES_FOR_FOLD_{year}")
    assert max(train.race_date) < val_start
    assert min(valid.race_date) >= val_start and max(valid.race_date) < pred_start
    assert min(predict.race_date) >= pred_start and max(predict.race_date) < pred_end
    ids = [set(x.race_id) for x in (train, valid, predict)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for part in (train, valid, predict):
        race_sizes = part.groupby("race_id", sort=False).size()
        assert race_sizes.between(7, 9).all()
        assert (part.groupby("race_id", sort=False).race_date.nunique() == 1).all()
    return train, valid, predict


def to_xy(df: pd.DataFrame, features: list[str]):
    ordered = sorted_races(df)
    x = ordered[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(ordered.finish_position, errors="raise")
    groups = ordered.groupby("race_id", sort=False).size().tolist()
    assert sum(groups) == len(ordered)
    assert not x.isna().any().any()
    return ordered, x, y, groups


def self_test():
    assert len(FOLDS) == 3
    assert quarter(0) == "LOW_0.00_0.25"
    assert quarter(0.249999) == "LOW_0.00_0.25"
    assert quarter(0.25) == "LOW_0.25_0.50"
    assert quarter(0.50) == "LOW_0.50_0.75"
    assert quarter(0.75) == "LOW_0.75_1.00"
    assert quarter(1.0) == "NOT_LOW"
    assert len({x[0] for x in FOLDS}) == 3
    for year, val, start, end in FOLDS:
        assert "2023-01-01" < val < start < end
        assert year == start[:4]
    print("WALKFORWARD_PROTOCOL_SELF_TEST_OK", flush=True)


def load_all():
    if not DB_PATH.exists():
        raise RuntimeError("DECRYPTED_HISTORICAL_DB_NOT_FOUND")
    with sqlite3.connect(DB_PATH) as conn:
        raw_payouts = pd.read_sql_query(
            "SELECT race_id, sanrentan_combo, sanrentan_payout FROM race_payouts", conn)
        names = pd.read_sql_query(
            "SELECT race_id, race_name, rider_count FROM races", conn)
    if raw_payouts.race_id.duplicated().any() or names.race_id.duplicated().any():
        raise RuntimeError("DUPLICATE_RACE_METADATA")
    df, feature_cols, quality_valid = dataset()
    if len(feature_cols) != FEATURE_COUNT or len(set(feature_cols)) != FEATURE_COUNT:
        raise RuntimeError("ORIGINAL_44_FEATURE_CONTRACT_CHANGED")
    if len(df) == 0 or "class" not in df or "grade" not in df:
        raise RuntimeError("EMPTY_OR_INCOMPATIBLE_HISTORY")
    df = df.merge(names, on="race_id", how="left", validate="many_to_one")
    df = df.merge(raw_payouts, on="race_id", how="left", validate="many_to_one")
    df = df[(df.race_date >= "2023-01-01") & (df.race_date < "2026-10-01")].copy()
    rows_before = df.race_id.nunique()
    groups = df.groupby("race_id", sort=False)
    counts = groups.size()
    exclude_not_7_9 = set(counts[~counts.between(7, 9)].index)
    girls = (
        df["class"].fillna("").astype(str).str.match(r"(?i)^L\d*") |
        df["race_name"].fillna("").astype(str).str.contains(
            r"ガールズ|ガ決|ガ般|Ｌ級|L級", regex=True)
    )
    exclude_girls = set(df.loc[girls, "race_id"])
    exclude_bad_bikes = set(
        df.loc[df.groupby("race_id")["bike_number"].transform("nunique") !=
               df.groupby("race_id")["bike_number"].transform("size"), "race_id"]
    )
    remove = exclude_not_7_9 | exclude_girls | exclude_bad_bikes
    df = df.loc[~df.race_id.isin(remove)].copy()
    if df.race_id.nunique() < 10000:
        raise RuntimeError("INSUFFICIENT_ELIGIBLE_HISTORY")
    if df[feature_cols].shape[1] != FEATURE_COUNT:
        raise RuntimeError("FEATURE_COLUMN_ERROR")
    return sorted_races(df), list(feature_cols), {
        "quality_valid_races_in_db": quality_valid,
        "races_before_7to9_girls_filter": rows_before,
        "excluded_wrong_field_size": len(exclude_not_7_9),
        "excluded_girls": len(exclude_girls),
        "excluded_duplicate_bike": len(exclude_bad_bikes),
        "eligible_races": int(df.race_id.nunique()),
    }


def race_predictions(frame: pd.DataFrame, scores: np.ndarray, year: str):
    if len(frame) != len(scores) or not np.isfinite(scores).all():
        raise RuntimeError("INVALID_PREDICTION_SCORES")
    p = frame.copy()
    p["predicted_rank_score"] = scores
    emitted = []
    year_summary = defaultdict(lambda: {
        "races": 0, "top1_hits": 0, "top3_overlap_count": 0,
        "verified_payouts": 0,
    })
    for rid, group in p.groupby("race_id", sort=False):
        predicted = group.sort_values(
            ["predicted_rank_score", "bike_number"], ascending=[False, True],
            kind="stable")
        finished = group.sort_values("finish_position", kind="stable")
        bikes = [int(x) for x in predicted.bike_number]
        actual = [int(x) for x in finished.head(3).bike_number]
        if len(set(bikes)) != len(bikes) or len(set(actual)) != 3:
            raise RuntimeError(f"BAD_RACE_ORDER_{rid}")
        gap = float(predicted.iloc[0].predicted_rank_score -
                    predicted.iloc[1].predicted_rank_score)
        score_band = quarter(max(0., gap))
        top1_hit = int(bikes[0] == actual[0])
        overlap = len(set(bikes[:3]) & set(actual))
        meta = predicted.iloc[0]
        combo = str(meta.sanrentan_combo) if pd.notna(meta.sanrentan_combo) else ""
        payout = meta.sanrentan_payout
        nums = [int(i) for i in re.findall(r"\d+", combo)]
        verified = bool(len(nums) == 3 and nums == actual and
                        pd.notna(payout) and float(payout) >= 100)
        row = {
            "race_id": str(rid), "race_date": str(meta.race_date),
            "forecast_year": year, "grade": str(meta.grade),
            "field_size": len(bikes),
            "predicted_bikes_by_rank": "-".join(map(str, bikes)),
            "predicted_scores_by_rank": ";".join(
                f"{float(v):.10f}" for v in predicted.predicted_rank_score),
            "score_gap": f"{gap:.10f}",
            "low_quarter": score_band,
            "actual_top3": "-".join(map(str, actual)),
            "predicted_top1_won": top1_hit,
            "predicted_top3_overlap": overlap,
            "trifecta_combo": "-".join(map(str, actual)) if verified else "",
            "trifecta_payout": int(float(payout)) if verified else "",
        }
        emitted.append(row)
        s = year_summary[score_band]
        s["races"] += 1
        s["top1_hits"] += top1_hit
        s["top3_overlap_count"] += overlap
        s["verified_payouts"] += int(verified)
    metrics = {}
    for band, s in sorted(year_summary.items()):
        ct = s["races"]
        metrics[band] = {
            **s, "top1_accuracy_percent": round(100*s["top1_hits"]/ct, 4),
            "top3_overlap_percent": round(
                100*s["top3_overlap_count"]/(3*ct), 4),
        }
    return emitted, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    frame, features, quality = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = OUT / "independent_models"
    model_dir.mkdir(exist_ok=True)
    raw_rows = []
    report = {
        "system": "URA_WALKFORWARD_INDEPENDENT_RANKERS_V1",
        "research_only": True,
        "original_flagship_modified": False,
        "existing_v3_v4_types_modified": False,
        "model_family": "LightGBM LambdaRank original 44 features",
        "features": features,
        "hyperparameters": LGBM_PARAMS,
        "boost_round_max": LGBM_NUM_BOOST_ROUND,
        "early_stopping_rounds": LGBM_EARLY_STOPPING_ROUNDS,
        "feature_quality": quality,
        "folds": {},
        "warnings": [
            "Chronologically disjoint labels/models; historical model hyperparameters "
            "were already tuned upstream, so not a newly untouched model-design study.",
            "Pre-race availability/provenance of all original feature fields "
            "is NOT proven. Historical line and class feature as-of audit required.",
            "Historical payoff verification uses after-race data ONLY for scoring, "
            "never for train/prediction features.",
            "A given fold's raw LightGBM score gaps may differ in scale across "
            "newly fitted models; do not assume the old flagship LOW thresholds "
            "have stable coverage without empirical calibration.",
            "2026 outcome labels are already known in other research contexts; "
            "this is a strictly chronological but not globally untouched test.",
        ],
    }
    for year, val_start, pred_start, pred_end in FOLDS:
        train, valid, target = split_check(
            frame, year, val_start, pred_start, pred_end)
        train, tx, ty, tg = to_xy(train, features)
        valid, vx, vy, vg = to_xy(valid, features)
        target, px, _, pg = to_xy(target, features)
        assert not set(train.race_id) & set(target.race_id)
        assert not set(valid.race_id) & set(target.race_id)
        print("CHRONOLOGICAL_SPLIT",
              year, len(tg), len(vg), len(pg), flush=True)
        model = LGBMRanker()
        model.train(tx, ty, tg, vx, vy, vg)
        model_path = model_dir / f"ranker_predict_{year}.pkl"
        model.save(str(model_path))
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        prediction = np.asarray(model.predict(px), dtype=np.float64)
        emitted, metrics = race_predictions(target, prediction, year)
        raw_rows.extend(emitted)
        report["folds"][year] = {
            "train_dates": [str(train.race_date.min()), str(train.race_date.max())],
            "validation_dates": [str(valid.race_date.min()), str(valid.race_date.max())],
            "forecast_dates": [str(target.race_date.min()), str(target.race_date.max())],
            "train_races": len(tg),
            "validation_races": len(vg),
            "forecast_races": len(pg),
            "num_trees": model.model.num_trees(),
            "best_iteration": model.model.best_iteration,
            "model_artifact": model_path.name,
            "model_sha256": digest,
            "yearly_top1_accuracy_percent": round(
                100 * sum(int(v["predicted_top1_won"]) for v in emitted) /
                len(emitted), 5),
            "yearly_top3_overlap_percent": round(
                100 * sum(int(v["predicted_top3_overlap"]) for v in emitted) /
                (3 * len(emitted)), 5),
            "score_quarters": metrics,
        }
        print("YEAR_FORECAST_COMPLETE", json.dumps({
            "year": year,
            "races": len(emitted),
            "top1_percent": report["folds"][year]["yearly_top1_accuracy_percent"],
            "low_quarters": {k: v["races"] for k,v in metrics.items()},
        }, ensure_ascii=False), flush=True)
    if len({r["race_id"] for r in raw_rows}) != len(raw_rows):
        raise RuntimeError("DUPLICATE_FORECAST_RACE_ID")
    predictions = OUT / "race_level_chronological_predictions_2024_2026.csv.gz"
    with gzip.open(predictions, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_NAMES)
        writer.writeheader()
        writer.writerows(raw_rows)
    with (OUT / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("ALL_INDEPENDENT_WALKFORWARD_FORECASTS_COMPLETE",
          len(raw_rows), flush=True)


if __name__ == "__main__":
    main()
