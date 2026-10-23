#!/usr/bin/env python3
"""Audit and retrain the original 44-feature LightGBM LambdaRank only.

Never replaces the released model or database.
"""
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import CLASS_MAP, DB_PATH, DEFAULT_GEAR_RATIO
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from models.trainer import evaluate_test_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("base_ai_44")
OUT = Path("training_outputs")
TRAIN_END = "2026-04-01"
VAL_END = "2026-05-01"
TEST_END = "2026-05-19"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_groups(frame, feature_names):
    ordered = frame.sort_values(
        ["race_date", "race_id", "entry_bike_number"], kind="stable"
    ).reset_index(drop=True)
    sizes = ordered.groupby("race_id", sort=False).size().tolist()
    assert sum(sizes) == len(ordered)
    for rid, group in ordered.groupby("race_id", sort=False):
        assert len(group) >= 5, rid
    features = ordered.loc[:, feature_names].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    return ordered, features, ordered["finish_position"], sizes


def main():
    OUT.mkdir(exist_ok=True)
    baseline_path = Path("results/model_lgbm.pkl")
    assert DB_PATH.is_file() and baseline_path.is_file()
    db_digest = sha256(DB_PATH)
    model_digest = sha256(baseline_path)
    assert db_digest == "188a04241809a41ae9f7bfac3861225666361c6d3cb81e3387ddf4815e6b14e0"
    assert model_digest == "5a6bfe4028e2d82c41d396e013763c825031862a074faf4b0dee2b542eea2a34"

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    valid_races = {
        row[0] for row in conn.execute("""
            SELECT r.race_id
            FROM races r JOIN race_results rr ON rr.race_id=r.race_id
            GROUP BY r.race_id
            HAVING COUNT(*)=r.rider_count
               AND SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
               AND COUNT(DISTINCT rr.finish_position)=COUNT(*)
        """)
    }
    gear = pd.read_sql_query(
        "SELECT race_id,rider_id,gear_ratio AS entry_gear_ratio_source FROM entries",
        conn,
    )
    entry_coverage = pd.read_sql_query("""
        SELECT substr(r.date,1,7) AS month, count(DISTINCT e.race_id) AS races
        FROM entries e JOIN races r ON e.race_id=r.race_id
        GROUP BY 1 ORDER BY 1
    """, conn)
    conn.close()
    log.info("Quality-valid races: %s", len(valid_races))

    builder = FeatureBuilder()
    feature_names = builder.feature_names
    assert len(feature_names) == 44 and len(set(feature_names)) == 44
    df = builder.build_dataset(2022, 2026)
    initial_rows = len(df)
    df = df[df["race_id"].isin(valid_races)].copy()
    assert len(df) and df["race_id"].nunique() > 1000

    # Preserve the original feature definitions and count. Fix future leakage:
    # riders.class is the currently recorded class, not class at race time.
    df["rider_class_num"] = df["class"].map(CLASS_MAP).fillna(6).astype(float)
    df["class_rank_in_race"] = df.groupby("race_id")[
        "rider_class_num"
    ].rank(method="min")
    df["rider_class_x_jiku"] = (
        df["rider_class_num"] * df["line_is_jiku"]
    )

    # Use the pre-race entry gear when present, not post-race NULL values.
    df = df.merge(
        gear, on=["race_id", "rider_id"], how="left", validate="many_to_one"
    )
    df["entry_gear_ratio"] = pd.to_numeric(
        df["entry_gear_ratio_source"], errors="coerce"
    ).fillna(DEFAULT_GEAR_RATIO)
    df.drop(columns=["entry_gear_ratio_source"], inplace=True)

    train = df[df["race_date"] < TRAIN_END]
    val = df[(df["race_date"] >= TRAIN_END) & (df["race_date"] < VAL_END)]
    test = df[(df["race_date"] >= VAL_END) & (df["race_date"] < TEST_END)]
    assert min(len(train), len(val), len(test)) > 1000
    assert train["race_date"].max() < val["race_date"].min()
    assert val["race_date"].max() < test["race_date"].min()

    train, x_train, y_train, g_train = prepare_groups(train, feature_names)
    val, x_val, y_val, g_val = prepare_groups(val, feature_names)
    test, x_test, y_test, g_test = prepare_groups(test, feature_names)
    log.info(
        "Rows train=%s val=%s test=%s; races=%s/%s/%s",
        len(train), len(val), len(test), len(g_train), len(g_val), len(g_test)
    )

    # Read-only descriptive benchmark; its own training cutoff is unknown.
    baseline = LGBMRanker()
    baseline.load(str(baseline_path))
    old_features = list(baseline.feature_names)
    old_feature_match = old_features == feature_names
    log.info("Released model feature count=%s exact_match=%s trees=%s",
             len(old_features), old_feature_match, baseline.model.num_trees())
    old_test = evaluate_test_set(test, baseline, old_features)

    # Original class, original objective, original tuned parameters, 44 inputs.
    retrained = LGBMRanker()
    retrained.train(x_train, y_train, g_train, x_val, y_val, g_val)
    new_val = evaluate_test_set(val, retrained, feature_names)
    new_test = evaluate_test_set(test, retrained, feature_names)

    model_out = OUT / "model_lgbm_44_retrained.pkl"
    retrained.save(str(model_out))
    importance = retrained.feature_importance("gain").merge(
        retrained.feature_importance("split"),
        on="feature", suffixes=("_gain", "_split")
    )
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    entry_coverage.to_csv(OUT / "entry_coverage.csv", index=False)
    report = {
        "architecture": "original LightGBM LambdaRank",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "original_model_sha256": model_digest,
        "original_db_sha256": db_digest,
        "released_model_feature_count": len(old_features),
        "released_model_feature_names_exact_match": old_feature_match,
        "original_training_period_unknown": True,
        "splits": {
            "train": {"end_exclusive": TRAIN_END, "races": len(g_train), "rows": len(train)},
            "validation": {"start": TRAIN_END, "end_exclusive": VAL_END, "races": len(g_val), "rows": len(val)},
            "holdout": {"start": VAL_END, "end_exclusive": TEST_END, "races": len(g_test), "rows": len(test)}
        },
        "quality": {
            "valid_races_before_builder_filters": len(valid_races),
            "rows_before_quality_filter": initial_rows,
            "rows_after_quality_filter": len(df)
        },
        "released_model_holdout_reference_not_independent": old_test,
        "retrained_validation": new_val,
        "retrained_holdout": new_test,
        "retrained_model_sha256": sha256(model_out),
        "future_data_guard": "historical class from each race; all rider history strictly before race date",
        "betting_logic_changed": False
    }
    with open(OUT / "training_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    log.info("Training complete. Artifacts at %s", OUT)


if __name__ == "__main__":
    main()
