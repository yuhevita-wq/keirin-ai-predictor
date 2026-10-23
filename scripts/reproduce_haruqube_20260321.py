#!/usr/bin/env python3
"""Independent March 21, 2026 haruqube strategy restoration on fresh 2023+ data.

The GitHub workflow restores immutable upstream config/builder/line_features
snapshots from commit 336fbd87b4a04632ffe65c58de91553d431003e2
into the *temporary CI checkout* before running this script.
Nothing in production's config, data, or trained model is overwritten.

The historical source is used, with the prior reproduction's explicit historical
class/gear corrections, complete-race QA and 2025-07 to 2026-09 holdout.
Line bonus and torigami filters are not simulated (no historical formations or
per-ticket pre-race odds), and the 3/21 author-reported ROI is not presumed.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BET_STRATEGY, TRIFECTA_STRATEGY, DB_PATH, get_bet_category, get_trifecta_category
from models.lgbm_ranker import LGBMRanker
from models.trainer import evaluate_test_set
from scripts.reproduce_haruqube_fresh import prepare_db, dataset, split, TRAIN_END, VALID_END, TEST_END

OUT = Path("reproduction_20260321")
SOURCE_COMMIT = "336fbd87b4a04632ffe65c58de91553d431003e2"
EXPECTED_BLOBS = {
    "config.py": "011c2091816d16358dfe478726bcb0ff2bc74bf9",
    "features/builder.py": "b47b793b5cf763bcb6d05137a115105006e50eba",
    "features/line_features.py": "ecf3ab7c762475946054f806bb1d4b802c889389",
}
FIELDS = ["version", "race_id", "date", "grade", "tier", "confidence",
          "ticket_count", "tickets", "winning_combo", "stake_each_yen",
          "invested_yen", "returned_yen", "hit", "payout_verified"]


def verify_sources():
    for name, expected in EXPECTED_BLOBS.items():
        got = subprocess.check_output(["git", "hash-object", name], text=True).strip()
        if got != expected:
            raise RuntimeError(f"Historical source SHA mismatch: {name}: {got} != {expected}")
    assert BET_STRATEGY == [
        (1.0, "HIGH", 500, "500円×4点=2,000円"),
        (0.8, "MED+", 200, "200円×4点=800円"),
        (0.5, "MED", 100, "100円×4点=400円")], BET_STRATEGY
    assert TRIFECTA_STRATEGY == [
        (2.0, "S3-HIGH", 500, "500円×6点=3,000円"),
        (1.5, "S3-MED+", 200, "200円×6点=1,200円"),
        (1.0, "S3-MED", 100, "100円×6点=600円")], TRIFECTA_STRATEGY
    print("HISTORICAL_SOURCES_VERIFIED", SOURCE_COMMIT, flush=True)


def parse_combo(value, n):
    if value is None or pd.isna(value):
        return None
    digits = re.findall(r"\d+", str(value))
    if len(digits) != n:
        return None
    return tuple(int(x) for x in digits)


def summarize(rows):
    total = [x for x in rows if x["payout_verified"]]
    invested = sum(x["invested_yen"] for x in total)
    returned = sum(x["returned_yen"] for x in total)
    return {
        "attempted_races": len(rows),
        "payout_verified_races": len(total),
        "unverifiable_payouts": len(rows) - len(total),
        "hits": sum(x["hit"] for x in total),
        "invested_yen": invested,
        "returned_yen": returned,
        "profit_yen": returned - invested,
        "roi_percent": round(100 * returned / invested, 3) if invested else None,
    }


def verify_payout(race_row, actual, prefix):
    combo = parse_combo(race_row.iloc[0][f"{prefix}_combo"], len(actual))
    payout = race_row.iloc[0][f"{prefix}_payout"]
    valid = combo == actual and pd.notna(payout) and float(payout) >= 100
    return bool(valid), int(payout) if valid else None


def settle(version, race, gap, grade, tier, stake_each, tickets, actual, prefix):
    ok, payout = verify_payout(race, actual, prefix)
    hit = int(actual in tickets) if ok else 0
    return {"version": version, "race_id": str(race.iloc[0]["race_id"]),
            "date": str(race.iloc[0]["race_date"]), "grade": grade,
            "tier": tier, "confidence": round(gap, 6),
            "ticket_count": len(tickets),
            "tickets": ";".join(">".join(map(str, x)) for x in tickets),
            "winning_combo": ">".join(map(str, actual)),
            "stake_each_yen": stake_each,
            "invested_yen": stake_each * len(tickets) if ok else 0,
            "returned_yen": (payout * stake_each // 100) if hit else 0,
            "hit": hit, "payout_verified": ok}


def simulate(holdout, model, features):
    frame = holdout.copy()
    frame["pred_score"] = model.predict(
        frame[features].apply(pd.to_numeric, errors="coerce").fillna(0)).values
    rows = []
    for rid, race in frame.groupby("race_id", sort=False):
        pred = race.sort_values(["pred_score", "bike_number"],
                                ascending=[False, True], kind="stable")
        if len(pred) < 5:
            continue
        actual_rank = race.sort_values("finish_position", kind="stable")
        actual_exacta = tuple(int(x) for x in actual_rank.iloc[:2]["bike_number"])
        actual_trifecta = tuple(int(x) for x in actual_rank.iloc[:3]["bike_number"])
        grade = str(pred.iloc[0]["grade"])
        gap = float(pred.iloc[0]["pred_score"] - pred.iloc[1]["pred_score"])
        bikes = [int(x) for x in pred.head(5)["bike_number"]]
        assert len(set(bikes)) == 5, rid
        exacta_tickets = [(bikes[0], partner) for partner in bikes[1:5]]
        tri_tickets = [(bikes[0], second, third)
                       for second, third in itertools.permutations(bikes[1:4], 2)]
        assert len(tri_tickets) == 6

        # 2026-03-21 ORIGINAL exacta categories (historical 3 bands, F2 omitted).
        ex_tier, _, ex_stake = get_bet_category(gap, grade)
        if ex_stake:
            rows.append(settle("2026-03-21_exacta", pred, gap, grade, ex_tier,
                               ex_stake, exacta_tickets, actual_exacta, "nisyatan"))

        # 2026-03-21 ORIGINAL trifecta categories (6 non-duplicating tickets).
        tri_tier, _, tri_stake = get_trifecta_category(gap, grade)
        if tri_stake:
            rows.append(settle("2026-03-21_trifecta", pred, gap, grade, tri_tier,
                               tri_stake, tri_tickets, actual_trifecta, "sanrentan"))

        # Same 03/21 model, but 03/22 PURCHASE RULE ONLY, for fair price/selection comparison.
        # From later config: HIGH >=1.50, 500x4; exclude F2/G2/G3, without online odds filter.
        if grade not in {"F2", "G2", "G3"} and gap >= 1.50:
            rows.append(settle("2026-03-22_rule_on_0321_model", pred, gap, grade,
                               "HIGH", 500, exacta_tickets, actual_exacta, "nisyatan"))

    with (OUT / "holdout_all_selected_tickets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    groups = {v: [row for row in rows if row["version"] == v] for v in
              ("2026-03-21_exacta", "2026-03-21_trifecta",
               "2026-03-22_rule_on_0321_model")}
    summaries = {k: summarize(v) for k, v in groups.items()}
    for version in ("2026-03-21_exacta", "2026-03-21_trifecta"):
        tiers = {}
        for tier in sorted({x["tier"] for x in groups[version]}):
            tiers[tier] = summarize([x for x in groups[version] if x["tier"] == tier])
        summaries[version]["tiers"] = tiers

    both = groups["2026-03-21_exacta"] + groups["2026-03-21_trifecta"]
    summaries["2026-03-21_combined"] = summarize(both)
    # Payout missingness is not a losing ticket. No ROI unless verifiable.
    return summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The shared collector writes to its module-level OUT; point it at this
    # isolated experiment directory before preparing the database.
    import scripts.reproduce_haruqube_fresh as fresh_base
    fresh_base.OUT = OUT
    verify_sources()
    manifest = prepare_db()
    df, feature_names, valid_count = dataset()
    # Add 3連単 historical payment records without reading an upstream released database.
    with sqlite3.connect(DB_PATH) as con:
        tri_pay = pd.read_sql_query(
            "SELECT race_id,sanrentan_combo,sanrentan_payout FROM race_payouts", con)
    if tri_pay["race_id"].duplicated().any():
        raise RuntimeError("Duplicate payout race IDs")
    df = df.merge(tri_pay, on="race_id", how="left", validate="many_to_one")

    train = df[df["race_date"] < TRAIN_END]
    val = df[(df["race_date"] >= TRAIN_END) & (df["race_date"] < VALID_END)]
    test = df[(df["race_date"] >= VALID_END) & (df["race_date"] < TEST_END)]
    train, xt, yt, gt = split(train, feature_names)
    val, xv, yv, gv = split(val, feature_names)
    test, xh, yh, gh = split(test, feature_names)
    if min(len(gt), len(gv), len(gh)) < 1000:
        raise RuntimeError("Not enough races in split")
    print("HISTORICAL_SPLITS", len(gt), len(gv), len(gh), flush=True)
    model = LGBMRanker()
    model.train(xt, yt, gt, xv, yv, gv)
    roi = simulate(test, model, feature_names)
    model_file = OUT / "haruqube_20260321_fresh_model.pkl"
    model.save(str(model_file))
    report = {
        "source_upstream_commit": SOURCE_COMMIT,
        "historical_source_blob_sha": EXPECTED_BLOBS,
        "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "date_ranges": {
            "train": "2023-01-01 to 2024-12-31",
            "validation": "2025-01-01 to 2025-06-30",
            "holdout": "2025-07-01 to 2026-09-30"},
        "races": {"merged": manifest["races_imported"], "valid": valid_count,
                  "training": len(gt), "validation": len(gv), "holdout": len(gh)},
        "validation_metrics": evaluate_test_set(val, model, feature_names),
        "holdout_metrics": evaluate_test_set(test, model, feature_names),
        "holdout_betting": roi,
        "caveats": [
            "3/21 original source restored but model retrained on fresh 2023+ data",
            "same pre-race historical class/gear corrections and QA as 3/22 experiment",
            "upstream original published ROI was in-sample/backtest and is not assumed",
            "3/22 purchase-only rules evaluated on SAME 3/21 model for clean comparison",
            "different model versus prior 3/22 experiment, so standalone ROIs not a causal comparison",
            "line bonus and 4.0x odds torigami online filters not applied",
            "3連単 ROI uses ONLY races whose 3連単 payouts match independently known finish order",
        ],
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HISTORICAL_FINAL_REPORT", json.dumps({
        "races": report["races"],
        "holdout_metrics": report["holdout_metrics"],
        "holdout_betting": roi,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
