#!/usr/bin/env python3
"""Read-only V3-1/V3-2 LOW confidence quarter-band diagnostic (2023-2025).

RESEARCH ONLY. Uses the frozen March 21 HaruQ model unchanged; never trains,
rewrites operational thresholds, changes bets, or publishes a purchase signal.
Only independently verified official 3連単 settlements enter ROI denominators.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import DB_PATH
from models.lgbm_ranker import LGBMRanker
from scripts.reproduce_haruqube_fresh import dataset

MODEL_SHA256 = "e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193"
V3 = {
    "V3_1": ((0, 2), (0, 0), (0, 3)),
    "V3_2": ((6, 6), (1, 1), (3, 3)),
}
BANDS = [
    ("LOW_0.00_0.25", 0.00, 0.25),
    ("LOW_0.25_0.50", 0.25, 0.50),
    ("LOW_0.50_0.75", 0.50, 0.75),
    ("LOW_0.75_1.00", 0.75, 1.00),
    ("MID_1.00_1.50", 1.00, 1.50),
    ("MEDPLUS_1.50_2.00", 1.50, 2.00),
    ("HIGH_2.00_PLUS", 2.00, math.inf),
]
STAKE_PER_TICKET = 100
FIELDS = (
    "race_id", "race_date", "year", "n", "grade", "score_gap",
    "band", "actual", "official_trifecta_payout", "payout_verified",
    "v3_1_tickets", "v3_1_hit", "v3_2_tickets", "v3_2_hit",
    "union_tickets", "union_hit",
)
OUT = Path("ura_haruq_independent/diagnostics/low_quarter_2023_2025")


def band_for(gap):
    for name, lower, upper in BANDS:
        if lower <= gap < upper:
            return name
    raise ValueError(f"negative or non-finite score gap: {gap}")


def rank_bin(index, n):
    """V3 saved model: 0-based normalized seven-bin rank."""
    return (7 * index) // n


def type_tickets(ordered_bikes, bounds):
    n = len(ordered_bikes)
    # Original independent V3 restriction: at least one bottom-half rank.
    bottom_start = (n + 1) // 2
    choices = [
        [i for i in range(n) if lo <= rank_bin(i, n) <= hi]
        for lo, hi in bounds
    ]
    chosen = set()
    for a, b, c in itertools.product(*choices):
        if len({a, b, c}) != 3:
            continue
        if max(a, b, c) < bottom_start:
            continue
        chosen.add((ordered_bikes[a], ordered_bikes[b], ordered_bikes[c]))
    return chosen


def parse_combo(value):
    if value is None or pd.isna(value):
        return None
    digits = re.findall(r"\d+", str(value))
    if len(digits) != 3:
        return None
    parsed = tuple(int(v) for v in digits)
    return parsed if len(set(parsed)) == 3 else None


def summarize(records, bin_name):
    spend, receive, hits, tickets = 0, 0, 0, 0
    eligible, settled, unsettled = 0, 0, 0
    for r in records:
        t = int(r[f"{bin_name}_tickets"])
        if not t:
            continue
        eligible += 1
        if not r["payout_verified"]:
            unsettled += 1
            continue
        settled += 1
        tickets += t
        spend += STAKE_PER_TICKET * t
        if int(r[f"{bin_name}_hit"]):
            hits += 1
            receive += int(r["official_trifecta_payout"])
    return {
        "eligible_races": eligible,
        "settled_races": settled,
        "unverified_races": unsettled,
        "tickets": tickets,
        "hits": hits,
        "hit_rate_percent": round(100 * hits / settled, 5) if settled else None,
        "investment_yen": spend,
        "return_yen": receive,
        "profit_yen": receive - spend,
        "roi_percent": round(100 * receive / spend, 5) if spend else None,
    }


def aggregate(records, denominator):
    verified = [x for x in records if x["payout_verified"]]
    payout = [int(x["official_trifecta_payout"]) for x in verified]
    return {
        "races": len(records),
        "share_of_eligible_percent": round(100 * len(records) / denominator, 5)
            if denominator else None,
        "settlement_verified_races": len(verified),
        "settlement_unverified_races": len(records) - len(verified),
        "actual_trifecta_payout_mean_yen": round(statistics.mean(payout), 2)
            if payout else None,
        "actual_trifecta_payout_median_yen": round(statistics.median(payout), 2)
            if payout else None,
        "v3_1": summarize(records, "v3_1"),
        "v3_2": summarize(records, "v3_2"),
        "v3_1_plus_v3_2_union": summarize(records, "union"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-races", type=int, default=0,
                        help="Optional debug limit; 0 = ALL eligible historical races.")
    args = parser.parse_args()
    if args.max_races < 0:
        raise ValueError("negative max races")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("FROZEN_MODEL_SHA_MISMATCH")
    if not DB_PATH.exists():
        raise RuntimeError("HISTORICAL_DB_MISSING")
    with sqlite3.connect(DB_PATH) as conn:
        present = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        need = {"races", "entries", "race_results", "race_payouts", "riders"}
        if not need <= present:
            raise RuntimeError(f"HISTORY_TABLES_MISSING: {sorted(need-present)}")
        metadata = pd.read_sql_query(
            "SELECT race_id,race_name,rider_count FROM races", conn)
        tri_pay = pd.read_sql_query(
            "SELECT race_id,sanrentan_combo,sanrentan_payout FROM race_payouts", conn)
        if tri_pay["race_id"].duplicated().any():
            raise RuntimeError("DUPLICATE_TRIFECTA_PAYOUT_RACE_ID")
    metadata = metadata.drop_duplicates("race_id").set_index("race_id")

    model = LGBMRanker()
    model.load(str(args.model))
    frame, names, valid_history = dataset()  # Archived history only; NO retraining.
    frame = frame.merge(tri_pay, on="race_id", how="left", validate="many_to_one")
    if len(names) != 44 or list(model.feature_names) != list(names):
        raise RuntimeError("FROZEN_44_FEATURE_CONTRACT_MISMATCH")
    frame = frame[
        (frame["race_date"].astype(str) >= "2023-01-01") &
        (frame["race_date"].astype(str) < "2026-01-01")
    ].copy()
    counts = frame.groupby("race_id", sort=False).size()
    ids = set(counts[(counts >= 7) & (counts <= 9)].index)
    frame = frame[frame["race_id"].isin(ids)].copy()
    if frame.empty:
        raise RuntimeError("NO_VALID_2023_2025_RACES")
    X = frame[names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scores = model.predict(X)
    frame["frozen_score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(frame["frozen_score"]).all():
        raise RuntimeError("NONFINITE_FROZEN_SCORES")
    if "class" not in frame:
        raise RuntimeError("HISTORICAL_CLASS_FIELD_MISSING")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_classified = 0
    excluded_girls = 0
    missing_meta = 0
    bad_ranks = 0
    for race_id, race in frame.groupby("race_id", sort=False):
        raw_classified += 1
        if race_id not in metadata.index:
            missing_meta += 1
            continue
        meta = metadata.loc[race_id]
        race_name = str(meta["race_name"] or "")
        classes = race["class"].fillna("").astype(str)
        if (classes.str.match(r"(?i)^L\d*").any() or
                bool(re.search(r"ガールズ|ガ決|ガ般|Ｌ級|L級", race_name))):
            excluded_girls += 1
            continue
        n = len(race)
        ordered = race.sort_values(
            ["frozen_score", "bike_number"],
            ascending=[False, True], kind="stable")
        bikes = [int(x) for x in ordered["bike_number"]]
        if len(set(bikes)) != n:
            bad_ranks += 1
            continue
        gap = float(ordered.iloc[0]["frozen_score"] -
                    ordered.iloc[1]["frozen_score"])
        if not math.isfinite(gap) or gap < -1e-8:
            raise RuntimeError(f"INVALID_SCORE_GAP {race_id} {gap}")
        gap = max(0.0, gap)
        settled = race.sort_values("finish_position", kind="stable")
        actual = tuple(int(v) for v in settled.head(3)["bike_number"])
        if len(set(actual)) != 3:
            bad_ranks += 1
            continue
        payout_combo = parse_combo(race.iloc[0]["sanrentan_combo"])
        payout_value = race.iloc[0]["sanrentan_payout"]
        verified = bool(payout_combo == actual and
                        pd.notna(payout_value) and
                        float(payout_value) >= 100)
        if verified and not float(payout_value).is_integer():
            raise RuntimeError(f"NON_INTEGER_PAYOUT {race_id}")
        t1 = type_tickets(bikes, V3["V3_1"]) if gap < 1.0 else set()
        t2 = type_tickets(bikes, V3["V3_2"]) if gap < 1.0 else set()
        union = t1 | t2
        rows.append({
            "race_id": str(race_id),
            "race_date": str(race.iloc[0]["race_date"]),
            "year": str(race.iloc[0]["race_date"])[:4],
            "n": n,
            "grade": str(race.iloc[0]["grade"]),
            "score_gap": round(gap, 9),
            "band": band_for(gap),
            "actual": "-".join(map(str, actual)),
            "official_trifecta_payout": int(payout_value) if verified else "",
            "payout_verified": int(verified),
            "v3_1_tickets": len(t1), "v3_1_hit": int(actual in t1),
            "v3_2_tickets": len(t2), "v3_2_hit": int(actual in t2),
            "union_tickets": len(union), "union_hit": int(actual in union),
        })
        if args.max_races and len(rows) >= args.max_races:
            break
    if not rows:
        raise RuntimeError("NO_COMPARABLE_RACES")
    overall = len(rows)
    low = [x for x in rows if float(x["score_gap"]) < 1.0]
    report = {
        "system": "URA_LOW_QUARTER_DIAGNOSTIC_2023_2025",
        "research_only": True,
        "original_model_mutated": False,
        "original_purchase_policy_mutated": False,
        "model_sha256": MODEL_SHA256,
        "source": "Frozen 2026-03-21 44-feature LambdaRank model + 2023-2025 encrypted history cache",
        "period": "2023-01-01 through 2025-12-31",
        "score_gap_definition": "top predicted LambdaRank score minus second predicted LambdaRank score",
        "bins": "[lower,upper) for all bands; 1.00 belongs to MID",
        "v3_patterns": {
            name: [list(b) for b in bounds] for name, bounds in V3.items()
        },
        "normalized_rank_bin_zero_based": "floor(7 * predicted_rank_index_zero_based / field_size)",
        "lower_rank_requirement": "at least one rider index >= floor((field_size+1)/2)",
        "stake_per_ticket_yen": STAKE_PER_TICKET,
        "denominators": {
            "full_history_quality_valid_before_7to9_filter": valid_history,
            "7to9_history_before_non_girls_filter": raw_classified,
            "excluded_girls": excluded_girls,
            "metadata_missing": missing_meta,
            "rank_errors": bad_ranks,
            "analyzed_races": overall,
            "low_races": len(low),
            "low_fraction_percent": round(100 * len(low) / overall, 5),
        },
        "yearly": {},
        "three_year": {},
        "caveats": [
            "Historical in-sample/validation years are not a clean out-of-sample ROI test.",
            "Historical features may contain post-race information; as-of race-time parity is unproven.",
            "The stored prior V3 archive was not republished or used to look up outcome labels for prediction.",
            "V3 type ticket reconstruction is the 7-bin rank cuboid with the lower-half rule.",
            "A race with missing or disagreeing official 3連単 payout is excluded from ALL ROI denominators.",
            "No score threshold, production policy, model or staking configuration was changed."
        ],
    }
    for year in ("2023", "2024", "2025"):
        subset = [x for x in rows if x["year"] == year]
        year_low = [x for x in subset if float(x["score_gap"]) < 1.0]
        report["yearly"][year] = {
            "total_qualified": len(subset),
            "low": aggregate(year_low, len(subset)),
            "bins": {name: aggregate([x for x in subset if x["band"] == name],
                                     len(subset)) for name, _, _ in BANDS},
        }
    report["three_year"] = {
        "total_qualified": overall,
        "low": aggregate(low, overall),
        "bins": {name: aggregate([x for x in rows if x["band"] == name], overall)
                 for name, _, _ in BANDS},
    }
    # No statistical proof of separation from one profitability peak:
    # always publish denominators and per-year stability side by side.
    with gzip.open(OUT / "all_races.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with (OUT / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print("URA_LOW_QUARTER_COMPLETE",
          json.dumps({"denominators": report["denominators"],
                      "three_year": report["three_year"]["bins"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
