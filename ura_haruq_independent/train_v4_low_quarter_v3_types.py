#!/usr/bin/env python3
"""Independent Ura HaruQ V4 research: re-learn V3-style types per LOW quarter.

2023 = discovery, 2024 = selection, 2025 = untouched holdout evaluation.
Never updates the frozen model, HaruQ rules, live predictions or purchases.
Uses all verified 2023-2025 races from the archived historical DB.
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
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import DB_PATH
from models.lgbm_ranker import LGBMRanker
from scripts.reproduce_haruqube_fresh import dataset
from ura_haruq_independent.diagnose_low_quarters_2023_2025 import (
    MODEL_SHA256, band_for, parse_combo, type_tickets,
)

OUT = Path("ura_haruq_independent/diagnostics/v4_four_quarter_v3_relearn")
BANDS = (
    "LOW_0.00_0.25", "LOW_0.25_0.50",
    "LOW_0.50_0.75", "LOW_0.75_1.00",
)
YEARS = ("2023", "2024", "2025")
INTERVALS = tuple((lo, hi) for lo in range(7) for hi in range(lo, 7))
CANDIDATES = tuple(itertools.product(INTERVALS, repeat=3))
MAX_TYPES_PER_BAND = 5
MIN_INCREMENTAL_HITS_EACH_YEAR = 5
MIN_AVERAGE_NEW_POINTS = 1.0
MAX_AVERAGE_NEW_POINTS = 24.0
BET_UNIT_YEN = 100


def build_ticket_geometry():
    """The *same* 21,952 7-bin cuboids and bottom-half rule as V3."""
    bounds = np.array(CANDIDATES, dtype=np.int8)
    masks = {}
    perms = {}
    perm_index = {}
    for n in (7, 8, 9):
        bottom_start = (n + 1) // 2
        triples = tuple(
            (a, b, c) for a, b, c in itertools.permutations(range(n), 3)
            if max(a, b, c) >= bottom_start
        )
        index = {t: i for i, t in enumerate(triples)}
        ranks = np.asarray(triples, dtype=np.int8)
        mask = np.ones((len(CANDIDATES), len(triples)), dtype=np.bool_)
        for pos in range(3):
            rbin = (7 * ranks[:, pos]) // n
            mask &= (bounds[:, pos, 0, None] <= rbin)
            mask &= (rbin <= bounds[:, pos, 1, None])
        masks[n] = mask
        perms[n] = triples
        perm_index[n] = index
    return masks, perms, perm_index


def self_test():
    masks, perms, _ = build_ticket_geometry()
    assert len(CANDIDATES) == 28 ** 3 == 21952
    assert band_for(0.0) == "LOW_0.00_0.25"
    assert band_for(0.25) == "LOW_0.25_0.50"
    assert band_for(0.50) == "LOW_0.50_0.75"
    assert band_for(0.75) == "LOW_0.75_1.00"
    assert band_for(1.0) == "MID_1.00_1.50"
    for bounds in (((0, 2), (0, 0), (0, 3)),
                   ((6, 6), (1, 1), (3, 3))):
        k = CANDIDATES.index(bounds)
        for n in (7, 8, 9):
            expected = type_tickets(list(range(1, n + 1)), bounds)
            found = {
                tuple(v + 1 for v in triple)
                for triple, flag in zip(perms[n], masks[n][k]) if flag
            }
            assert expected == found, (bounds, n, len(expected), len(found))
    print("V4_GEOMETRY_SELF_TEST_OK", len(CANDIDATES), flush=True)


def load_races(model_path):
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("FROZEN_MODEL_SHA_MISMATCH")
    if not DB_PATH.exists():
        raise RuntimeError("HISTORICAL_DB_MISSING")
    with sqlite3.connect(DB_PATH) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {"races", "entries", "race_results", "race_payouts", "riders"} - tables
        if missing:
            raise RuntimeError("MISSING_HISTORICAL_TABLES:" + repr(sorted(missing)))
        metadata = pd.read_sql_query(
            "SELECT race_id,race_name,rider_count FROM races", conn)
        payouts = pd.read_sql_query(
            "SELECT race_id,sanrentan_combo,sanrentan_payout FROM race_payouts", conn)
    if payouts["race_id"].duplicated().any():
        raise RuntimeError("DUPLICATE_PAYOUT_ID")
    metadata = metadata.drop_duplicates("race_id").set_index("race_id")
    model = LGBMRanker()
    model.load(str(model_path))
    frame, names, valid_history = dataset()
    if len(names) != 44 or list(model.feature_names) != list(names):
        raise RuntimeError("FROZEN_44_FEATURE_CONTRACT_MISMATCH")
    frame = frame.merge(payouts, on="race_id", how="left", validate="many_to_one")
    frame = frame[
        (frame["race_date"].astype(str) >= "2023-01-01") &
        (frame["race_date"].astype(str) < "2026-01-01")
    ].copy()
    counts = frame.groupby("race_id", sort=False).size()
    valid_ids = set(counts[counts.between(7, 9)].index)
    frame = frame[frame["race_id"].isin(valid_ids)].copy()
    if frame.empty:
        raise RuntimeError("NO_2023_TO_2025_HISTORY")
    features = frame[names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scores = np.asarray(model.predict(features), dtype=float)
    if not np.isfinite(scores).all():
        raise RuntimeError("NONFINITE_SCORES")
    frame["frozen_score"] = scores
    if "class" not in frame:
        raise RuntimeError("MISSING_RACE_CLASS")
    _, _, perm_index = build_ticket_geometry()
    rows = []
    counts_diagnostic = defaultdict(int)
    for race_id, group in frame.groupby("race_id", sort=False):
        counts_diagnostic["history_7to9_races"] += 1
        if race_id not in metadata.index:
            counts_diagnostic["missing_metadata"] += 1
            continue
        name = str(metadata.loc[race_id]["race_name"] or "")
        classes = group["class"].fillna("").astype(str)
        if classes.str.match(r"(?i)^L\d*").any() or re.search(
            r"ガールズ|ガ決|ガ般|Ｌ級|L級", name
        ):
            counts_diagnostic["girls_excluded"] += 1
            continue
        ordered = group.sort_values(
            ["frozen_score", "bike_number"], ascending=[False, True],
            kind="stable")
        bikes = [int(i) for i in ordered["bike_number"]]
        n = len(bikes)
        if len(set(bikes)) != n:
            counts_diagnostic["duplicate_bike_excluded"] += 1
            continue
        gap = float(ordered.iloc[0]["frozen_score"] -
                    ordered.iloc[1]["frozen_score"])
        if not math.isfinite(gap) or gap < -1e-8:
            raise RuntimeError("INVALID_SCORE_GAP")
        gap = max(gap, 0.0)
        if gap >= 1.0:
            continue
        settled = group.sort_values("finish_position", kind="stable")
        actual = tuple(int(v) for v in settled.head(3)["bike_number"])
        if len(set(actual)) != 3:
            counts_diagnostic["invalid_result_excluded"] += 1
            continue
        payout_combo = parse_combo(group.iloc[0]["sanrentan_combo"])
        payout = group.iloc[0]["sanrentan_payout"]
        if payout_combo != actual or pd.isna(payout) or float(payout) < 100:
            counts_diagnostic["unverified_payout_excluded"] += 1
            continue
        if not float(payout).is_integer():
            raise RuntimeError("NONINTEGER_TRIFECTA_PAYOUT")
        lookup = {bike: i for i, bike in enumerate(bikes)}
        actual_ranks = tuple(lookup[b] for b in actual)
        rows.append({
            "race_id": str(race_id),
            "race_date": str(group.iloc[0]["race_date"]),
            "year": str(group.iloc[0]["race_date"])[:4],
            "band": band_for(gap),
            "score_gap": round(gap, 9),
            "n": n,
            "grade": str(group.iloc[0]["grade"]),
            "actual": "-".join(map(str, actual)),
            "official_trifecta_payout": int(payout),
            "actual_perm_index": perm_index[n].get(actual_ranks, -1),
        })
    if not rows:
        raise RuntimeError("NO_ELIGIBLE_LOW_RACES")
    counts_diagnostic["settled_low_races"] = len(rows)
    return rows, dict(counts_diagnostic), valid_history


def make_aggregated_data(rows, permutations):
    data = {}
    for band in BANDS:
        data[band] = {}
        for y in YEARS:
            by_n = {}
            for n in (7, 8, 9):
                part = [r for r in rows if r["band"] == band
                        and r["year"] == y and r["n"] == n]
                pays = np.zeros(len(permutations[n]), dtype=np.int64)
                hits = np.zeros(len(permutations[n]), dtype=np.int64)
                for r in part:
                    idx = r["actual_perm_index"]
                    if idx >= 0:
                        pays[idx] += r["official_trifecta_payout"]
                        hits[idx] += 1
                by_n[n] = {
                    "races": len(part), "pays": pays, "hits": hits,
                }
            data[band][y] = by_n
    return data


def marginal_all_candidates(masks, purchased, years_data):
    metrics = {}
    for y, by_n in years_data.items():
        cost = np.zeros(len(CANDIDATES), dtype=np.int64)
        value = np.zeros(len(CANDIDATES), dtype=np.int64)
        won = np.zeros(len(CANDIDATES), dtype=np.int64)
        for n in (7, 8, 9):
            additions = masks[n] & ~purchased[n][None, :]
            cost += (BET_UNIT_YEN * by_n[n]["races"] *
                     additions.sum(axis=1, dtype=np.int64))
            value += additions @ by_n[n]["pays"]
            won += additions @ by_n[n]["hits"]
        metrics[y] = {"cost": cost, "value": value, "hits": won}
    return metrics


def discover(band, data, masks):
    # The 2025 label dictionary is deliberately never passed here.
    years_data = {y: data[band][y] for y in ("2023", "2024")}
    purchased = {n: np.zeros(masks[n].shape[1], dtype=np.bool_)
                 for n in (7, 8, 9)}
    selected = []
    avg_races = sum(years_data[y][n]["races"]
                    for y in years_data for n in (7, 8, 9))
    for step in range(MAX_TYPES_PER_BAND):
        m = marginal_all_candidates(masks, purchased, years_data)
        a, b = m["2023"], m["2024"]
        with np.errstate(divide="ignore", invalid="ignore"):
            roi_a = np.divide(a["value"], a["cost"],
                              out=np.zeros_like(a["value"], dtype=float),
                              where=a["cost"] > 0)
            roi_b = np.divide(b["value"], b["cost"],
                              out=np.zeros_like(b["value"], dtype=float),
                              where=b["cost"] > 0)
        avg = ((a["cost"] + b["cost"]) /
               (BET_UNIT_YEN * max(avg_races, 1)))
        valid = (
            (a["cost"] > 0) & (b["cost"] > 0) &
            (a["hits"] >= MIN_INCREMENTAL_HITS_EACH_YEAR) &
            (b["hits"] >= MIN_INCREMENTAL_HITS_EACH_YEAR) &
            (roi_a > 1.0) & (roi_b > 1.0) &
            (avg >= MIN_AVERAGE_NEW_POINTS) &
            (avg <= MAX_AVERAGE_NEW_POINTS)
        )
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            break
        # Stable V3 research tie-break: stronger worst-year incremental ROI,
        # then larger 2024 incremental profit, then more 2024 hits.
        best = max(
            (int(i) for i in candidates),
            key=lambda i: (
                min(float(roi_a[i]), float(roi_b[i])),
                int(b["value"][i] - b["cost"][i]),
                int(b["hits"][i]), -int(i),
            ))
        item = {
            "type_id": f"{band}_V4_{step + 1}",
            "candidate_index": best,
            "bin_bounds_0based": [list(x) for x in CANDIDATES[best]],
            "incremental_average_new_points": round(float(avg[best]), 5),
            "incremental_2023": {
                "tickets": int(a["cost"][best] // BET_UNIT_YEN),
                "hits": int(a["hits"][best]),
                "investment_yen": int(a["cost"][best]),
                "return_yen": int(a["value"][best]),
                "roi_percent": round(float(100 * roi_a[best]), 5),
            },
            "incremental_2024": {
                "tickets": int(b["cost"][best] // BET_UNIT_YEN),
                "hits": int(b["hits"][best]),
                "investment_yen": int(b["cost"][best]),
                "return_yen": int(b["value"][best]),
                "roi_percent": round(float(100 * roi_b[best]), 5),
            },
        }
        selected.append(item)
        for n in (7, 8, 9):
            purchased[n] |= masks[n][best]
    return selected, purchased


def evaluate(band, year, rows, purchased):
    subset = [r for r in rows if r["band"] == band and r["year"] == year]
    n_tickets = {n: int(purchased[n].sum()) for n in (7, 8, 9)}
    tickets = sum(n_tickets[r["n"]] for r in subset)
    invested = BET_UNIT_YEN * tickets
    winners = [
        r for r in subset if r["actual_perm_index"] >= 0 and
        purchased[r["n"]][r["actual_perm_index"]]
    ]
    returned = sum(r["official_trifecta_payout"] for r in winners)
    highest = max((r["official_trifecta_payout"] for r in winners), default=0)
    return {
        "races": len(subset),
        "ticket_count_by_field_size": {str(n): n_tickets[n] for n in (7, 8, 9)},
        "tickets": tickets,
        "hits": len(winners),
        "investment_yen": invested,
        "return_yen": returned,
        "profit_yen": returned - invested,
        "roi_percent": round(100 * returned / invested, 5)
            if invested else None,
        "largest_single_hit_yen": highest,
        "roi_without_largest_hit_percent": round(
            100 * (returned - highest) / invested, 5
        ) if invested else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if args.model is None:
        parser.error("--model is required unless --self-test")
    masks, permutations, _ = build_ticket_geometry()
    rows, counts, valid_history = load_races(args.model)
    data = make_aggregated_data(rows, permutations)
    report = {
        "system": "URA_V4_FOUR_QUARTER_V3_STYLE_RELEARN",
        "research_only": True,
        "purchase_approved": False,
        "original_model_mutated": False,
        "original_purchase_policy_mutated": False,
        "original_model_sha256": MODEL_SHA256,
        "candidate_count": len(CANDIDATES),
        "type_family": "V3 axis-aligned cuboids over 7 normalized rank bins",
        "bottom_half_rule": "any predicted rank index >= (field_size+1)//2",
        "date_protocol": {
            "candidate_discovery": "2023-01-01 through 2023-12-31",
            "selection": "2024-01-01 through 2024-12-31",
            "untouched_holdout": "2025-01-01 through 2025-12-31",
        },
        "type_selection": {
            "max_types_per_band": MAX_TYPES_PER_BAND,
            "min_incremental_hits_in_each_2023_2024": MIN_INCREMENTAL_HITS_EACH_YEAR,
            "roi_incremental_each_2023_2024": "strictly above 100%",
            "mean_new_points_over_2023_2024_races": [
                MIN_AVERAGE_NEW_POINTS, MAX_AVERAGE_NEW_POINTS],
            "greedy_rank": "maximum minimum 2023/2024 ROI, then 2024 profit",
            "2025_labels_used_for_selection": False,
        },
        "denominators": {
            "historical_quality_valid": valid_history,
            **counts,
        },
        "quarters": {},
        "caveats": [
            "The frozen 2026 model might have been trained using historical years: "
            "the 2025 type holdout does not guarantee model-level out-of-sample prediction.",
            "Historical as-of-race-time feature parity is not proven.",
            "Types that fail selection remain empty; no forced winner or fixed point count.",
            "Huge 3連単 dividends can dominate ROI. Report largest-hit sensitivity.",
            "Results are research only, never auto-published to live prediction.",
        ],
    }
    purchased_by_band = {}
    for band in BANDS:
        types, purchased = discover(band, data, masks)
        purchased_by_band[band] = purchased
        report["quarters"][band] = {
            "type_count": len(types),
            "types": types,
            "yearly": {
                y: evaluate(band, y, rows, purchased) for y in YEARS
            },
        }
        print("QUARTER_V4_COMPLETE",
              json.dumps({
                  "band": band, "type_count": len(types),
                  "2025": report["quarters"][band]["yearly"]["2025"],
              }, ensure_ascii=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "v4_four_quarter_relearn.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (OUT / "selected_types.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        columns = [
            "band", "type_id", "candidate_index", "bin_bounds_0based",
            "incremental_average_new_points", "2023_incremental_roi",
            "2024_incremental_roi", "2023_incremental_hits",
            "2024_incremental_hits",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for band, item in report["quarters"].items():
            for t in item["types"]:
                writer.writerow({
                    "band": band, "type_id": t["type_id"],
                    "candidate_index": t["candidate_index"],
                    "bin_bounds_0based": json.dumps(t["bin_bounds_0based"]),
                    "incremental_average_new_points":
                        t["incremental_average_new_points"],
                    "2023_incremental_roi":
                        t["incremental_2023"]["roi_percent"],
                    "2024_incremental_roi":
                        t["incremental_2024"]["roi_percent"],
                    "2023_incremental_hits":
                        t["incremental_2023"]["hits"],
                    "2024_incremental_hits":
                        t["incremental_2024"]["hits"],
                })
    with gzip.open(OUT / "race_level_outcomes.csv.gz", "wt",
                   encoding="utf-8", newline="") as f:
        columns = [
            "race_id", "race_date", "year", "band", "score_gap", "n",
            "grade", "actual", "official_trifecta_payout", "ticket_count",
            "hit", "investment_yen", "return_yen",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            mask = purchased_by_band[r["band"]][r["n"]]
            ticket_count = int(mask.sum())
            hit = int(r["actual_perm_index"] >= 0 and
                      mask[r["actual_perm_index"]])
            writer.writerow({
                "race_id": r["race_id"], "race_date": r["race_date"],
                "year": r["year"], "band": r["band"],
                "score_gap": r["score_gap"], "n": r["n"],
                "grade": r["grade"], "actual": r["actual"],
                "official_trifecta_payout": r["official_trifecta_payout"],
                "ticket_count": ticket_count, "hit": hit,
                "investment_yen": BET_UNIT_YEN * ticket_count,
                "return_yen": r["official_trifecta_payout"] if hit else 0,
            })
    print("URA_V4_FOUR_QUARTER_DONE", json.dumps({
        band: {
            "types": rep["type_count"],
            "2025_roi": rep["yearly"]["2025"]["roi_percent"],
        } for band, rep in report["quarters"].items()
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
