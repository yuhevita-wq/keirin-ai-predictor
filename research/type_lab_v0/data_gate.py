#!/usr/bin/env python3
"""FAIL-CLOSED census for the full historical keirin universe.

This is a connection gate, NOT a collector and NOT proof that a model trained.
Each CSV must be exported in full from independently versioned source data.
No --head, limit, random sampling, or silent filtering is supported.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import sys
from pathlib import Path

# Historical full-archive reference: 2,005 days / 133,485 race IDs, 2021-2026 H1.
# This is a reported reference, NOT a recount by this script.
REFERENCE_FULL_ARCHIVE_RACES = 133_485
REFERENCE_ORIGINAL_7_9 = 107_072 + 11_282
REFERENCE_BY_YEAR = {"2021": 23668, "2022": 24325, "2023": 24923, "2024": 24540, "2025": 24174, "2026": 11855}
REFERENCE_BY_CARS = {7: 107072, 9: 11282}


def read_csv(path, required):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing source CSV: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: required columns {list(required)}; got {reader.fieldnames}")
        for lineno, row in enumerate(reader, start=2):
            if not (row.get("race_id") or "").strip():
                raise ValueError(f"{path}:{lineno}: empty race_id")
            yield row


def tally(*, universe, entries, results, odds, exclusions=None,
          minimum_universe=REFERENCE_FULL_ARCHIVE_RACES, source_revision=""):
    issues = []
    U = {}  # race_id -> (year, entry_count, grade)
    year_count = collections.Counter()
    n_count = collections.Counter()
    grade_count = collections.Counter()

    for row in read_csv(universe, ("race_id", "race_date", "entry_count")):
        rid = row["race_id"].strip()
        if rid in U:
            issues.append(f"duplicate universe race_id: {rid}")
            continue
        try:
            n = int(row["entry_count"])
            if n not in (5, 6, 7, 8, 9):
                raise ValueError
            year = int(row["race_date"][:4])
            if not 2010 <= year <= 2100:
                raise ValueError
        except (ValueError, TypeError):
            issues.append(f"universe invalid date / entry_count: {rid}")
            continue
        U[rid] = (str(year), n, row.get("meeting_grade") or "")
        year_count[str(year)] += 1
        n_count[n] += 1
        grade_count[row.get("meeting_grade") or "unknown"] += 1

    exclusion = {}
    if exclusions:
        for row in read_csv(exclusions, ("race_id", "reason")):
            rid = row["race_id"].strip()
            if rid in exclusion or not row["reason"].strip():
                issues.append(f"duplicate/blank exclusion: {rid}")
            exclusion[rid] = row["reason"].strip()
            if rid not in U:
                issues.append(f"exclusion outside universe: {rid}")
    entry_cars = collections.defaultdict(set)
    duplicate_cars = 0
    orphan_rows = collections.Counter()
    for row in read_csv(entries, ("race_id", "car")):
        rid = row["race_id"].strip()
        if rid not in U:
            orphan_rows["entries"] += 1
            continue
        try:
            car = int(row["car"])
        except (ValueError, TypeError):
            issues.append(f"invalid car: {rid}")
            continue
        if car in entry_cars[rid]:
            duplicate_cars += 1
        entry_cars[rid].add(car)

    results_seen = set()
    results_duplicates = 0
    for row in read_csv(results, ("race_id", "first", "second", "third")):
        rid = row["race_id"].strip()
        if rid not in U:
            orphan_rows["results"] += 1
            continue
        if rid in results_seen:
            results_duplicates += 1
        results_seen.add(rid)
        try:
            winner = [int(row[k]) for k in ("first", "second", "third")]
            if len(set(winner)) < 3 or any(x not in range(1, U[rid][1] + 1) for x in winner):
                raise ValueError
        except (ValueError, TypeError):
            issues.append(f"invalid/ambiguous finishing order: {rid}")

    odds_kind_count = collections.Counter()
    odds_races = set()
    odds_trifecta = set()
    odds_trio = set()
    for row in read_csv(odds, ("race_id", "bet_type")):
        rid = row["race_id"].strip()
        if rid not in U:
            orphan_rows["odds"] += 1
            continue
        kind = row["bet_type"].strip().lower()
        odds_kind_count[kind] += 1
        odds_races.add(rid)
        if kind in ("3rentan", "trifecta"):
            odds_trifecta.add(rid)
        if kind in ("3renpuku", "trio"):
            odds_trio.add(rid)

    coverage = {
        "entries_races": len(entry_cars),
        "results_races": len(results_seen),
        "any_odds_races": len(odds_races),
        "trifecta_odds_races": len(odds_trifecta),
        "trio_odds_races": len(odds_trio),
    }
    missing = {
        "entries": sorted(U.keys() - entry_cars.keys()),
        "results": sorted(U.keys() - results_seen),
        "odds": sorted(U.keys() - odds_races),
        "trifecta": sorted(U.keys() - odds_trifecta),
        "trio": sorted(U.keys() - odds_trio),
    }
    incomplete_lineup = []
    for rid, (_, n, _) in U.items():
        if rid in entry_cars and entry_cars[rid] != set(range(1, n + 1)):
            incomplete_lineup.append(rid)

    missing_any = set().union(*map(set, missing.values())) | set(incomplete_lineup)
    unexplained = missing_any - exclusion.keys()
    unused_exclusion = exclusion.keys() - missing_any
    original_supported = {rid for rid, (_, n, _) in U.items() if n in (7, 9)}
    eligible = original_supported - missing_any - exclusion.keys()
    unsupported = set(U) - original_supported

    # Strict: sources cannot be silently reduced even if the remaining rows are healthy.
    if len(U) < minimum_universe:
        issues.append(f"universe below minimum {minimum_universe}: {len(U)}")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision.strip().lower()):
        issues.append("source_revision must be a full 40-character Git commit SHA")
    if minimum_universe >= REFERENCE_FULL_ARCHIVE_RACES:
        for year, expected in REFERENCE_BY_YEAR.items():
            if year_count[year] < expected:
                issues.append(f"historical coverage below reference in {year}: {year_count[year]} < {expected}")
        for cars, expected in REFERENCE_BY_CARS.items():
            if n_count[cars] < expected:
                issues.append(f"historical coverage below reference for {cars}-car: {n_count[cars]} < {expected}")
    if not len(U):
        issues.append("empty universe")
    if duplicate_cars:
        issues.append(f"duplicate car records: {duplicate_cars}")
    if results_duplicates:
        issues.append(f"duplicate result labels: {results_duplicates}")
    if any(orphan_rows.values()):
        issues.append(f"orphan source rows: {dict(orphan_rows)}")
    if unexplained:
        issues.append(f"missing data for {len(unexplained)} races with no explicit exclusion reason")
    if unused_exclusion:
        issues.append(f"{len(unused_exclusion)} exclusions have no missing-data justification")

    report = {
        "status": "PASS" if not issues else "FAIL",
        "source_revision": source_revision,
        "reference_only_not_recount": {
            "historical_all_vehicle_races": REFERENCE_FULL_ARCHIVE_RACES,
            "historical_7_and_9_car_races": REFERENCE_ORIGINAL_7_9,
        },
        "actual_universe_races": len(U),
        "original_engine_7_9_car_races": len(original_supported),
        "eligible_7_9_car_races_with_three_modalities": len(eligible),
        "not_eligible_original_engine_5_6_8_car_races": len(unsupported),
        "excluded_races_explicit": len(exclusion),
        "missing_races_with_documented_reason": len(missing_any & exclusion.keys()),
        "missing_races_without_documented_reason": len(unexplained),
        "year_counts": dict(sorted(year_count.items())),
        "entry_counts_by_car_count": dict(sorted(n_count.items())),
        "grade_counts": dict(sorted(grade_count.items())),
        "modality_coverage": coverage,
        "odds_rows_by_type": dict(sorted(odds_kind_count.items())),
        "missing_breakdown": {k: len(v) for k, v in missing.items()},
        "incomplete_lineup_races": len(incomplete_lineup),
        "duplicate_car_records": duplicate_cars,
        "duplicate_result_records": results_duplicates,
        "source_rows_not_in_universe": dict(orphan_rows),
        "issues": issues,
        "notes": [
            "This counts a NORMALIZED source export, not the complete Git history or encrypted archives.",
            "Universe completeness must be checked independently against archive file inventory and checksums.",
            "Membership alone does not prove full odds combination coverage or pre-race availability.",
            "No model was trained and no money was bet.",
            "Original Type Lab only accepts 7/9-car input; unsupported races remain in the audit.",
        ],
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fail-closed full-archive data census")
    parser.add_argument("--universe", required=True, help="all expected race IDs and metadata")
    parser.add_argument("--entries", required=True, help="all competitors from source assets")
    parser.add_argument("--results", required=True, help="all standardized results")
    parser.add_argument("--odds", required=True, help="all available odds records")
    parser.add_argument("--exclusions", help="race_id,reason; ALL omissions must be justified")
    parser.add_argument("--source-revision", required=True, help="immutable source Git commit")
    parser.add_argument("--minimum-universe", type=int, default=REFERENCE_FULL_ARCHIVE_RACES)
    parser.add_argument("--report", required=True, help="report JSON path")
    args = parser.parse_args(argv)
    try:
        report = tally(
            universe=args.universe, entries=args.entries, results=args.results,
            odds=args.odds, exclusions=args.exclusions,
            minimum_universe=args.minimum_universe, source_revision=args.source_revision,
        )
    except (ValueError, OSError, csv.Error) as exc:
        report = {"status": "FAIL", "issues": [str(exc)]}
    dest = Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "universe": report.get("actual_universe_races"),
        "eligible_7_9": report.get("eligible_7_9_car_races_with_three_modalities"),
        "issues": report["issues"][:10],
        "report": str(dest),
    }, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
