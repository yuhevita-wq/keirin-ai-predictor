#!/usr/bin/env python3
"""Reproduce haruqube's 44-feature LambdaRank using newly collected encrypted days.

The original feature definitions, model parameters, and betting thresholds are
imported unchanged. Neither the original upstream DB nor released model is used.
Production files are not modified. Partial collection days are quarantined.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import CLASS_MAP, DB_PATH, DEFAULT_GEAR_RATIO, get_bet_category
from db.schema import init_db
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from models.trainer import evaluate_test_set

OUT = Path("reproduction_outputs")
TRAIN_END = "2025-01-01"
VALID_END = "2025-07-01"
TEST_END = "2026-10-01"
MERGE_TABLES = ("riders", "races", "entries", "race_results", "race_payouts", "source_records")


def shell(args, *, to_file=None):
    if to_file:
        with open(to_file, "wb") as fp:
            proc = subprocess.run(args, stdout=fp, stderr=subprocess.PIPE, check=False)
    else:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(f"{args[0]} exited {proc.returncode}: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout


def cols(conn, schema, table):
    return [r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")]


def merge_one(conn, db_file):
    conn.execute("ATTACH DATABASE ? AS daily", (str(db_file),))
    counts = {}
    try:
        if conn.execute("PRAGMA daily.integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Source SQLite integrity check failed")
        for table in MERGE_TABLES:
            target, source = cols(conn, "main", table), cols(conn, "daily", table)
            if not source:
                if table == "source_records":
                    continue
                raise RuntimeError(f"Required table missing: {table}")
            use = [col for col in target if col in source and col not in ("id", "created_at")]
            if not use:
                continue
            names = ",".join('"' + name + '"' for name in use)
            before = conn.total_changes
            conn.execute(f"INSERT OR IGNORE INTO main.{table} ({names}) SELECT {names} FROM daily.{table}")
            counts[table] = conn.total_changes - before
        conn.commit()
    finally:
        conn.rollback()
        conn.execute("DETACH DATABASE daily")
    return counts


def prepare_db():
    secret = os.getenv("COLLECTOR_PASSPHRASE", "")
    if len(secret) < 32:
        raise RuntimeError("GitHub Actions COLLECTOR_PASSPHRASE is required")
    if DB_PATH.exists():
        raise RuntimeError("Refusing to overwrite existing database")
    DB_PATH.parent.mkdir(exist_ok=True)
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("CREATE TABLE IF NOT EXISTS source_records (race_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    branches = shell(["git", "for-each-ref", "--format=%(refname:short)",
                      "refs/remotes/origin/collected-data-*"]).decode().splitlines()
    branches = sorted(x for x in branches if re.fullmatch(r"origin/collected-data-20\d\d-\d\d", x))
    if len(branches) < 24:
        raise RuntimeError(f"Only {len(branches)} monthly branches fetched; expected at least 24")
    manifest = {"branches": len(branches), "days_complete": 0, "days_partial": 0,
                "days_without_report": 0, "races_imported": 0, "entries_imported": 0,
                "results_imported": 0, "payouts_imported": 0, "days": []}
    with tempfile.TemporaryDirectory(prefix="haruqube-shard-") as td:
        encrypted = Path(td) / "daily.gpg"
        decrypted = Path(td) / "daily.sqlite"
        for branch in branches:
            names = shell(["git", "ls-tree", "-r", "--name-only", branch, "encrypted/"]).decode().splitlines()
            for name in sorted(names):
                found = re.fullmatch(r"encrypted/netkeirin-(20\d\d-\d\d-\d\d)\.sqlite\.gpg", name)
                if not found:
                    continue
                day = found.group(1)
                if day < "2023-01-01" or day >= "2026-10-23":
                    continue
                try:
                    report = json.loads(shell(["git", "show", f"{branch}:reports/netkeirin-{day}.json"]))
                except Exception:
                    manifest["days_without_report"] += 1
                    continue
                if report.get("date") != day or report.get("status") != "complete":
                    manifest["days_partial"] += 1
                    continue
                shell(["git", "show", f"{branch}:{name}"], to_file=encrypted)
                proc = subprocess.run(
                    ["gpg", "--batch", "--yes", "--pinentry-mode", "loopback",
                     "--passphrase-fd", "0", "--output", str(decrypted),
                     "--decrypt", str(encrypted)], input=secret.encode(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
                if proc.returncode:
                    raise RuntimeError(f"Decryption failed for {day}; plaintext was not uploaded")
                imported = merge_one(con, decrypted)
                if imported.get("races", 0) != report["races"]:
                    raise RuntimeError(f"Race count mismatch for {day}: {imported} vs {report['races']}")
                manifest["days_complete"] += 1
                for key, table in (("races_imported", "races"), ("entries_imported", "entries"),
                                   ("results_imported", "race_results"), ("payouts_imported", "race_payouts")):
                    manifest[key] += imported.get(table, 0)
                manifest["days"].append({"date": day, "races": imported["races"]})
                encrypted.unlink(missing_ok=True)
                decrypted.unlink(missing_ok=True)
                if manifest["days_complete"] % 50 == 0:
                    print("IMPORT", manifest["days_complete"], manifest["races_imported"], flush=True)
    con.commit()
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("Merged SQLite integrity check failed")
    con.close()
    if manifest["days_complete"] < 300 or manifest["races_imported"] < 10000:
        raise RuntimeError("Too few complete newly-collected races")
    (OUT / "collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("MERGE_OK", manifest["days_complete"], manifest["races_imported"], flush=True)
    return manifest


def dataset():
    conn = sqlite3.connect(DB_PATH)
    valid = {x[0] for x in conn.execute("""
        SELECT r.race_id FROM races r JOIN race_results rr ON rr.race_id=r.race_id
        GROUP BY r.race_id
        HAVING COUNT(*)=r.rider_count AND COUNT(*)>=5
          AND SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
          AND SUM(CASE WHEN rr.finish_position IS NULL THEN 1 ELSE 0 END)=0
          AND COUNT(DISTINCT rr.finish_position)=COUNT(*)
    """)}
    gear = pd.read_sql_query("SELECT race_id,rider_id,gear_ratio AS source_gear FROM entries", conn)
    race_meta = pd.read_sql_query("""SELECT race_id, grade, date, nisyatan_combo, nisyatan_payout
        FROM races LEFT JOIN race_payouts USING (race_id)""", conn)
    conn.close()
    builder = FeatureBuilder()
    features = builder.feature_names
    if len(features) != 44 or len(set(features)) != 44:
        raise RuntimeError("Upstream 44-feature contract changed")
    df = builder.build_dataset(2023, 2026)
    df = df[df["race_id"].isin(valid)].copy()
    # Avoid using a rider's later master-level class in earlier races;
    # use that race's historical result, without adding any features.
    df["rider_class_num"] = df["class"].map(CLASS_MAP).fillna(6).astype(float)
    df["class_rank_in_race"] = df.groupby("race_id")["rider_class_num"].rank(method="min")
    df["rider_class_x_jiku"] = df["rider_class_num"] * df["line_is_jiku"]
    df = df.merge(gear, on=["race_id", "rider_id"], how="left", validate="many_to_one")
    df["entry_gear_ratio"] = pd.to_numeric(df["source_gear"], errors="coerce").fillna(DEFAULT_GEAR_RATIO)
    df.drop(columns=["source_gear"], inplace=True)
    df = df.merge(race_meta, on="race_id", how="left", validate="many_to_one")
    if df["race_id"].nunique() < 10000:
        raise RuntimeError("Too few quality-valid races")
    return df, features, len(valid)


def split(frame, features):
    frame = frame.sort_values(["race_date", "race_id", "bike_number"], kind="stable").reset_index(drop=True)
    x = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = frame["finish_position"]
    group = frame.groupby("race_id", sort=False).size().tolist()
    if sum(group) != len(frame):
        raise RuntimeError("Invalid LambdaRank group sizes")
    return frame, x, y, group


def eval_bets(frame, model, features):
    df = frame.copy()
    df["score"] = model.predict(df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)).values
    bets = []
    missing = 0
    eligible = 0
    for race_id, race in df.groupby("race_id", sort=False):
        ordered = race.sort_values(["score", "bike_number"], ascending=[False, True], kind="stable")
        if len(ordered) < 5:
            continue
        top = ordered.iloc[0]
        gap = float(top["score"] - ordered.iloc[1]["score"])
        grade = str(top["grade"])
        _label, _text, stake_each = get_bet_category(gap, grade)
        if stake_each <= 0:
            continue
        eligible += 1
        winning = race.sort_values("finish_position", kind="stable")
        actual = tuple(int(v) for v in winning.head(2)["bike_number"])
        tickets = [(int(top["bike_number"]), int(x)) for x in ordered.iloc[1:5]["bike_number"]]
        assert len(set(tickets)) == 4
        nums = [int(x) for x in re.findall(r"\d+", str(top["nisyatan_combo"]))]
        payout = top["nisyatan_payout"]
        if len(nums) != 2 or not pd.notna(payout) or tuple(nums) != actual or float(payout) < 100:
            missing += 1
            continue
        hit = int(actual in tickets)
        returned = float(payout) * (stake_each / 100.0) if hit else 0.0
        bets.append({"race_id": race_id, "date": str(top["race_date"]), "grade": grade,
                     "score_gap": round(gap, 5),
                     "tickets": ";".join(f"{a}>{b}" for a, b in tickets),
                     "actual": f"{actual[0]}>{actual[1]}", "stake": stake_each * 4,
                     "returned": round(returned), "hit": hit})
    with (OUT / "holdout_exacta_tickets.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["race_id", "date", "grade", "score_gap", "tickets", "actual",
                  "stake", "returned", "hit"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bets)
    spent = sum(row["stake"] for row in bets)
    earned = sum(row["returned"] for row in bets)
    return {"eligible_races": eligible, "settled_races": len(bets),
            "unverifiable_payouts_excluded": missing,
            "hits": sum(row["hit"] for row in bets),
            "stake_yen": spent, "return_yen": earned,
            "roi_percent": round(100 * earned / spent, 3) if spent else None,
            "online_torigami_filter_applied": False,
            "historical_line_bonus_applied": False,
            "strategy": "unchanged config.get_bet_category; 4 exacta tickets"}


def main():
    OUT.mkdir(exist_ok=True)
    manifest = prepare_db()
    df, features, valid_count = dataset()
    train = df[df["race_date"] < TRAIN_END]
    val = df[(df["race_date"] >= TRAIN_END) & (df["race_date"] < VALID_END)]
    test = df[(df["race_date"] >= VALID_END) & (df["race_date"] < TEST_END)]
    train, xtr, ytr, gtr = split(train, features)
    val, xv, yv, gv = split(val, features)
    test, xt, yt, gt = split(test, features)
    if min(len(gtr), len(gv), len(gt)) < 1000:
        raise RuntimeError("Train, validation, or holdout set too small")
    print("SPLITS", len(gtr), len(gv), len(gt), flush=True)
    model = LGBMRanker()
    model.train(xtr, ytr, gtr, xv, yv, gv)
    report = {
        "model": "original haruqube LightGBM LambdaRank",
        "features": features,
        "data_source": "newly collected SQLite shards only; no public upstream DB",
        "splits": {"training": len(gtr), "validation": len(gv), "holdout": len(gt)},
        "data_quality": {"valid_races": valid_count,
                         "prepared_races": df["race_id"].nunique(),
                         "merged_races": manifest["races_imported"]},
        "validation_metrics": evaluate_test_set(val, model, features),
        "holdout_metrics": evaluate_test_set(test, model, features),
        "holdout_exacta": eval_bets(test, model, features),
        "caveats": [
            "2023-start data lacks 2022 rider history",
            "original model exact seed and training split are not published",
            "original online line bonus and torigami odds filter unavailable in offline ROI"
        ]
    }
    model_file = OUT / "haruqube_fresh_44_model.pkl"
    model.save(str(model_file))
    report["model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
    (OUT / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL_REPORT", json.dumps({
        "holdout_metrics": report["holdout_metrics"],
        "holdout_exacta": report["holdout_exacta"]
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
