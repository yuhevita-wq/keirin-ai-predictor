#!/usr/bin/env python3
"""Read-only, database-free boot adapter for kiseki's original Type Lab.

This adapter contains no reimplementation of the A-F rules or ticket strategy.
Provide PRE-RACE model forecasts; it never reads results or submits a bet.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _number(value, label, lower=None, upper=None):
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: numeric value required") from exc
    if not math.isfinite(x) or (lower is not None and x < lower) or (upper is not None and x > upper):
        raise ValueError(f"{label}: invalid value {value!r}")
    return x


def load_original(source_root: str):
    root = Path(source_root).resolve()
    path = root / "src" / "type_lab.py"
    if not path.is_file():
        raise FileNotFoundError(f"original src/type_lab.py not found: {path}")
    # Prevent accidental import of the research repo's modules or a globally installed src.
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    module = importlib.import_module("src.type_lab")
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("source module did not resolve to the pinned original")
    return module


def source_revision(source_root: str):
    root = Path(source_root).resolve()
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=3
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _board(items, kind, n, field):
    if items is None:
        return None
    if not isinstance(items, list):
        raise ValueError(f"{kind}.{field} must be an array")
    expected = n * (n - 1) * (n - 2) if kind == "trifecta" else math.comb(n, 3)
    if len(items) != expected:
        raise ValueError(f"{kind}.{field}: expected full {expected}-combination board, got {len(items)}")
    result = {}
    for item in items:
        combo = item["combo"]
        if len(combo) != 3 or len(set(combo)) != 3:
            raise ValueError(f"{kind}: three distinct car numbers required")
        cars = tuple(int(c) for c in combo)
        if any(c < 1 or c > n for c in cars):
            raise ValueError(f"{kind}: invalid car number")
        key = cars if kind == "trifecta" else frozenset(cars)
        if key in result:
            raise ValueError(f"{kind}: duplicate combination {combo}")
        result[key] = _number(item[field], f"{kind}.{field}", lower=0)
        if result[key] <= 0 and field == "odds":
            raise ValueError(f"{kind}: odds must be positive")
    return result


def run(payload: dict, original):
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) not in (7, 9):
        raise ValueError("entries: a 7-car or 9-car complete lineup is required")
    n = len(entries)
    car_ids = [int(e["car"]) for e in entries]
    if sorted(car_ids) != list(range(1, n + 1)):
        raise ValueError("car numbers must be distinct and cover 1..N")

    day = int(payload["day_index"])
    if day not in (1, 2, 3, 4):
        raise ValueError("day_index must be 1..4")
    p3 = {int(e["car"]): _number(e["p3"], "p3", 0, 1) for e in entries}
    p1 = {int(e["car"]): _number(e["p1"], "p1", 0, 1) for e in entries}
    group = {int(e["car"]): e.get("group") for e in entries}
    pos = {int(e["car"]): e.get("position") for e in entries}
    style = {int(e["car"]): str(e.get("style") or "") for e in entries}
    score = {int(e["car"]): _number(e["score"], "score") for e in entries}
    behind = {int(e["car"]): _number(e["behind_pct"], "behind_pct", 0, 100) for e in entries}
    shape = original.race_shape(p3, group, pos, style, score, behind, day, win_probs=p1)
    if shape is None:
        raise ValueError("race_shape returned no classification")

    odds_data = payload.get("predicted_odds") or {}
    probs_data = payload.get("combination_probabilities") or {}
    boards = {}
    for kind in ("trifecta", "trio"):
        o = _board(odds_data.get(kind), kind, n, "odds")
        p = _board(probs_data.get(kind), kind, n, "prob")
        if (o is None) != (p is None):
            raise ValueError(f"{kind}: odds and probabilities must both be provided")
        if o is not None and set(o) != set(p):
            raise ValueError(f"{kind}: odds/probability combination mismatch")
        boards[kind] = (o, p) if o is not None else None

    # For A, the original selection requires the optional 2-ticket trio gate.
    trio_ok = None
    if shape.type_label == "A" and n == 7 and boards["trio"]:
        trio_odds, trio_prob = boards["trio"]
        trial = original.build_with_gate_fallback(
            shape, original.PLANS["A_trio"], trio_odds, trio_prob, n_entries=n
        )
        trio_ok = bool(
            trial and original.mean_expected_payout(trial[1], trio_odds) > original.MIN_MEAN_PAYOUT
            and all(trio_odds[k] >= original.MIN_POINT_ODDS for k in trial[1])
        )
    selected = original.sell_plans_for(
        shape.type_label, n, str(payload.get("race_type", "")),
        pw_ent=shape.pw_ent, trio_ok=trio_ok, axis_sum=shape.axis_sum,
    )
    if len(selected) > 1:
        raise AssertionError("original sell_plans_for unexpectedly returned multiple plans")
    base = {
        "research_only": True,
        "bets_submitted": False,
        "race_id": str(payload.get("race_id", "sample")),
        "type": shape.type_label,
        "axis_sum": shape.axis_sum,
        "arare": shape.arare,
        "p3_order": list(shape.order),
        "win_entropy": shape.pw_ent,
        "rule_version": original.rule_version(n),
        "status": "no_candidate",
        "plan": None,
        "legs": [],
    }
    if not selected:
        return base
    plan = selected[0]
    if boards[plan.bet_type] is None:
        raise ValueError(f"missing full pre-race {plan.bet_type} predicted odds/probabilities board")
    odds, probs = boards[plan.bet_type]
    built = original.build_with_gate_fallback(shape, plan, odds, probs, n_entries=n)
    if not built:
        return {**base, "status": "unbuildable", "plan": plan.key}
    legs, stakes, used = built
    if not stakes or any(stake <= 0 or stake % original.UNIT for stake in stakes.values()):
        raise AssertionError("invalid allocation returned by original Type Lab")
    if sum(stakes.values()) > original.BUDGET:
        raise AssertionError("allocation exceeds original Type Lab budget")
    tickets = [
        {
            "combo": sorted(key) if used.bet_type == "trio" else list(key),
            "stake_yen": int(stakes[key]),
            "predicted_odds": odds[key],
            "model_probability": probs[key],
        }
        for key in legs if key in stakes
    ]
    if len(tickets) != len(stakes):
        raise AssertionError("allocated combinations and returned legs disagree")
    mean_payout = original.mean_expected_payout(stakes, odds)
    # This is ONLY a common payout-price gate; production also checks lineup,
    # axis reliability, daily quotas and special-date exceptions.
    common_gate = (
        mean_payout > original.MIN_MEAN_PAYOUT
        and all(odds[key] >= original.MIN_POINT_ODDS for key in stakes)
    )
    return {
        **base,
        "status": "candidate" if common_gate else "paper_gate_fail",
        "plan": used.key,
        "bet_type": used.bet_type,
        "stake_total_yen": sum(stakes.values()),
        "predicted_mean_payout_yen": mean_payout,
        "common_price_gate_pass": common_gate,
        "production_submission_gate_checked": False,
        "legs": tickets,
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only Type Lab v0 smoke/analysis")
    parser.add_argument("--source-root", default=os.environ.get("TYPE_LAB_SOURCE", "external/kiseki/keirin"))
    parser.add_argument("--input", required=True, help="pre-race JSON; see README")
    parser.add_argument("--output", help="optional research output JSON")
    args = parser.parse_args()
    try:
        model = load_original(args.source_root)
        request = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = run(request, model)
        result["original_source_commit"] = source_revision(args.source_root)
        result["original_source_file"] = str(Path(model.__file__).resolve())
        output = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(output)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, AssertionError) as exc:
        parser.exit(2, f"Type Lab v0 input/boot error: {exc}\n")


if __name__ == "__main__":
    main()
