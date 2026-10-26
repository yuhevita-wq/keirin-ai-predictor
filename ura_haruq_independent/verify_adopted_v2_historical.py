#!/usr/bin/env python3
"""Reproduce adopted V2's saved historic betting arithmetic, WITHOUT refitting.

Consumes pre-generated frozen-V1 eligible-races CSV, including pre-race skill
spread and saved official payout. No payout may enter the participation rule.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ura_haruq_independent.adopted_v2 import MAX_COMPETITION_SCORE_SPREAD, POLICY_ID

GOLDEN = {
    "2024": {"races":313,"investment_yen":187800,"return_yen":419090,"hits":52},
    "2025": {"races":291,"investment_yen":174600,"return_yen":371080,"hits":48},
    "2026": {"races":209,"investment_yen":125400,"return_yen":280200,"hits":35},
}


def replay(path):
    report=defaultdict(lambda:{"races":0,"investment_yen":0,"return_yen":0,"hits":0})
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        data=list(csv.DictReader(f))
    if len(data)!=3501 or len(set(r["race_id"] for r in data))!=len(data):
        raise ValueError("FROZEN_V1_HISTORY_CARDINALITY_MISMATCH")
    for r in data:
        if r["forecast_year"] not in GOLDEN:
            raise ValueError("WRONG_FORECAST_YEAR")
        if r.get("skill_spread") in (None,""):
            continue  # no verified pre-race competition score for all seven
        if round(float(r["skill_spread"]),4)>MAX_COMPETITION_SCORE_SPREAD:
            continue
        # V1's tickets/stake stay exactly as already recorded. Result only grades
        # the fixed decisions after selection, never informs the selection.
        if int(r["points"])!=6 or int(r["cost_yen"])!=600:
            raise AssertionError("UNEXPECTED_BASE_V1_FORMATION_OR_STAKE")
        x=report[r["forecast_year"]]
        x["races"]+=1
        x["investment_yen"]+=int(r["cost_yen"])
        x["return_yen"]+=int(r["return_yen"])
        x["hits"]+=int(r["hit"])
    result={}
    for year in sorted(GOLDEN):
        r=report[year]
        if r!=GOLDEN[year]:
            raise AssertionError("GOLDEN_HISTORICAL_BACKTEST_CHANGED "+year+" "+str(r))
        result[year]={**r,"roi_pct":round(r["return_yen"]/r["investment_yen"]*100,4),
                      "profit_yen":r["return_yen"]-r["investment_yen"]}
    return {"policy":POLICY_ID,"race_rows_audited":len(data),
            "scope":"retrospective same known 2024–2026 dataset, not newly blind",
            "yearly":result}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--historical-csv",required=True)
    args=ap.parse_args()
    print(json.dumps(replay(args.historical_csv),ensure_ascii=False,indent=2))
