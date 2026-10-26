#!/usr/bin/env python3
"""Official Ura-HaruQ V2 participation gate, adopted 2026-10-26.

Runs AFTER the unchanged frozen Ura V1 ticket formation.
It does not predict ranks, create alternative tickets, or modify stake sizing.
The only new purchase constraint is pre-race seven-rider competition-score
(max - min) <= 7.1. Historical ROI is descriptive, not guaranteed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping

from ura_haruq_independent.frozen_ura_haru_v1.URA_HARU_V1_FROZEN_RULE import (
    BET_YEN_PER_TICKET,
    POLICY_ID as BASE_POLICY_ID,
    formation,
)

POLICY_ID = "URA_HARU_V2_SKILL_SPREAD_7_1_20261026"
MAX_COMPETITION_SCORE_SPREAD = 7.1
BET_TYPES = ("3連単",)


def decide(predicted_bikes_by_rank, line_groups_by_bike, score_gap,
           competition_scores_by_bike):
    """Return an auditable BET / NO_BET; never infer missing scores.

    Args:
      predicted_bikes_by_rank: sequence in descending *independent ranker* score order.
      line_groups_by_bike: {bike: line identifier} from the pre-race card.
      score_gap: ranker score[0] - score[1] (must be the saved model's scale).
      competition_scores_by_bike: {bike: pre-race competition score}.
    """
    bikes = tuple(int(x) for x in predicted_bikes_by_rank)
    base_tickets = formation(bikes, line_groups_by_bike, score_gap)
    result = {
        "policy_id": POLICY_ID,
        "base_policy_id": BASE_POLICY_ID,
        "base_tickets": ["-".join(map(str, t)) for t in base_tickets],
        "decision": "NO_BET",
        "reason": "BASE_V1_INELIGIBLE",
        "skill_spread": None,
        "max_skill_spread": MAX_COMPETITION_SCORE_SPREAD,
        "bet_type": "3連単",
        "tickets": [],
        "points": 0,
        "stake_yen": 0,
        "yen_per_ticket": BET_YEN_PER_TICKET,
    }
    if not base_tickets:
        return result
    if len(bikes) != 7 or not isinstance(competition_scores_by_bike, Mapping):
        result["reason"] = "COMPETITION_SCORES_UNAVAILABLE"
        return result
    try:
        normalized = {int(k): float(v) for k, v in competition_scores_by_bike.items()}
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "COMPETITION_SCORES_INVALID"
        return result
    if set(normalized) != set(bikes) or any(not math.isfinite(v) for v in normalized.values()):
        result["reason"] = "COMPETITION_SCORES_INVALID"
        return result
    # Four decimals mirrors the saved 3,501-row backtest's per-race skill_spread.
    spread = round(max(normalized.values()) - min(normalized.values()), 4)
    result["skill_spread"] = spread
    if spread > MAX_COMPETITION_SCORE_SPREAD:
        result["reason"] = "SKILL_SPREAD_EXCEEDS_7_1"
        return result
    if len(base_tickets) != 6:
        raise RuntimeError("FROZEN_V1_TICKET_COUNT_INVARIANT_FAILED")
    result.update(
        decision="BET",
        reason="ADOPTED_V2_ELIGIBLE",
        tickets=result["base_tickets"].copy(),
        points=len(base_tickets),
        stake_yen=BET_YEN_PER_TICKET * len(base_tickets),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Official Ura-HaruQ V2 filter over frozen V1.")
    parser.add_argument("--input",required=True,help="JSON file of racecard + independent ranker forecast")
    parser.add_argument("--out",help="Output JSON path (default: stdout only)")
    args = parser.parse_args()
    from pathlib import Path
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(payload,dict) or not isinstance(payload.get("races"),list):
        raise ValueError("Expected {'races': [...]} and independently calculated forecasts")
    output = {"policy_id": POLICY_ID, "source": "fixed V1 + as-of racecard score spread", "races": []}
    for row in payload["races"]:
        if not all(k in row for k in ("race_id","predicted_bikes_by_rank",
                                      "line_groups_by_bike","score_gap",
                                      "competition_scores_by_bike")):
            raise ValueError("Missing required pre-race feature(s): "+str(row.get("race_id")))
        decision = decide(row["predicted_bikes_by_rank"],row["line_groups_by_bike"],
                          row["score_gap"],row["competition_scores_by_bike"])
        output["races"].append({"race_id":str(row["race_id"]),**decision})
    text = json.dumps(output,ensure_ascii=False,indent=2)+"\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(text,encoding="utf-8")
    print(text,end="")


if __name__ == "__main__":
    main()
