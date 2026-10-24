#!/usr/bin/env python3
"""Display-only recommendation layer for frozen Ura Haru V1.

Never modifies the frozen V1 formation, a wager, a decision, a stake, a model or
the flagship HaruQ. A star score is a descriptive historical comparison,
NOT a predicted probability or a betting decision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ura_haruq_independent.frozen_ura_haru_v1.URA_HARU_V1_FROZEN_RULE import (
    BET_YEN_PER_TICKET,
    POLICY_ID,
    formation,
    parse_line_assignments,
)

ADVISORY_ID = "URA_HARU_V1_LOW_QUARTER_DISPLAY_20261024"
PINNED_V1_COMMIT = "7adf108f33a16c5043e79b2f6cf737d9bc397095"
HISTORICAL_WINDOW = "2024-01-01 to 2026-09-30, 2026 is Jan-Sep only"
GRADE_TABLE = (
    {
        "lower": 0.0, "upper": 0.25, "stars": 3,
        "label": "標準", "key": "LOW_0.00_0.25",
        "sample_active_races": 1150, "hit_rate_pct": 14.26,
        "historical_roi_pct": [213.76, 116.68, 144.54],
        "target_band_roi_pct": [201.62, 101.09, 126.44],
        "positive_months": [12, 6, 6],
        "months_observed": [12, 12, 9],
        "reason": "全期間で黒字。ただし2025年の指定配当帯ROIは101.09%と余裕が小さい。",
    },
    {
        "lower": 0.25, "upper": 0.5, "stars": 5,
        "label": "最優先", "key": "LOW_0.25_0.50",
        "sample_active_races": 972, "hit_rate_pct": 17.49,
        "historical_roi_pct": [136.48, 176.45, 229.21],
        "target_band_roi_pct": [113.75, 156.87, 206.97],
        "positive_months": [10, 9, 8],
        "months_observed": [12, 12, 9],
        "reason": "4区間で最も高いレース的中率。指定配当帯でも各年黒字、黒字月も多い。",
    },
    {
        "lower": 0.5, "upper": 0.75, "stars": 4,
        "label": "優先", "key": "LOW_0.50_0.75",
        "sample_active_races": 733, "hit_rate_pct": 16.23,
        "historical_roi_pct": [129.81, 179.38, 147.48],
        "target_band_roi_pct": [115.17, 149.61, 129.65],
        "positive_months": [6, 9, 7],
        "months_observed": [12, 12, 9],
        "reason": "指定配当帯でも各年黒字。単発の超高額配当に依存しない比較的安定した帯。",
    },
    {
        "lower": 0.75, "upper": 1.0, "stars": 2,
        "label": "注意", "key": "LOW_0.75_1.00",
        "sample_active_races": 646, "hit_rate_pct": 15.79,
        "historical_roi_pct": [168.95, 130.72, 341.00],
        "target_band_roi_pct": [155.74, 115.38, 80.03],
        "positive_months": [7, 8, 4],
        "months_observed": [12, 12, 9],
        "reason": "2026年ROI341%は26万8,200円の1的中に依存。指定配当帯では2026年80.03%。",
    },
)


def rating_for_gap(score_gap: Any) -> dict[str, Any] | None:
    """Return a *descriptive* V1 recommendation for 0 <= gap < 1."""
    try:
        gap = float(score_gap)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(gap):
        return None
    for row in GRADE_TABLE:
        if row["lower"] <= gap < row["upper"]:
            return {
                "advisory_version": ADVISORY_ID,
                "low_quarter": row["key"],
                "score_range": [row["lower"], row["upper"]],
                "stars": row["stars"],
                "stars_text": "★" * row["stars"] + "☆" * (5 - row["stars"]),
                "recommendation": row["label"],
                "reason": row["reason"],
                "historical": {
                    "years": ["2024", "2025", "2026-Jan-Sep"],
                    "active_races": row["sample_active_races"],
                    "race_hit_rate_pct": row["hit_rate_pct"],
                    "all_payout_roi_pct": list(row["historical_roi_pct"]),
                    "target_payout_3000_to_99999_roi_pct": list(row["target_band_roi_pct"]),
                    "positive_months": list(row["positive_months"]),
                    "months_observed": list(row["months_observed"]),
                },
                "advisory_only": True,
                "not_a_probability_or_ev": True,
            }
    return None


def _normalize_bikes(data: Any) -> list[int]:
    if isinstance(data, str):
        data = data.split("-")
    if not isinstance(data, (list, tuple)):
        raise ValueError("predicted_bikes_by_rank must be a rank-ordered list or '1-2-...'")
    try:
        return [int(b) for b in data]
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid bike numbers") from exc


def _normalize_lines(data: Any) -> dict[int, str]:
    if isinstance(data, str):
        return parse_line_assignments(data)
    if not isinstance(data, dict):
        raise ValueError("line_groups_by_bike must be a dict or '1:A;2:A;...'")
    try:
        return {int(k): str(v) if v is not None else "" for k, v in data.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid line mapping") from exc


def evaluate_record(input_record: dict[str, Any]) -> dict[str, Any]:
    """Read pre-race data, invoke frozen V1 unchanged, append advisory metadata.

    This function does NOT modify input_record or decide purchase amounts.
    The fields tickets/points show the frozen V1 formation as-is.
    """
    result: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "frozen_v1_commit": PINNED_V1_COMMIT,
        "advisory_version": ADVISORY_ID,
        "advisory_only": True,
        "status": "insufficient_input",
        "recommendation": None,
        "tickets": [],
        "points": 0,
        "research_stake_yen": 0,
    }
    required = ("score_gap", "predicted_bikes_by_rank", "line_groups_by_bike")
    missing = [field for field in required if field not in input_record]
    if missing:
        result["reason"] = "発走前情報が不足: " + ", ".join(missing)
        return result
    try:
        gap = float(input_record["score_gap"])
        bikes = _normalize_bikes(input_record["predicted_bikes_by_rank"])
        lines = _normalize_lines(input_record["line_groups_by_bike"])
        if len(bikes) == 7 and 0 <= gap < 1 and any(not str(lines.get(b, "")).strip() for b in bikes):
            result["status"] = "line_data_missing"
            result["reason"] = "出走表のライン予想が未公開・不完全なため、固定V1の対象判定ができません。"
            return result
        tickets = formation(bikes, lines, gap)  # exact frozen source, never reimplement
    except (TypeError, ValueError, OverflowError) as exc:
        result["status"] = "invalid_input"
        result["reason"] = str(exc)
        return result
    if not tickets:
        result["status"] = "v1_not_applicable"
        result["reason"] = "固定V1の条件不成立（LOW、7車、2-2-2-1、上位同ライン2人）。"
        return result
    grade = rating_for_gap(gap)
    if grade is None:
        raise AssertionError("Frozen formation generated tickets outside rating gap")
    result.update({
        "status": "v1_eligible",
        "low_score_gap": gap,
        "recommendation": grade,
        "tickets": ["-".join(map(str, t)) for t in tickets],
        "points": len(tickets),
        "research_stake_yen": BET_YEN_PER_TICKET * len(tickets),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Add display-only LOW recommendation to Ura Haru V1 output")
    parser.add_argument("--input", required=True, type=Path, help="Pre-race JSON")
    parser.add_argument("--output", type=Path, help="Separate output JSON; original file is never overwritten")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        parser.error("Input must be one JSON object")
    evaluated = evaluate_record(data)
    # No record mutation, overwriting, or change of the original V1 policy.
    output = {"original_input": data, "ura_haru_v1_display": evaluated}
    serialized = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        if args.output.resolve() == args.input.resolve():
            parser.error("Overwrite of source input is prohibited")
        if args.output.exists():
            parser.error("Output path already exists (overwrite prohibited)")
        args.output.write_text(serialized, encoding="utf-8")
        print(f"URA_HARU_V1_ADVISORY_SAVED {args.output}")


if __name__ == "__main__":
    main()
