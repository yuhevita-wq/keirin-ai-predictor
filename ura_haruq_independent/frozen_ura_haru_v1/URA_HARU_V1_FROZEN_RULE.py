#!/usr/bin/env python3
"""URA-HARU V1: Adopted 2026-10-24. Read-only purchase-pattern definition.

RESEARCH BASELINE ONLY. Does not run purchases or touch flagship HaruQ.
Input must be pre-race information. Historical settlement is used in tests only.
"""
from __future__ import annotations
from collections import Counter
from itertools import permutations

POLICY_ID = "URA_HARU_V1_20261024"
BASE_RACE_PATTERN = (2, 2, 2, 1)
MIN_LOW_SCORE_GAP = 0.0
MAX_LOW_SCORE_GAP_EXCLUSIVE = 1.0
BET_YEN_PER_TICKET = 100
PREDICTED_THIRD_RANKS_ONE_BASED = (5, 6, 7)


def formation(predicted_bikes_by_rank, line_groups_by_bike, score_gap):
    """Return sorted 3連単 tuples of bicycle numbers, without results or odds.

    The fixed rule is:
    - 7 riders, all seven line assignments known, line sizes 2-2-2-1
    - the *independent* forecast score difference [0,1)
    - two distinct riders within predicted ranks 1-3, from the same line,
      placed in both 1-2 and 2-1 orders
    - third place is a predicted-rank-5/6/7 rider from a different line
    - if no matching pair, abstain; never duplicate a ticket
    """
    bikes = [int(b) for b in predicted_bikes_by_rank]
    if len(bikes) != 7 or len(set(bikes)) != 7:
        return ()
    if not 0 <= float(score_gap) < MAX_LOW_SCORE_GAP_EXCLUSIVE:
        return ()
    lines = {int(b): str(g).strip() for b, g in line_groups_by_bike.items()}
    if any(b not in lines or not lines[b] for b in bikes):
        return ()
    if tuple(sorted(Counter(lines[b] for b in bikes).values(), reverse=True)) != BASE_RACE_PATTERN:
        return ()
    selected = set()
    for v, w in permutations(bikes[:3], 2):
        if lines[v] != lines[w]:
            continue
        for c in bikes[4:]:
            if lines[c] != lines[v]:
                selected.add((v, w, c))
    return tuple(sorted(selected))


def parse_line_assignments(serialized: str):
    """Parse saved entries line field: '1:A;2:B;...' (pre-race only)."""
    result = {}
    for pair in str(serialized).split(';'):
        number, sep, group = pair.partition(':')
        if sep and number.strip():
            result[int(number)] = group
    return result


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranked-bikes', required=True,
                        help='e.g. 3-5-1-2-6-7-4, in predicted rank order')
    parser.add_argument('--lines', required=True,
                        help='e.g. 3:A;5:A;1:B;2:B;6:C;7:C;4:D')
    parser.add_argument('--gap', required=True, type=float)
    args = parser.parse_args()
    bets = formation(args.ranked_bikes.split('-'), parse_line_assignments(args.lines), args.gap)
    print(json.dumps({'policy': POLICY_ID, 'tickets': ['-'.join(map(str, t)) for t in bets],
                      'points': len(bets), 'research_stake_yen': len(bets) * BET_YEN_PER_TICKET},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
