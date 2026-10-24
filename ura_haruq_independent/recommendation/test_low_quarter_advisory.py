"""Regression tests: recommendation is metadata only, frozen V1 has not changed."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from ura_haruq_independent.frozen_ura_haru_v1.URA_HARU_V1_FROZEN_RULE import formation
from ura_haruq_independent.recommendation.low_quarter_advisory import (
    ADVISORY_ID, evaluate_record, rating_for_gap,
)

BASIC = {
    "predicted_bikes_by_rank": "3-5-1-2-6-7-4",
    "line_groups_by_bike": "3:A;5:A;1:B;2:B;6:C;7:C;4:D",
    "score_gap": 0.37,
    "race_id": "sample",
    "bets": {"original": "unchanged"},
    "total_yen": 999,
}

class LowQuarterAdvisoryTests(unittest.TestCase):
    def test_exact_quarter_boundaries(self):
        cases = ((0.0, 3), (0.24999999, 3), (0.25, 5),
                 (0.49999999, 5), (0.50, 4), (0.74999999, 4),
                 (0.75, 2), (0.999999, 2))
        for score, want in cases:
            with self.subTest(score=score):
                self.assertEqual(rating_for_gap(score)["stars"], want)
        for gap in (-0.001, 1, 1.001, None, "unknown", float("nan"), float("inf")):
            self.assertIsNone(rating_for_gap(gap))

    def test_frozen_core_checksum(self):
        path = Path(__file__).resolve().parents[1] / "frozen_ura_haru_v1" / "URA_HARU_V1_FROZEN_RULE.py"
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(h, "b0891724787f3c300a2b58d0d19cc6cf4ff421225ea53c81b94ab3add7194818")

    def test_advisory_keeps_frozen_bets_exactly_and_no_mutation(self):
        source = copy.deepcopy(BASIC)
        frozen = formation([3, 5, 1, 2, 6, 7, 4],
                          {3:"A",5:"A",1:"B",2:"B",6:"C",7:"C",4:"D"}, .37)
        evaluated = evaluate_record(source)
        self.assertEqual(evaluated["status"], "v1_eligible")
        self.assertEqual(evaluated["advisory_version"], ADVISORY_ID)
        self.assertEqual(evaluated["points"], 6)
        self.assertEqual(evaluated["research_stake_yen"], 600)
        self.assertEqual(evaluated["tickets"], ["-".join(map(str, t)) for t in frozen])
        self.assertEqual(evaluated["recommendation"]["stars"], 5)
        self.assertEqual(evaluated["recommendation"]["historical"]["target_payout_3000_to_99999_roi_pct"],
                         [113.75, 156.87, 206.97])
        self.assertEqual(source, BASIC)
        self.assertEqual(source["total_yen"], 999)
        self.assertEqual(source["bets"], {"original":"unchanged"})
        self.assertEqual(len(json.dumps(evaluated, ensure_ascii=False)), len(json.dumps(json.loads(json.dumps(evaluated, ensure_ascii=False)), ensure_ascii=False)))

    def test_research_only_no_rating_if_not_eligible(self):
        for gap in (-0.01, 1.0, None):
            source = {**BASIC, "score_gap": gap}
            result = evaluate_record(source)
            self.assertEqual(result["status"], "v1_not_applicable")
            self.assertIsNone(result["recommendation"])
            self.assertEqual(result["points"], 0)
        lines = "3:A;5:B;1:C;2:A;6:B;7:C;4:D"
        result = evaluate_record({**BASIC, "line_groups_by_bike": lines})
        self.assertEqual(result["status"], "v1_not_applicable")
        self.assertEqual(result["points"], 0)

    def test_missing_or_bad_input_never_guess(self):
        self.assertEqual(evaluate_record({})["status"], "insufficient_input")
        self.assertEqual(evaluate_record({**BASIC, "score_gap": "bad"})["status"], "invalid_input")
        self.assertEqual(evaluate_record({**BASIC, "line_groups_by_bike": ""})["status"], "v1_not_applicable")

if __name__ == "__main__":
    unittest.main()
