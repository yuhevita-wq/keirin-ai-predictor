"""Unit tests for the explicitly adopted 2026-10-26 Ura V2 selector."""
import unittest

from ura_haruq_independent.legacy.adopted_v2 import (
    BASE_POLICY_ID, MAX_COMPETITION_SCORE_SPREAD, POLICY_ID, decide,
)
from ura_haruq_independent.frozen_ura_haru_v1.URA_HARU_V1_FROZEN_RULE import formation


class AdoptedParticipationTests(unittest.TestCase):
    bikes = [2, 6, 5, 1, 4, 3, 7]
    lines = {2: "A", 6: "A", 5: "B", 1: "B", 4: "C", 3: "C", 7: "D"}
    scores = {2: 82.00, 6: 84.54, 5: 82.5, 1: 84.1, 4: 83.2, 3: 83, 7: 82.7}

    def test_core_unmodified_when_selector_accepts(self):
        r = decide(self.bikes, self.lines, 0.56417727, self.scores)
        expected = ["-".join(map(str, t)) for t in formation(self.bikes, self.lines, 0.56417727)]
        self.assertEqual(r["base_policy_id"], BASE_POLICY_ID)
        self.assertEqual(r["policy_id"], POLICY_ID)
        self.assertEqual(r["decision"], "BET")
        self.assertEqual(r["tickets"], expected)
        self.assertEqual(r["base_tickets"], expected)
        self.assertEqual(r["points"], 6)
        self.assertEqual(r["stake_yen"], 600)
        self.assertEqual(r["skill_spread"], 2.54)

    def test_score_gap_boundary_is_unchanged(self):
        r = decide(self.bikes, self.lines, 1.0, self.scores)
        self.assertEqual(r["decision"], "NO_BET")
        self.assertEqual(r["reason"], "BASE_V1_INELIGIBLE")
        self.assertEqual(r["base_tickets"], [])

    def test_skill_boundary_inclusive(self):
        scores = {1: 86.0, 2: 78.9, 3: 82.0, 4: 82.2,
                  5: 85.2, 6: 80.1, 7: 84.0}
        r = decide(self.bikes, self.lines, 0.1, scores)
        self.assertEqual(r["skill_spread"], 7.1)
        self.assertEqual(r["decision"], "BET")

    def test_over_limit_preserves_base_but_suppresses_purchase(self):
        scores = dict(self.scores)
        scores[3] = 100.1
        r = decide(self.bikes, self.lines, 0.1, scores)
        self.assertEqual(r["decision"], "NO_BET")
        self.assertEqual(r["reason"], "SKILL_SPREAD_EXCEEDS_7_1")
        self.assertEqual(r["points"], 0)
        self.assertEqual(r["stake_yen"], 0)
        self.assertEqual(len(r["base_tickets"]), 6)
        self.assertEqual(r["tickets"], [])

    def test_no_line_pair_means_no_bet(self):
        bikes = [2, 5, 4, 6, 1, 3, 7]
        r = decide(bikes, self.lines, 0.4, self.scores)
        self.assertEqual(r["decision"], "NO_BET")
        self.assertEqual(r["reason"], "BASE_V1_INELIGIBLE")

    def test_missing_score_is_not_interpreted_as_zero(self):
        invalid = dict(self.scores)
        del invalid[2]
        r = decide(self.bikes, self.lines, 0.1, invalid)
        self.assertEqual(r["reason"], "COMPETITION_SCORES_INVALID")
        self.assertEqual(r["tickets"], [])

    def test_nan_score_is_not_accepted(self):
        scores = dict(self.scores)
        scores[2] = float("nan")
        r = decide(self.bikes, self.lines, 0.1, scores)
        self.assertEqual(r["reason"], "COMPETITION_SCORES_INVALID")

    def test_third_score_absolute_threshold_is_removed(self):
        # Previous research selector excluded this due to 3rd candidates around 82.
        r = decide(self.bikes, self.lines, 0.1, self.scores)
        self.assertEqual(r["decision"], "BET")
        self.assertLess(max(self.scores.values()), 95.8767)

    def test_constant_threshold(self):
        self.assertEqual(MAX_COMPETITION_SCORE_SPREAD, 7.1)


if __name__ == "__main__":
    unittest.main()

