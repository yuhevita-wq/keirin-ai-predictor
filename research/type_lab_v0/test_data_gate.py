"""Fail-closed data connection tests. Tiny fixtures ONLY verify the gate, never train."""
import csv
import tempfile
import unittest
from pathlib import Path

from data_gate import REFERENCE_FULL_ARCHIVE_RACES, tally


def write_table(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


class FullDataGateTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.params = {
            "universe": str(self.root / "universe.csv"),
            "entries": str(self.root / "entries.csv"),
            "results": str(self.root / "results.csv"),
            "odds": str(self.root / "odds.csv"),
            "source_revision": "a" * 40,
            "minimum_universe": 3,
        }
        write_table(self.params["universe"], ["race_id", "race_date", "entry_count", "meeting_grade"], [
            dict(race_id="one", race_date="2023-01-01", entry_count=7, meeting_grade="F1"),
            dict(race_id="two", race_date="2024-01-01", entry_count=9, meeting_grade="G3"),
            dict(race_id="three", race_date="2025-01-01", entry_count=5, meeting_grade="F2"),
        ])
        write_table(self.params["entries"], ["race_id", "car"], [
            dict(race_id=rid, car=i)
            for rid, n in [("one", 7), ("two", 9), ("three", 5)]
            for i in range(1, n + 1)
        ])
        write_table(self.params["results"], ["race_id", "first", "second", "third"], [
            dict(race_id=rid, first=1, second=2, third=3) for rid in ("one", "two", "three")
        ])
        write_table(self.params["odds"], ["race_id", "bet_type"], [
            dict(race_id=rid, bet_type=kind)
            for rid in ("one", "two", "three")
            for kind in ("3rentan", "3renpuku", "2shatan", "2shafuku", "wide")
        ])

    def test_full_fixture_is_a_valid_census_only(self):
        report = tally(**self.params)
        self.assertEqual(report["status"], "PASS", report["issues"])
        self.assertEqual(report["actual_universe_races"], 3)
        self.assertEqual(report["original_engine_7_9_car_races"], 2)
        self.assertEqual(report["eligible_7_9_car_races_with_three_modalities"], 2)
        self.assertEqual(report["not_eligible_original_engine_5_6_8_car_races"], 1)

    def test_default_cannot_pretend_three_is_full_archive(self):
        self.params["minimum_universe"] = REFERENCE_FULL_ARCHIVE_RACES
        report = tally(**self.params)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("below minimum" in msg for msg in report["issues"]))

    def test_partial_odds_is_not_silent_success(self):
        write_table(self.params["odds"], ["race_id", "bet_type"], [
            dict(race_id="one", bet_type="3rentan"),
            dict(race_id="two", bet_type="3rentan"),
        ])
        report = tally(**self.params)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["missing_breakdown"]["trio"], 3)
        self.assertGreater(report["missing_races_without_documented_reason"], 0)

    def test_explicit_exclusion_does_not_fake_eligible(self):
        write_table(self.params["odds"], ["race_id", "bet_type"], [
            dict(race_id="one", bet_type="3rentan"),
            dict(race_id="one", bet_type="3renpuku"),
            dict(race_id="two", bet_type="3rentan"),
            dict(race_id="two", bet_type="3renpuku"),
        ])
        p = self.root / "exclusions.csv"
        write_table(p, ["race_id", "reason"], [
            dict(race_id="three", reason="raw data has no valid odds")
        ])
        report = tally(**self.params, exclusions=p)
        self.assertEqual(report["status"], "PASS", report["issues"])
        self.assertEqual(report["excluded_races_explicit"], 1)
        self.assertEqual(report["eligible_7_9_car_races_with_three_modalities"], 2)

    def test_duplicate_car_rejected(self):
        with open(self.params["entries"], "a", newline="", encoding="utf-8") as h:
            csv.writer(h).writerow(["one", "1"])
        report = tally(**self.params)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["duplicate_car_records"], 1)

    def test_missing_source_revision_rejected(self):
        self.params["source_revision"] = "main"
        report = tally(**self.params)
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
