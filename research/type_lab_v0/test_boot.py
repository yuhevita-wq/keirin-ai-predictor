"""Integration smoke tests: these MUST import the pinned upstream Type Lab.

All generated forecasts are SYNTHETIC test inputs, not actual odds or predictions.
"""
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import boot  # noqa: E402


def make_race(n=7, firm=True, day=1):
    p3 = ([.92, .78, .48, .35, .25, .15, .07] if firm
          else [.68, .60, .47, .40, .34, .30, .21])
    p1 = [.65, .15, .07, .05, .04, .03, .01]
    if n == 9:
        p3 = [.67, .58, .53, .45, .41, .34, .30, .23, .16]
        p1 = [.34, .18, .13, .10, .08, .06, .05, .04, .02]
    groups = [1, 1, 1, 2, 2, 3, 3] + ([4, 4] if n == 9 else [])
    positions = [1, 2, 3, 1, 2, 1, 2] + ([1, 2] if n == 9 else [])
    entries = [
        dict(
            car=i + 1, p3=p3[i], p1=p1[i],
            group=groups[i], position=positions[i],
            style="逃" if positions[i] == 1 else "追",
            score=103 - i * 2,
            behind_pct=18 if firm and i == 0 else 2,
        )
        for i in range(n)
    ]
    # Synthetic full-board forecasts used exclusively for smoke testing.
    # These are not an inference model and must never be used as live prices.
    raw = {
        perm: p1[perm[0]-1] * p3[perm[1]-1] * p3[perm[2]-1]
        for perm in itertools.permutations(range(1, n+1), 3)
    }
    total = sum(raw.values())
    trip = {k: value / total for k, value in raw.items()}
    trio = {}
    for key, prob in trip.items():
        unordered = tuple(sorted(key))
        trio[unordered] = trio.get(unordered, 0.0) + prob
    return {
        "race_id": f"synthetic-{n}",
        "race_type": "予選",
        "day_index": day,
        "entries": entries,
        "predicted_odds": {
            "trifecta": [{"combo": list(k), "odds": round(.75 / p, 3)} for k, p in trip.items()],
            "trio": [{"combo": list(k), "odds": round(.75 / p, 3)} for k, p in trio.items()],
        },
        "combination_probabilities": {
            "trifecta": [{"combo": list(k), "prob": p} for k, p in trip.items()],
            "trio": [{"combo": list(k), "prob": p} for k, p in trio.items()],
        },
    }


class TypeLabBootSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = os.environ.get("TYPE_LAB_SOURCE")
        if not cls.source:
            raise RuntimeError("TYPE_LAB_SOURCE must point at pinned kiseki/keirin checkout")
        cls.original = boot.load_original(cls.source)

    def test_exact_original_a_f_mapping(self):
        self.assertEqual(
            [self.original.type_label_of(a, s) for a in (1.5, 1.3) for s in (-2, 0, 2)],
            list("ABCDEF"),
        )

    def test_seven_car_boot_candidate(self):
        response = boot.run(make_race(), self.original)
        self.assertEqual(response["type"], "A")
        self.assertTrue(response["research_only"])
        self.assertFalse(response["bets_submitted"])
        self.assertIn(response["status"], ("candidate", "paper_gate_fail"))
        self.assertTrue(response["legs"])
        self.assertEqual(response["stake_total_yen"], 10_000)
        self.assertTrue(all(item["stake_yen"] % 100 == 0 for item in response["legs"]))

    def test_nine_car_boot_candidate(self):
        response = boot.run(make_race(9, firm=False, day=3), self.original)
        self.assertEqual(response["type"], "F")
        self.assertEqual(response["plan"], "F_line")
        self.assertEqual(response["bet_type"], "trio")
        self.assertTrue(response["legs"])
        self.assertEqual(response["stake_total_yen"], 10_000)

    def test_reject_missing_forecast(self):
        sample = make_race()
        sample["predicted_odds"] = {}
        with self.assertRaisesRegex(ValueError, "odds and probabilities must both be provided"):
            boot.run(sample, self.original)

    def test_reject_invalid_probabilities(self):
        sample = make_race()
        sample["entries"][0]["p3"] = 92  # mistakenly passed 0..100 instead of 0..1
        with self.assertRaisesRegex(ValueError, "p3"):
            boot.run(sample, self.original)

    def test_cli_standalone_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            infile = Path(tmp) / "pre_race.json"
            outfile = Path(tmp) / "answer.json"
            infile.write_text(json.dumps(make_race()), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(HERE / "boot.py"),
                "--source-root", self.source, "--input", str(infile),
                "--output", str(outfile),
            ], capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(outfile.read_text(encoding="utf-8"))
            self.assertTrue(result["original_source_commit"])
            self.assertEqual(result["type"], "A")
            self.assertFalse(result["bets_submitted"])


if __name__ == "__main__":
    unittest.main()
