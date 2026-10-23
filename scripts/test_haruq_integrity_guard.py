"""Tampering and fixed purchase invariants, without network, DB or model."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.haruq_integrity_guard import (
    ROOT, IntegrityError, verify_integrity, verify_result,
)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = json.loads((ROOT / 'protection/haruq-lock.json').read_text())
        for relative in list(self.manifest['files']) + ['protection/haruq-lock.json']:
            p = self.root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, p)

    def test_unchanged_source_and_exact_runtime(self):
        verify_integrity(self.root)
        for relative, rule in self.manifest['files'].items():
            if 'runtime_source' in rule:
                shutil.copyfile(self.root / rule['runtime_source'], self.root / relative)
        verify_integrity(self.root, runtime=True)

    def test_every_protected_file_change_and_missing_file_stops(self):
        for relative in self.manifest['files']:
            with self.subTest(path=relative):
                p = self.root / relative
                original = p.read_bytes()
                p.write_bytes(original + b'\n# unauthorized change\n')
                with self.assertRaises(IntegrityError):
                    verify_integrity(self.root)
                p.unlink()
                with self.assertRaises(IntegrityError):
                    verify_integrity(self.root)
                p.write_bytes(original)

    def test_manifest_rewrite_stops(self):
        p = self.root / 'protection/haruq-lock.json'
        p.write_bytes(p.read_bytes() + b'\n')
        with self.assertRaises(IntegrityError):
            verify_integrity(self.root)

    def test_symlink_replacement_stops(self):
        relative = 'scripts/haruq_operating_policy.py'
        p = self.root / relative
        p.unlink()
        p.symlink_to(ROOT / relative)
        with self.assertRaises(IntegrityError):
            verify_integrity(self.root)

    def test_unrelated_request_does_not_change_contract(self):
        (self.root / 'request.json').write_text('{"race_id":"202610234706"}')
        verify_integrity(self.root)


def result(gap=2.1):
    bikes = [1, 6, 4, 3, 2, 5, 7]
    scores = [gap + 2, 2, 1, 0, -1, -2, -3]
    ex = [f'1>{b}' for b in bikes[1:5]] if gap >= 1 else []
    tri = [f'1>{b}>{c}' for b, c in itertools.permutations(bikes[1:4], 2)] if gap >= 2 else []
    bets = {}
    for market, tickets, label in [('exacta', ex, 'HIGH' if ex else 'MED+'),
                                    ('trifecta', tri, 'S3-HIGH' if tri else 'LOW')]:
        bets[market] = dict(label=label, tickets=tickets, points=len(tickets),
                            stake_per_ticket_yen=500 if tickets else 0,
                            total_yen=500 * len(tickets))
    return dict(ranked=[dict(rank=i+1, bike=b, score=s) for i,(b,s) in enumerate(zip(bikes,scores))],
                confidence=gap, bets=bets, total_yen=sum(x['total_yen'] for x in bets.values()))


class ResultTests(unittest.TestCase):
    def test_original_boundaries_and_matsusaka_not_reclassified(self):
        for gap in [0.99900955, 1.0, 1.5, 1.99999999, 2.0, 2.1]:
            self.assertTrue(verify_result(result(gap)))
        self.assertEqual(result(0.99900955)['total_yen'], 0)

    def test_ticket_stake_rank_gap_and_tier_tampering_stops(self):
        mutations = [
            lambda v: v['bets']['exacta']['tickets'].__setitem__(0, '1>7'),
            lambda v: v['bets']['exacta'].__setitem__('stake_per_ticket_yen', 1000),
            lambda v: v['ranked'][0].__setitem__('score', -10),
            lambda v: v.__setitem__('confidence', 9),
            lambda v: v.__setitem__('total_yen', 1),
        ]
        for mutate in mutations:
            value = result()
            mutate(value)
            with self.assertRaises(IntegrityError):
                verify_result(value)
        value = result(0.99900955)
        value['bets']['exacta'] = copy.deepcopy(result()['bets']['exacta'])
        value['total_yen'] = 2000
        with self.assertRaises(IntegrityError):
            verify_result(value)


if __name__ == '__main__':
    unittest.main()
