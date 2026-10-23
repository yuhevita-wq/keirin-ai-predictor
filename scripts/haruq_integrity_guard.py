"""Fail closed on unauthorized HaruQ code changes. No unlock/update command."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_SHA256 = "2fedcb1073859ef8a51c66e0c77840f180c4b8c6335a8044e4073a1fdd328b49"


class IntegrityError(RuntimeError):
    pass


def verify_integrity(root=ROOT, *, runtime=False):
    root = Path(root).resolve()
    manifest_path = root / "protection/haruq-lock.json"
    try:
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise IntegrityError("保護台帳が変更されています")
        manifest = json.loads(raw)
        for relative, rule in manifest["files"].items():
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise IntegrityError("保護ファイルが欠落/置換されています: " + relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = rule["runtime_sha256"] if runtime else rule["source_sha256"]
            if actual != expected:
                raise IntegrityError("無断変更を検出しました: " + relative)
    except (OSError, ValueError, KeyError) as exc:
        raise IntegrityError("保護台帳を検証できません") from exc
    return manifest


def verify_result(value):
    """Reject rewritten rankings, formations, stakes or tier promotion at publish."""
    ranked = value["ranked"]
    bikes = [int(row["bike"]) for row in ranked]
    if len(bikes) < 5 or len(set(bikes)) != len(bikes):
        raise IntegrityError("順位/車番が不正です")
    if [row["rank"] for row in ranked] != list(range(1, len(ranked) + 1)):
        raise IntegrityError("順位番号が不正です")
    scores = [float(row["score"]) for row in ranked]
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise IntegrityError("スコア順位が変更されています")
    gap = float(value["confidence"])
    if abs((scores[0] - scores[1]) - gap) > 0.000002:
        raise IntegrityError("信頼度がスコア差と一致しません")
    rules = {
        "exacta": ("HIGH", 1.0, [f"{bikes[0]}>{b}" for b in bikes[1:5]]),
        "trifecta": ("S3-HIGH", 2.0, [f"{bikes[0]}>{b}>{c}" for b, c in itertools.permutations(bikes[1:4], 2)]),
    }
    total = 0
    for market, (high, threshold, tickets) in rules.items():
        bet = value["bets"][market]
        # Output confidence is rounded to 8 places by the unchanged predictor.
        # Do not reclassify a boundary score from that rounded representation.
        if bet["label"] == high:
            if gap < threshold - 0.00000001:
                raise IntegrityError("HIGHへの無断昇格を検出しました")
            if (bet["tickets"] != tickets or bet["points"] != len(tickets)
                    or bet["stake_per_ticket_yen"] != 500
                    or bet["total_yen"] != 500 * len(tickets)):
                raise IntegrityError("買い目/点数/配分が変更されています")
        else:
            if (bet["tickets"] or bet["points"] or bet["stake_per_ticket_yen"]
                    or bet["total_yen"]):
                raise IntegrityError("HIGH未満を購入対象にしています")
        total += bet["total_yen"]
    if value["total_yen"] != total:
        raise IntegrityError("購入額合計が不正です")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    verify_integrity(runtime=args.runtime)
    print("HARUQ_INTEGRITY_OK")


if __name__ == "__main__":
    main()
