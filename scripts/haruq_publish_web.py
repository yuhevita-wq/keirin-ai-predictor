#!/usr/bin/env python3
"""Publish strictly unmodified HaruQ prediction results to the public web UI.

The frozen model, feature scores, tickets and stake sizes are never changed here.
The only new fields are web_record with GitHub Actions audit metadata.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re

from scripts.flagship_live_predict import MODEL_SHA256
from scripts.haruq_operating_policy import POLICY_VERSION


def export_result(source: Path, directory: Path, run_id: str, commit: str):
    value = json.loads(source.read_text(encoding="utf-8"))
    race_id = str(value.get("race_id", ""))
    if not re.fullmatch(r"\d{12}", race_id):
        raise ValueError("Invalid race id: refuse to publish")
    if value.get("system") != "FLAGSHIP_HARUQUBE_20260321_V1":
        raise ValueError("Non-flagship output: refuse to publish")
    if value.get("model_sha256") != MODEL_SHA256:
        raise ValueError("Wrong pretrained model: refuse to publish")
    if value.get("display_name") != "ハルキュー":
        raise ValueError("Non-HaruQ result: refuse to publish")
    if value.get("s3_policy", {}).get("policy_version") != POLICY_VERSION:
        raise ValueError("Unapproved S3 policy: refuse to publish")
    if value.get("f2_all_grades_eligible") is not True:
        raise ValueError("F2 purchase-eligibility override not present")
    if value.get("bets") != value["s3_policy"]["approved_bets"]:
        raise ValueError("Published tickets differ from policy output")
    if value.get("total_yen") != value["s3_policy"]["approved_total_yen"]:
        raise ValueError("Published monetary total inconsistent")
    if not re.fullmatch(r"\d+", str(run_id)):
        raise ValueError("Invalid GitHub workflow run id")

    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    value["web_record"] = {
        "source_run_id": int(run_id),
        "source_commit": commit,
        "created_at": now,
        "origin": "verified-github-actions-output",
    }
    pretty = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    (directory / f"{race_id}.json").write_text(pretty, encoding="utf-8")
    archives = directory / "runs"
    archives.mkdir(parents=True, exist_ok=True)
    immutable = archives / f"{run_id}-{race_id}.json"
    if immutable.exists():
        raise ValueError("Duplicate run ID; immutable snapshot collision")
    immutable.write_text(pretty, encoding="utf-8")

    manifest_path = directory / "index.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else
                {"schema_version": 1, "model": value["system"], "items": []})
    summary = {
        "race_id": race_id,
        "date": value["date"],
        "velodrome": value["velodrome"],
        "race_number": value["race_number"],
        "start_time": value.get("start_time"),
        "confidence": value["confidence"],
        "status": value["status"],
        "s3_decision": value["s3_decision"],
        "trifecta_tier": value["s3_policy"]["trifecta_tier"],
        "exacta_tier": value["bets"]["exacta"]["label"],
        "total_yen": value["total_yen"],
        "updated_at": now,
        "run_id": int(run_id),
    }
    manifest["items"] = [summary] + [
        item for item in manifest.get("items", []) if item.get("race_id") != race_id
    ]
    manifest["items"] = manifest["items"][:100]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("HARUQ_WEB_PUBLISHED", race_id, "run", run_id, "entries", len(manifest["items"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--directory", default="docs/haruq/results", type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--commit", required=True)
    args = p.parse_args()
    export_result(args.source, args.directory, args.run_id, args.commit)


if __name__ == "__main__":
    main()
