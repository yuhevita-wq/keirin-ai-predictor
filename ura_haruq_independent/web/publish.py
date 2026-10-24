#!/usr/bin/env python3
"""Write ONLY docs/ura-haru/results; never write flagship docs/haruq/results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from ura_haruq_independent.recommendation.low_quarter_advisory import ADVISORY_ID
from ura_haruq_independent.web.live_predict import SYSTEM_ID

FIELD_SET = ("race_id","date","velodrome","race_number","race_name",
 "start_time","grade","status","score_gap","line_pattern","rider_count",
 "stars","recommendation","points","total_yen","advisory_version","published_at")


def verify_document(v):
    if not isinstance(v,dict) or v.get("system") != SYSTEM_ID:
        raise ValueError("INCOMPATIBLE_INDEPENDENT_OUTPUT")
    rid = str(v.get("race_id", ""))
    if re.fullmatch(r"\d{12}", rid) is None:
        raise ValueError("BAD_RACE_ID")
    if v.get("date") != f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}":
        raise ValueError("RACE_DATE_MISMATCH")
    if v.get("original_flagship_data_used") is not False:
        raise ValueError("ORIGINAL_MODEL_DATA_CANNOT_BE_MIXED_IN")
    ev = v.get("ura_haru_v1_display")
    if not isinstance(ev,dict) or ev.get("advisory_version") != ADVISORY_ID:
        raise ValueError("MISSING_V1_ADVISORY")
    if ev.get("policy_id") != "URA_HARU_V1_20261024":
        raise ValueError("WRONG_FIXED_POLICY")
    tickets=ev.get("tickets")
    if not isinstance(tickets,list) or len(set(tickets)) != len(tickets):
        raise ValueError("NON_UNIQUE_TICKETS")
    if ev.get("points") != len(tickets) or ev.get("research_stake_yen") != 100*len(tickets):
        raise ValueError("TICKET_COUNT_OR_STAKE_MISMATCH")
    if ev.get("status") == "v1_eligible":
        if len(tickets) != 6 or v.get("line_pattern")!="2-2-2-1" or not 0 <= float(v.get("score_gap",-1)) < 1:
            raise ValueError("INVALID_ELIGIBLE_V1_OUTPUT")
        if ev.get("recommendation") is None:
            raise ValueError("ELIGIBLE_V1_HAS_NO_RECOMMENDATION")
    elif len(tickets):
        raise ValueError("INELIGIBLE_V1_CONTAINS_TICKETS")
    return rid


def publish_one(value:dict, target:Path, run_id:str) -> dict:
    rid = verify_document(value)
    path = target / f"{rid}.json"
    if path.exists():
        older = json.loads(path.read_text(encoding="utf-8"))
        verify_document(older)
        if str(older["race_id"]) != rid:
            raise RuntimeError("RACE_ID_COLLISION")
    value["web_record"] = {
        "producer":"independent-ura-haru-github-actions",
        "run_id":str(run_id),
        "public_data_only":True,
    }
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    ev=value["ura_haru_v1_display"]
    g=ev.get("recommendation")
    return {
        "race_id":rid,"date":value["date"],"velodrome":value.get("velodrome"),
        "race_number":value.get("race_number"),"race_name":value.get("race_name"),
        "grade":value.get("grade"),"start_time":value.get("start_time"),
        "status":ev.get("status"),"race_status":value.get("status"),
        "score_gap":value.get("score_gap"),"line_pattern":value.get("line_pattern"),
        "rider_count":value.get("rider_count"),
        "recommendation":{"stars":g["stars"],"stars_text":g["stars_text"],
          "recommendation":g["recommendation"]} if g else None,
        "points":ev["points"],"total_yen":ev["research_stake_yen"],
        "published_at":value.get("published_at"),
    }


def publish_batch(source:Path,target:Path,run_id:str,retain:int=800):
    if "docs/ura-haru/results" not in str(target).replace("\\","/"):
        raise ValueError("CANNOT_WRITE_OUTSIDE_URA_HARU_RESULTS")
    if not re.fullmatch(r"\d+",str(run_id)):
        raise ValueError("INVALID_ACTION_RUN_ID")
    target.mkdir(parents=True,exist_ok=True)
    manifest_path=target/"index.json"
    current={"schema_version":1,"system":SYSTEM_ID,"items":[]}
    if manifest_path.exists():
        current=json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("system") != SYSTEM_ID:
            raise RuntimeError("WRONG_EXISTING_FEED_SYSTEM")
    combined={str(row["race_id"]):row for row in current["items"]}
    count=0
    for file in sorted(source.glob("[0-9]"*12+".json")):
        value=json.loads(file.read_text(encoding="utf-8"))
        row=publish_one(value,target,run_id)
        combined[row["race_id"]]=row
        count+=1
    if not count:
        raise RuntimeError("NO_VALID_FORECAST_TO_PUBLISH")
    current["items"]=sorted(combined.values(), key=lambda r:(r["date"],r["race_id"]),reverse=True)[:retain]
    current["updated_at"]=datetime.now(timezone.utc).isoformat(timespec="seconds")
    current["source"]="independent-model-only"
    manifest_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("URA_WEB_PUBLISHED",count,"MANIFEST_SIZE",len(current["items"]))
    return current


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--target",default="docs/ura-haru/results",type=Path)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    publish_batch(a.source,a.target,a.run_id)


if __name__=="__main__":
    main()
