#!/usr/bin/env python3
"""Write ONLY docs/ura-haru/results; never write flagship docs/haruq/results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from ura.manual_formations import evaluate_race as manual_decision, POLICY_ID
from ura_haruq_independent.web.live_predict import SYSTEM_ID

FIELD_SET = ("race_id","date","velodrome","race_number","race_name",
 "start_time","grade","status","score_gap","line_pattern","rider_count",
 "stars","recommendation","points","total_yen","advisory_version","published_at")


def verify_document(v):
    if not isinstance(v,dict) or v.get('system') != SYSTEM_ID:
        raise ValueError('INCOMPATIBLE_MANUAL_OUTPUT')
    rid=str(v.get('race_id',''))
    if re.fullmatch(r'\d{12}',rid) is None:raise ValueError('BAD_RACE_ID')
    if v.get('date')!=f'{rid[:4]}-{rid[4:6]}-{rid[6:8]}':raise ValueError('RACE_DATE_MISMATCH')
    if v.get('original_flagship_data_used') is not False:raise ValueError('ORIGINAL_MODEL_DATA_CANNOT_BE_MIXED_IN')
    source=v.get('manual_input',{})
    if source.get('race_id')!=rid or source.get('model_sha256')!=v.get('model_sha256'):
        raise ValueError('MANUAL_SOURCE_MODEL_OR_RACE_MISMATCH')
    ranks=[{'bike':r['bike'],'score':r['score']} for r in v['ranked']]
    if source.get('ranked')!=ranks:raise ValueError('RANK_SNAPSHOT_MISMATCH')
    actual=v.get('ura_haru_manual_display')
    if not isinstance(actual,dict) or actual.get('policy_id')!=POLICY_ID:
        raise ValueError('MANUAL_POLICY_REQUIRED')
    if actual!=manual_decision(source):raise ValueError('MANUAL_FORMATION_RECOMPUTATION_MISMATCH')
    return rid


def publish_one(value:dict, target:Path, run_id:str) -> dict:
    rid = verify_document(value)
    path = target / f"{rid}.json"
    if path.exists():
        older = json.loads(path.read_text(encoding="utf-8"))
        if str(older["race_id"]) != rid:
            raise RuntimeError("RACE_ID_COLLISION")
    value["web_record"] = {
        "producer":"independent-ura-haru-github-actions",
        "run_id":str(run_id),
        "public_data_only":True,
    }
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    ev=value["ura_haru_manual_display"]
    return {
        "race_id":rid,"date":value["date"],"velodrome":value.get("velodrome"),
        "race_number":value.get("race_number"),"race_name":value.get("race_name"),
        "grade":value.get("grade"),"start_time":value.get("start_time"),
        "status":ev.get("decision"),"reason":ev.get("reason"),"race_status":value.get("status"),
        "score_gap":value.get("score_gap"),"line_pattern":value.get("line_pattern"),
        "rider_count":value.get("rider_count"),
        "points":ev['points'],"total_yen":ev['stake_yen'],
        "formation_notations":[b['notation'] for b in ev['selected']['blocks']] if ev['selected'] else [],
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
            current={"schema_version":2,"system":SYSTEM_ID,"items":[]} # Preserve old files; exclude legacy feed entries
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

