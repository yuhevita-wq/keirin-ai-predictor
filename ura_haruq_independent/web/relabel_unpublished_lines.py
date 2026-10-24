#!/usr/bin/env python3
"""Repair only the status wording of line-unpublished V1 forecasts.

Never recalculate forecast scores. Never invent line relations. Never change
frozen V1 tickets or stake. Applies only to Ura-only publication directory.
"""
import json
from pathlib import Path
from ura_haruq_independent.recommendation.low_quarter_advisory import evaluate_record

DIR=Path("docs/ura-haru/results")

def main():
    manifest=DIR/"index.json"
    m=json.loads(manifest.read_text(encoding="utf-8"))
    if m["system"]!="URA_HARU_V1_20261024":
        raise RuntimeError("WRONG_PUBLICATION_SYSTEM")
    adjusted=0
    for summary in m["items"]:
        rid=summary["race_id"]
        path=DIR/(rid+".json")
        doc=json.loads(path.read_text(encoding="utf-8"))
        old=doc["ura_haru_v1_display"]
        if old.get("status")!="v1_not_applicable":
            continue
        rank=doc.get("ranked") or []
        if len(rank)!=7 or not 0<=float(doc["score_gap"])<1:
            continue
        lines={int(r["bike"]):r.get("line_group") or "" for r in rank}
        if all(lines.values()):
            continue
        new=evaluate_record({
            "score_gap":doc["score_gap"],
            "predicted_bikes_by_rank":[int(r["bike"]) for r in rank],
            "line_groups_by_bike":lines,
        })
        if new.get("status")!="line_data_missing":
            raise RuntimeError("UNEXPECTED_NEW_STATUS_"+rid)
        if old["tickets"]!=new["tickets"] or old["points"]!=new["points"] or old["research_stake_yen"]!=new["research_stake_yen"]:
            raise RuntimeError("FROZEN_BET_WOULD_CHANGE_"+rid)
        doc["ura_haru_v1_display"]=new
        path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        summary["status"]=new["status"]
        adjusted+=1
    manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("URA_UNPUBLISHED_LINES_EXPLICITLY_MARKED",adjusted,"OF",len(m["items"]))

if __name__=="__main__":
    main()
