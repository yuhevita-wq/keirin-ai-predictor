#!/usr/bin/env python3
"""Independent Ura LIVE forecast with textbook formations. NEVER calls flagship model.

Uses a verified chronological independent 2026 ranker, preserved 44 pre-race
feature definitions and the textbook whole-formation policy. No V1 tickets.
The only publication output is docs/ura-haru/results/, not docs/haruq/.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from config import VELODROME_CODES, DB_PATH
from data.scraper import KeirinScraper
from db.schema import get_connection, insert_entry, insert_race, insert_rider
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.manual_formations import evaluate_race as manual_decision, POLICY_ID

ZONE = ZoneInfo("Asia/Tokyo")
SYSTEM_ID = "URA_MANUAL_FORMATIONS_V1_20261030"
log = logging.getLogger("ura-haru-manual-live")


def assert_manual_engine():
    if POLICY_ID != SYSTEM_ID:
        raise RuntimeError("MANUAL_POLICY_VERSION_MISMATCH")


def load_model(path: Path, report: Path):
    metadata = json.loads(report.read_text(encoding="utf-8"))
    expected = metadata["folds"]["2026"]["model_sha256"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("INDEPENDENT_MODEL_CHECKSUM_FAILURE")
    model = LGBMRanker()
    model.load(str(path))
    builder = FeatureBuilder()
    if model.feature_names != builder.feature_names or len(builder.feature_names) != 44:
        raise RuntimeError("INDEPENDENT_44_FEATURE_CONTRACT_MISMATCH")
    return model, builder, actual


def verify_history(day: str):
    if not DB_PATH.is_file():
        raise RuntimeError("VERIFIED_HISTORY_REQUIRED")
    with sqlite3.connect(DB_PATH) as con:
        latest, races = con.execute("SELECT MAX(date), COUNT(*) FROM races").fetchone()
    if not latest or races < 90000:
        raise RuntimeError("HISTORY_INCOMPLETE")
    horizon = (datetime.strptime(day, "%Y-%m-%d").date() -
               datetime.strptime(str(latest), "%Y-%m-%d").date()).days
    if horizon > 7:
        raise RuntimeError(f"HISTORY_STALE_{latest}_PREDICTION_DATE_{day}")
    return {"last_date": latest, "history_races": races}


def persist_entries(rid: str, data: dict):
    entries = data.get("entries") or []
    if not 7 <= len(entries) <= 9:
        raise RuntimeError(f"RANKER_REQUIRES_SEVEN_TO_NINE_ENTRIES_{len(entries)}")
    if any(str(e.get("class") or "").upper().startswith("L") for e in entries) or \
       re.search(r"ガールズ|ガ決|ガ般|Ｌ級|L級",str(data.get("race_name") or "")):
        raise RuntimeError("GIRLS_NOT_ELIGIBLE")
    if not data.get("grade") or data["grade"] not in {"GP","G1","G2","G3","F1","F2"}:
        raise RuntimeError("NO_TRUSTWORTHY_GRADE")
    bikes = [int(e.get("bike_number") or 0) for e in entries]
    riders = [str(e.get("rider_id") or "") for e in entries]
    if len(set(bikes)) != len(entries) or min(bikes) <= 0 or len(set(riders)) != len(entries) or "" in riders:
        raise RuntimeError("BAD_ENTRY_BIKES_OR_RIDERS")
    lines = {}
    for e in entries:
        group = str(e.get("line_group") or "").strip()
        lines[int(e["bike_number"])] = group
    from collections import Counter
    # Inapplicable line patterns remain visible, without V1 tickets.
    conn = get_connection()
    try:
        # The archive is a temporary copy on an isolated runner. A historical
        # entry MUST NEVER see the same event's results/features after finish.
        conn.execute("DELETE FROM race_results WHERE race_id = ?", (rid,))
        conn.execute("DELETE FROM entries WHERE race_id = ?", (rid,))
        insert_race(conn, {
            "race_id": rid,
            "date": f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}",
            "velodrome": VELODROME_CODES[rid[8:10]],
            "race_number": int(rid[10:]),
            "race_name": data.get("race_name"),
            "grade": data["grade"],
            "rider_count": len(entries),
            "start_time": data.get("start_time"),
        })
        for e in entries:
            uid = str(e["rider_id"])
            insert_rider(conn, {
                "rider_id": uid, "name": e.get("rider_name") or uid,
                "class": e.get("class"), "prefecture": e.get("prefecture"),
                "period": e.get("period"),
            })
            conn.execute("UPDATE riders SET class = ? WHERE rider_id = ?",
                         (e.get("class"), uid))
            insert_entry(conn, dict(e, race_id=rid))
        conn.commit()
    finally:
        conn.close()
    return lines


def one_race(rid: str, scraper, model, builder, model_sha: str, history: dict, manual_snapshot=None) -> dict:
    if re.fullmatch(r"\d{12}",rid) is None:
        raise ValueError("INVALID_RACE_ID")
    try:
        event_date = datetime.strptime(rid[:8], "%Y%m%d").date()
    except ValueError as e:
        raise ValueError("INVALID_RACE_DATE") from e
    day = event_date.isoformat()
    if rid[8:10] not in VELODROME_CODES or not 1 <= int(rid[-2:]) <= 12:
        raise ValueError("INVALID_VENUE_OR_RACE")
    if abs((event_date - datetime.now(ZONE).date()).days) > 3:
        raise ValueError("RACE_DATE_TOO_FAR_FROM_NOW")
    lineup_observed_at = datetime.now(ZONE).isoformat(timespec="seconds")
    data = scraper.scrape_race_entry(rid)
    if not data or not data.get("entries"):
        raise RuntimeError("ENTRY_NOT_FOUND")
    lines = persist_entries(rid, data)
    features = builder.build_race_features(rid, day)
    if len(features) != len(data["entries"]):
        raise RuntimeError("INCORRECT_FEATURE_ROW_COUNT")
    ordered_features = features[builder.feature_names].apply(pd.to_numeric, errors="coerce")
    if ordered_features.isna().all(axis=1).any():
        raise RuntimeError("EMPTY_FEATURES_FOR_A_RIDER")
    scores = model.predict(ordered_features.fillna(0.0))
    if len(scores) != len(data["entries"]) or not all(math.isfinite(float(s)) for s in scores):
        raise RuntimeError("BAD_MODEL_SCORES")
    rank_table = features.copy()
    rank_table["score"] = scores.to_numpy()
    id_to_entry = {str(e["rider_id"]): e for e in data["entries"]}
    rank_table["bike_number"] = rank_table["rider_id"].astype(str).map(
        lambda uid: id_to_entry[uid]["bike_number"] if uid in id_to_entry else None)
    if rank_table["bike_number"].isna().any():
        raise RuntimeError("FEATURE_RIDER_ID_MISMATCH")
    rank_table = rank_table.sort_values(["score","bike_number"],
       ascending=[False,True],kind="stable")
    gap = float(rank_table.iloc[0]["score"] - rank_table.iloc[1]["score"])
    bikes = [int(b) for b in rank_table["bike_number"]]
    ranked=[]
    for i,(_,r) in enumerate(rank_table.iterrows()):
        entry = id_to_entry[str(r["rider_id"])]
        ranked.append({
          "rank":i+1,"bike":int(r["bike_number"]),
          "name":entry.get("rider_name"),"class":entry.get("class"),
          "line_group":lines[int(r["bike_number"])],
          "line_role":entry.get("line_role"),
          "competition_score":entry.get("avg_competition_score"),
          "score":round(float(r["score"]),7)
        })
    now = datetime.now(ZONE)
    start = data.get("start_time")
    status = "発走時刻未取得"
    if re.fullmatch(r"\d{1,2}:\d{2}",str(start)):
        local_dt = datetime.strptime(day+"T"+start,"%Y-%m-%dT%H:%M").replace(tzinfo=ZONE)
        status = "発走前" if now < local_dt else "発走後・参考"
    snapshot = dict(manual_snapshot or {})
    # Scores/model/lineup always come from this verified run, never the sidecar.
    allowed = ('trifecta_odds','trifecta_probabilities','probability_provenance',
               'odds_observed_at','odds_kind')
    manual_input = {k:snapshot[k] for k in allowed if k in snapshot}
    manual_input.update(race_id=rid, grade=data.get('grade'),
        race_class='/'.join(sorted({str(e.get('class') or '') for e in data['entries']})),
        ranked=[{'bike':r['bike'],'score':r['score']} for r in ranked],
        confidence=round(gap,10), model_sha256=model_sha,
        model_role='INDEPENDENT_CHRONOLOGICAL_2026',
        line_formation=[[int(m['bike_number']) for m in line] for line in data.get('line_formation',[])],
        start_at=local_dt.isoformat() if re.fullmatch(r"\d{1,2}:\d{2}",str(start)) else None,
        prediction_at=now.isoformat(timespec='seconds'),lineup_at=lineup_observed_at)
    if snapshot and str(snapshot.get('race_id','')) != rid:
        raise ValueError('MANUAL_SNAPSHOT_RACE_MISMATCH')
    rec=manual_decision(manual_input)
    from collections import Counter
    group_counts = Counter(v for v in lines.values() if v)
    line_pattern="-".join(map(str,sorted(group_counts.values(),reverse=True))) if len(group_counts) and sum(group_counts.values()) == len(lines) else "ライン未確定"
    return {
      "schema_version":2,
      "system":SYSTEM_ID,"display_name":"裏ハルキュー",
      "race_id":rid,"date":day,"velodrome":VELODROME_CODES[rid[8:10]],
      "race_name":data.get("race_name"),"race_number":int(rid[-2:]),
      "grade":data.get("grade"),"start_time":start,
      "status":status,"score_gap":round(gap,10),
      "line_pattern":line_pattern,"rider_count":len(data["entries"]),
      "ranked":ranked,"manual_input":manual_input,"ura_haru_manual_display":rec,
      "model_role":"INDEPENDENT_CHRONOLOGICAL_2026",
      "model_sha256":model_sha,
      "history_latest":history["last_date"],
      "published_at":now.isoformat(timespec="seconds"),
      "input_url":"https://keirin.netkeiba.com/race/entry/?race_id="+rid,
      "asof_feature_audit_completed":False,
      "original_flagship_data_used":False
    }


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--race-id",help="One official 12-digit race id")
    src.add_argument("--date",help="Date YYYY-MM-DD (discover all races)")
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--limit",type=int,default=120)
    parser.add_argument("--exclude-venues",default="",help="Comma-separated venue codes omitted from this run")
    parser.add_argument("--manual-snapshots-dir",type=Path,help="Compatible independent joint probabilities and pre-start odds snapshots by race ID")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    assert_manual_engine()
    model,builder,sha=load_model(args.model,args.report)
    now_date=datetime.now(ZONE).date()
    d = datetime.strptime(args.date,"%Y-%m-%d").date() if args.date else \
        datetime.strptime(args.race_id[:8],"%Y%m%d").date()
    if not 0 <= (d-now_date).days <= 3:
        raise RuntimeError("ONLY_TODAY_TO_THREE_DAYS_AHEAD")
    history=verify_history(d.isoformat())
    scraper=KeirinScraper()
    ids=([args.race_id] if args.race_id else
        scraper.scrape_race_list(d.strftime("%Y%m%d")))
    ids=sorted({r for r in ids if re.fullmatch(r"\d{12}",str(r))
                and str(r).startswith(d.strftime("%Y%m%d"))
                and str(r)[8:10] not in set(args.exclude_venues.split(","))})
    if not 0 < len(ids) <= 200:
        raise RuntimeError(f"BAD_RACE_DISCOVERY_COUNT_{len(ids)}")
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    count=0
    skip=[]
    for rid in ids[:args.limit]:
        try:
            snapshot_path=args.manual_snapshots_dir/(rid+'.json') if args.manual_snapshots_dir else None
            snapshot=json.loads(snapshot_path.read_text()) if snapshot_path and snapshot_path.exists() else None
            result=one_race(rid,scraper,model,builder,sha,history,snapshot)
            (out/(rid+".json")).write_text(
               json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            count+=1
            print("URA_RACE_PROCESSED",rid,result["ura_haru_manual_display"]["reason"])
        except Exception as exc:
            skip.append({"race_id":rid,"error":str(exc)[:180]})
            log.warning("URA_RACE_SKIPPED %s %s",rid,exc)
    report={"system":SYSTEM_ID,"date":d.isoformat(),"discovered":len(ids),
            "attempted":min(len(ids),args.limit),"published":count,
            "skipped":skip,"history":history,"independent_model_sha256":sha}
    (out/"run_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    if count==0:
        raise RuntimeError("NO_VALID_URA_PREDICTIONS_CREATED")
    print("URA_HARU_LIVE_CREATED",count,"SKIPPED",len(skip))

if __name__=="__main__":
    main()

