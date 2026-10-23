#!/usr/bin/env python3
"""Exact frozen 2026-03-21 flagship betting decision from a named keirin race.

NO training, discretionary filtering, market logic, line bonus or bet selection.
Sources: verified GitHub Release model, historical March-21 Python feature source.
Live entry feature generation uses the historical author's build_race_features().
"""
import argparse
from datetime import date, datetime, timedelta
import hashlib
import itertools
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import VELODROME_CODES, CLASS_MAP, DB_PATH, get_bet_category, get_trifecta_category
from db.schema import get_connection, insert_race, insert_rider, insert_entry
from data.scraper import KeirinScraper
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from scripts.haruq_operating_policy import apply_user_policy

MODEL_SHA256 = "e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193"
MODEL_TAG = "haruqube-20260321-flagship-v1"
HISTORICAL_COMMIT = "336fbd87b4a04632ffe65c58de91553d431003e2"
TIMEZONE = ZoneInfo("Asia/Tokyo")
logger = logging.getLogger("flagship")

def normalize_race_id(value):
    """YYYYMMDD + 2-digit velodrome + 2-digit race number."""
    value = str(value).strip()
    if re.fullmatch(r"\d{12}", value) is None:
        raise ValueError("レースIDは12桁 (YYYYMMDD + 場コード2桁 + R2桁) 必須")
    day = date.fromisoformat(value[:4] + "-" + value[4:6] + "-" + value[6:8])
    code = value[8:10]
    number = int(value[10:])
    if code not in VELODROME_CODES or not 1 <= number <= 12:
        raise ValueError("場コードまたはレース番号が不正です")
    if day < date(2023,1,1):
        raise ValueError("この固定モデルの履歴基盤は2023年以降です")
    return value

def create_race_id(day, venue, race_number):
    target = date.fromisoformat(str(day))
    clean = str(venue).strip()
    reverse = {v: k for k,v in VELODROME_CODES.items()}
    code = clean if clean in VELODROME_CODES else reverse.get(clean)
    if not code:
        raise ValueError("未知の競輪場名/コード: " + clean)
    num = int(race_number)
    return normalize_race_id(target.strftime("%Y%m%d") + code + f"{num:02d}")

def verified_model(path):
    path = Path(path)
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != MODEL_SHA256:
        raise RuntimeError("固定モデルのSHA256不一致。買い目を出しません。")
    model = LGBMRanker()
    model.load(str(path))
    builder = FeatureBuilder()
    if len(builder.feature_names) != 44 or model.feature_names != builder.feature_names:
        raise RuntimeError("3/21固定版44特徴量契約が不一致。買い目を出しません。")
    return model, builder

def validate_history(target):
    if not DB_PATH.is_file():
        raise RuntimeError("履歴DBが未取得です。適当な特徴量での予測は禁止。")
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT MAX(date), COUNT(*) FROM races").fetchone()
    max_day, races = (row[0], int(row[1])) if row else (None, 0)
    if not max_day or races < 90000:
        raise RuntimeError(f"履歴DBが不完全です。最終日={max_day}, レース数={races}")
    d = date.fromisoformat(target[:4]+"-"+target[4:6]+"-"+target[6:8])
    history_end = date.fromisoformat(max_day)
    if d > history_end + timedelta(days=7):
        raise RuntimeError(f"履歴の更新が必要です。最終日={max_day}、対象日={d}")
    if d > datetime.now(TIMEZONE).date() + timedelta(days=7):
        raise RuntimeError("7日を超える未来の開催は、出走表確定前の可能性があるため不可")
    return {"races": races, "last_race_date": max_day}

def record_entries(race_id, data):
    entries = data.get("entries") or []
    if len(entries) < 5 or len(entries) > 9:
        raise RuntimeError(f"出走表が不完全です。選手数={len(entries)}")
    numbers = [e.get("bike_number") for e in entries]
    riders = [e.get("rider_id") for e in entries]
    if any(not x for x in numbers+riders):
        raise RuntimeError("選手IDまたは車番が欠けています")
    if len(set(map(str,numbers))) != len(entries) or len(set(map(str,riders))) != len(entries):
        raise RuntimeError("選手ID/車番の重複を検出しました")
    race_grade = data.get("grade")
    girls = all(e.get("class") is None for e in entries) or any(str(e.get("class") or "").upper().startswith("L") for e in entries) or bool(re.search(r"ガールズ|ガール|ガ決|ガ般|Ｌ級|L級", str(data.get("race_name") or "")))
    if girls:
        raise RuntimeError("ガールズケイリンはハルキュー購入対象外。L級出走表にS3-HIGHを適用しません。")
    if race_grade not in {"GP","G1","G2","G3","F1","F2"}:
        raise RuntimeError(f"グレードが取得できません: {race_grade!r}。推測して購入しません。")
    day = race_id[:4]+"-"+race_id[4:6]+"-"+race_id[6:8]
    conn = get_connection()
    try:
        # A freshly loaded, temporary historical DB can contain the completed
        # result of an old smoke-test race. Strictly discard any target result.
        conn.execute("DELETE FROM race_results WHERE race_id = ?", (race_id,))
        conn.execute("DELETE FROM entries WHERE race_id = ?", (race_id,))
        insert_race(conn, {
            "race_id": race_id, "date": day,
            "velodrome": VELODROME_CODES[race_id[8:10]],
            "race_number": int(race_id[10:]),
            "race_name": data.get("race_name"),
            "grade": race_grade, "rider_count": len(entries),
            "start_time": data.get("start_time"),
        })
        for e in entries:
            rid=str(e["rider_id"])
            cls=e.get("class")
            insert_rider(conn, {
                "rider_id":rid, "name":e.get("rider_name") or rid,
                "class":cls,"prefecture":e.get("prefecture"), "period":e.get("period"),
            })
            # The offline frozen model used contemporaneous result class,
            # never later master class. For real-time, use today's race class.
            conn.execute("UPDATE riders SET class = ? WHERE rider_id = ?",(cls,rid))
            insert_entry(conn,dict(e, race_id=race_id))
        conn.commit()
        return race_grade
    finally:
        conn.close()

def strict_tickets(ordered, gap, grade, *, allow_f2=False):
    bikes=[int(v) for v in ordered["bike_number"]]
    if len(bikes)<5 or len(bikes)!=len(set(bikes)):
        raise RuntimeError("予測車番の欠損・重複を検出")
    # Keep 2026-03-21 baseline untouched. Explicit 2026-10-23 user F2
    # eligibility affects ONLY purchase tier computation, not model/scores.
    purchase_grade = "" if allow_f2 and grade == "F2" else grade
    ex_label,_,ex_amount=get_bet_category(gap,purchase_grade)
    tri_label,_,tri_amount=get_trifecta_category(gap,purchase_grade)
    # EXACT 2026-03-21 logic: top 1 -> top 2,3,4,5; 4 distinct tickets.
    exacta=[f"{bikes[0]}>{b}" for b in bikes[1:5]] if ex_amount else []
    # EXACT 2026-03-21 logic: top 1 -> permutation of top 2,3,4; 6 tickets.
    tri=[f"{bikes[0]}>{b}>{c}" for b,c in itertools.permutations(bikes[1:4],2)] if tri_amount else []
    if ex_amount and len(exacta)!=4 or tri_amount and len(tri)!=6:
        raise RuntimeError("買い目の件数が固定戦略と一致しません")
    return {
        "exacta":{"label":ex_label,"stake_per_ticket_yen":ex_amount,"points":len(exacta),
                  "total_yen":ex_amount*len(exacta),"tickets":exacta},
        "trifecta":{"label":tri_label,"stake_per_ticket_yen":tri_amount,"points":len(tri),
                    "total_yen":tri_amount*len(tri),"tickets":tri},
    }

def main():
    p=argparse.ArgumentParser(description="FLAGSHIP HARUQUBE 03/21 frozen inference")
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--race-id",help="12 digit official netkeirin race_id")
    source.add_argument("--request-json",help="JSON with race_id or {date, venue, race}")
    p.add_argument("--model",required=True)
    p.add_argument("--out",default="flagship_result.json")
    p.add_argument("--allow-historical",action="store_true",help="Smoke-test past race only")
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

    if args.request_json:
        req=json.loads(Path(args.request_json).read_text(encoding="utf-8"))
        if "race_id" in req:
            target=normalize_race_id(req["race_id"])
        else:
            target=create_race_id(req["date"],req["venue"],req["race"])
    else:
        target=normalize_race_id(args.race_id)
    event_date = date.fromisoformat(target[:4]+"-"+target[4:6]+"-"+target[6:8])
    if event_date < datetime.now(TIMEZONE).date() and not args.allow_historical:
        raise RuntimeError("過去のレースです。実用買い目として表示しません。")

    model,builder=verified_model(args.model)
    hist=validate_history(target)
    scraper=KeirinScraper()
    # The source parser MUST fetch a real entry page. No invented entries.
    data=scraper.scrape_race_entry(target)
    if not data or not data.get("entries"):
        raise RuntimeError("出走表取得失敗。推測の買い目は出しません。")
    grade=record_entries(target,data)
    date_iso=event_date.isoformat()
    frame=builder.build_race_features(target,date_iso)
    if len(frame)!=len(data["entries"]):
        raise RuntimeError("44特徴量の生成人数が出走表と一致しません")
    feats=frame[builder.feature_names].apply(__import__("pandas").to_numeric,errors="coerce")
    if feats.isna().all(axis=1).any():
        raise RuntimeError("特徴量が全欠損の選手がいます")
    frame["score"]=model.predict(feats.fillna(0.0)).values
    # Stable deterministic score order, same tie rule as the reproducible holdout.
    bike_by_rider={str(e["rider_id"]):int(e["bike_number"]) for e in data["entries"]}
    name_by_rider={str(e["rider_id"]):e.get("rider_name","") for e in data["entries"]}
    frame["bike_number"]=frame["rider_id"].astype(str).map(bike_by_rider)
    if frame["bike_number"].isna().any():
        raise RuntimeError("特徴量と車番の対応が不明")
    frame=frame.sort_values(["score","bike_number"],ascending=[False,True],kind="stable")
    gap=float(frame.iloc[0]["score"]-frame.iloc[1]["score"])
    original_bets=strict_tickets(frame,gap,grade)
    eligible_bets=strict_tickets(frame,gap,grade,allow_f2=True)
    operation=apply_user_policy(eligible_bets)
    now=datetime.now(TIMEZONE)
    start=data.get("start_time")
    race_started=None
    if re.fullmatch(r"\d{1,2}:\d{2}",str(start)):
        start_time=datetime.fromisoformat(date_iso+"T"+start).replace(tzinfo=TIMEZONE)
        race_started=now >= start_time
    else:
        race_started=event_date < now.date()

    result={
        "system":"FLAGSHIP_HARUQUBE_20260321_V1",
        "display_name":"ハルキュー",
        "status":("過去レースの実験" if race_started else
                  "発走時刻未確認" if start is None else "発走前"),
        "race_id":target,"date":date_iso,
        "velodrome":VELODROME_CODES[target[8:10]],"race_number":int(target[-2:]),
        "start_time":start,"grade":grade,
        "history":hist,
        "confidence":round(gap,8),
        "ranked":[{"rank":i+1,"bike":int(row["bike_number"]),
                   "name":name_by_rider.get(str(row["rider_id"]),""),
                   "score":round(float(row["score"]),6)}
                  for i,(_,row) in enumerate(frame.iterrows())],
        "bets":operation["approved_bets"],
        "original_model_bets":original_bets,
        "f2_purchase_override_applied":grade == "F2",
        "f2_all_grades_eligible":True,
        "s3_decision":operation["trifecta_decision_ja"],
        "s3_policy":operation,
        "med_under_review":operation["under_review"]["trifecta"],
        "total_yen":operation["approved_total_yen"],
        "feature_source":"historical 2026-03-21 build_race_features",
        "model_sha256":MODEL_SHA256,
        "no_manual_selection":True,"no_retraining":True,"line_bonus_applied":False,
        "torigami_filter_applied":False,
        "input_url":f"https://keirin.netkeiba.com/race/entry/?race_id={target}",
    }
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FLAGSHIP_LIVE_RESULT "+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
