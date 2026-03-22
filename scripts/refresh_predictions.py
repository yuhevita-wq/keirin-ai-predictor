"""格納済み予測を現在のモデルで再生成

モデル再学習後に実行し、過去の予測を現在のモデルのスコアで更新する。
ダッシュボードのP&L計算がバックテストと一致するようになる。

Usage:
    python scripts/refresh_predictions.py
    python scripts/refresh_predictions.py --year 2026
    python scripts/refresh_predictions.py --sync  # Supabase同期も実行
"""

import sys
import io
import logging
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from config import RESULTS_DIR, MARKS
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from db.schema import get_connection

logger = logging.getLogger(__name__)

MARKS_FULL = ["◎", "○", "▲", "△", "△"]


def refresh_predictions(year: int = 2026):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # モデル読み込み
    model = LGBMRanker()
    model.load(str(RESULTS_DIR / "model_lgbm.pkl"))
    logger.info("Model loaded (%d features)", len(model.feature_names))

    # 特徴量ビルド
    builder = FeatureBuilder()
    feature_cols = builder.feature_names
    logger.info("Building dataset for %d...", year)
    df = builder.build_dataset(year, year)

    if df.empty:
        logger.error("No data")
        return

    logger.info("Dataset: %d rows", len(df))

    # 予測スコア計算
    X = df[feature_cols].fillna(0)
    df["pred_score"] = model.predict(X).values
    logger.info("Predictions done")

    # 既存予測があるレースのみ更新（予測がないレースは対象外）
    conn = get_connection()
    existing_races = set(
        r[0] for r in conn.execute(
            "SELECT DISTINCT race_id FROM predictions WHERE race_id LIKE ?",
            (f"{year}%",)
        ).fetchall()
    )
    logger.info("Existing prediction races: %d", len(existing_races))

    # 既存のexacta_oddsを保持
    existing_odds = {}
    for row in conn.execute(
        "SELECT race_id, rider_id, exacta_odds FROM predictions WHERE race_id LIKE ? AND exacta_odds IS NOT NULL",
        (f"{year}%",)
    ).fetchall():
        existing_odds[(row[0], row[1])] = row[2]

    updated_races = 0
    updated_riders = 0

    for race_id, group in df.groupby("race_id"):
        if race_id not in existing_races:
            continue

        # スコアで降順ソート
        pred = group.sort_values("pred_score", ascending=False).reset_index(drop=True)

        if len(pred) < 2:
            continue

        # 信頼度（◎-○ スコア差）
        confidence = pred.iloc[0]["pred_score"] - pred.iloc[1]["pred_score"]

        # 各選手の予測を更新
        for rank_idx, (_, row) in enumerate(pred.iterrows()):
            rank = rank_idx + 1
            mark = MARKS_FULL[rank_idx] if rank_idx < len(MARKS_FULL) else None
            rider_id = row["rider_id"]

            # 既存のexacta_oddsを保持
            e_odds = existing_odds.get((race_id, rider_id))

            conn.execute("""
                INSERT OR REPLACE INTO predictions
                (race_id, rider_id, predicted_score, predicted_rank, mark, confidence, exacta_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (race_id, rider_id, row["pred_score"], rank, mark, confidence, e_odds))
            updated_riders += 1

        updated_races += 1

    conn.commit()
    conn.close()

    logger.info("Updated: %d races, %d predictions", updated_races, updated_riders)

    # 検証: 更新後のP&L
    verify_pnl(year)


def verify_pnl(year: int):
    """更新後のP&Lを検証"""
    from config import get_bet_category

    conn = get_connection()

    preds = conn.execute("""
        SELECT p.race_id, p.rider_id, p.predicted_rank, p.predicted_score, p.confidence,
               e.bike_number, r.grade, r.date
        FROM predictions p
        JOIN races r ON p.race_id = r.race_id
        LEFT JOIN entries e ON p.race_id = e.race_id AND p.rider_id = e.rider_id
        WHERE r.date >= ? AND p.predicted_rank <= 5
        ORDER BY p.race_id, p.predicted_rank
    """, (f"{year}-01-01",)).fetchall()

    results = {}
    for row in conn.execute("""
        SELECT race_id, rider_id, finish_position
        FROM race_results WHERE race_id LIKE ?
        AND finish_position IS NOT NULL
    """, (f"{year}%",)).fetchall():
        if row['race_id'] not in results:
            results[row['race_id']] = {}
        results[row['race_id']][row['rider_id']] = row['finish_position']

    payouts = {}
    for row in conn.execute("""
        SELECT rp.race_id, rp.nisyatan_combo, rp.nisyatan_payout
        FROM race_payouts rp JOIN races r ON rp.race_id = r.race_id
        WHERE r.date >= ?
    """, (f"{year}-01-01",)).fetchall():
        payouts[row['race_id']] = (row['nisyatan_combo'], row['nisyatan_payout'])

    conn.close()

    # レースごとにグループ化
    by_race = {}
    for p in preds:
        rid = p['race_id']
        if rid not in by_race:
            by_race[rid] = {'date': p['date'], 'grade': p['grade'] or '', 'rows': []}
        by_race[rid]['rows'].append(dict(p))

    total_bet = 0
    total_ret = 0
    n2_races = 0
    n2_hits = 0

    for race_id, race in by_race.items():
        race_results = results.get(race_id, {})
        if not race_results:
            continue

        rows = sorted(race['rows'], key=lambda r: r['predicted_rank'])
        if len(rows) < 2:
            continue

        conf = rows[0]['predicted_score'] - rows[1]['predicted_score']
        grade = race['grade']
        label, rec, amount = get_bet_category(conf, grade)

        if amount <= 0:
            continue

        n2_races += 1
        bet = amount * 4
        total_bet += bet

        honmei_finish = race_results.get(rows[0]['rider_id'])
        if honmei_finish == 1:
            actual_2nd_rider = None
            for r in rows:
                if race_results.get(r['rider_id']) == 2:
                    actual_2nd_rider = r
                    break
            if actual_2nd_rider:
                targets = [r['bike_number'] for r in rows[1:]]
                if actual_2nd_rider['bike_number'] in targets:
                    payout_info = payouts.get(race_id)
                    if payout_info:
                        ret = int(payout_info[1] * amount / 100)
                        total_ret += ret
                        n2_hits += 1

    roi = total_ret / total_bet * 100 if total_bet > 0 else 0
    pnl = total_ret - total_bet
    hit_rate = n2_hits / n2_races * 100 if n2_races > 0 else 0

    print(f"\n=== 更新後P&L検証 ({year}) ===")
    print(f"  ベット対象: {n2_races}R  的中: {n2_hits} ({hit_rate:.1f}%)")
    print(f"  投資: {total_bet:,}  回収: {total_ret:,}")
    print(f"  ROI: {roi:.1f}%  収支: {pnl:+,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="予測再生成")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--sync", action="store_true", help="Supabase同期も実行")
    args = parser.parse_args()

    refresh_predictions(args.year)

    if args.sync:
        from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY
        supabase_key = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
        if SUPABASE_URL and supabase_key:
            from supabase import create_client
            from scripts.sync_to_supabase import sync_since, get_sqlite_conn
            client = create_client(SUPABASE_URL, supabase_key)
            conn = get_sqlite_conn()
            sync_since(client, conn, f"{args.year}-01-01")
            conn.close()
            print("Supabase sync complete")
        else:
            print("Supabase not configured")
