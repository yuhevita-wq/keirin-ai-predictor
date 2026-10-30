# 裏ハルキュー：教科書フォーメーションへの切り替え

2026-10-30のユーザー指示により、旧固定6点の購入・推奨を通常のWeb予測から撤去。
新しい唯一の判断元は `yuhevita-wq/ura-haruq` の `ura/manual_formations.py`。
使用コミットは `9199355d67b920958c15440815f4632fd18b35e4` に固定。

独立ウォークフォワード順位モデルは維持し、元の表の順位や、別モデル用の確率係数を流用しない。
独立スコアと並びから低評価選手が1着・2着・3着へ入る形を生成する。
全3連単の対応する独自確率と購入前オッズがそろえば、形全体の推定期待収益と費用で比較し、可変点数で採用・見送りを出す。
既存の2-2-2-1制約、固定6点、得点差7.1、旧おすすめ星は使わない。

現在の独立順位モデルは全3連単の較正済み確率を直接出力していない。
対応する確率がなければ `MISSING_COMPATIBLE_INDEPENDENT_PROBABILITIES` で見送り。
元の表モデル向けに学習した確率係数の転用や、順位スコアの無断確率化で穴埋めしない。
これは新方式の収支検証済み・実運用収益達成を意味しない。

`--manual-snapshots-dir` に指定したディレクトリの `<race_id>.json` から、独立確率と購入前オッズを読み込む。
通常ワークフローの配置先は `ura_haruq_independent/manual_snapshots/`。

スナップショットの項目：

- race_id
- trifecta_probabilities：全組番の確率、合計1、キー例1-2-3
- probability_provenance：race_id、source_model_sha256、ranked_scores_sha256、probability_model_id、trained_on_results=true、odds_used_as_teacher=false
- trifecta_odds：全組番の購入前オッズ
- odds_observed_at：タイムゾーン付き観測日時
- odds_kind：pre_start_snapshot

順位スコア・生成モデル・並びは今回の検証済み実行から取得し、スナップショットで上書きできない。
確率の `ranked_scores_sha256` は車番・スコアの降順一覧（キーbike/score、キー順、空白なしJSON）をSHA256化した値。
モデルと順位スナップショットの一致、発走前時刻、全組番の価格・確率を照合する。
Web公開時も判断を再計算し、点数・組番・金額が違えば停止する。

旧方式の過去ファイルは過去記録として保持。新フィードには旧版の買い目・過去回収率を混ぜない。
外部への投票・送金は実行しない。表ハルキューのモデル・出力・Webは変更しない。
