# 型ラボ研究版 v0 — 独立起動（読み取り専用）

**目的:** 既存の型ラボ本番環境・DB・入稿処理に触れず、オリジナルの
A〜F判定・商品選択・買い目生成・100円単位の配分を別プロセスで呼び出す。

## コードとバージョン

- オリジナル: https://github.com/nyanta-kun/kiseki
- **固定コミット:** 6c30a77612848ba241cd6d2b2a40c957a7c37b51
- 正本: upstream の keirin/src/type_lab.py
- このフォルダは**アダプターとテストのみ**。A〜F判定や買い目戦略を写経しない。
- オリジナルに LICENSE ファイルが見当たらないため、元ソースをこのリポジトリに転載しない。
  再配布・恒久的な再利用には権利確認が必要。

## CIで起動する

研究ブランチの pull request が作成・更新されると
.github/workflows/type-lab-v0.yml が実行される。

1. このリポジトリと、固定コミットの upstream を**別ディレクトリ**へ checkout。
2. Python 3.12 標準ライブラリで元の src.type_lab を import。
3. 6型の境界、7車／9車、賭け金配分、入力異常拒否、CLIをテスト。
4. 模擬レースをCLIで実行し、type_lab_v0_result.json を artifact として保存。

**本番への影響:** 自動投票なし、DBアクセスなし、結果データの参照なし、
本番スクリプト・設定・workflowの変更なし。研究用ブランチのファイル追加のみ。

## 手元で起動する

リポジトリのルートで:

~~~bash
git clone https://github.com/nyanta-kun/kiseki.git external/kiseki
git -C external/kiseki checkout 6c30a77612848ba241cd6d2b2a40c957a7c37b51

# 実際の元コードに対する自動テスト
TYPE_LAB_SOURCE=external/kiseki/keirin \
  python -m unittest discover -s research/type_lab_v0 -p 'test_*.py' -v

# **模擬**予測盤だけでCLI動作を確認（実オッズではない）
python -c 'import json,sys;sys.path.insert(0,"research/type_lab_v0");from test_boot import make_race;print(json.dumps(make_race()))' > /tmp/type_lab_demo.json
TYPE_LAB_SOURCE=external/kiseki/keirin \
  python research/type_lab_v0/boot.py \
  --input /tmp/type_lab_demo.json --output /tmp/type_lab_result.json
~~~

## 実データ入力契約

単一レースの JSON を boot.py --input FILE に渡す。

- race_id: 識別子
- race_type: 予選 / 準決勝 / 決勝 など
- day_index: 1〜4
- entries: 7または9要素。各要素は以下を含む
  - car (1..N), p3 (0..1の3着内モデル確率), p1 (0..1の1着モデル確率)
  - group (ラインID、単騎はnull), position (隊列位置、先頭1)
  - style (逃, 追, 両), score (競走得点), behind_pct (0..100)
- predicted_odds: trifectaとtrioそれぞれ、combo（車番配列）とodds（予測倍率）の配列
- combination_probabilities: 同じ形で odds の代わりに prob（モデル確率）

**重要:** predicted_odds は、元型ラボが利用する*事前の予測オッズ*。
締切前の実オッズ、確定オッズ、払戻を混同しない。必要な賭式について
7車三連単210点／三連複35点、9車三連単504点／三連複84点の**全盤**を要求。
モデル確率・予測オッズの推定そのものはこの v0 には含まない。

## 出力契約

JSON 出力に、元型ラボの type, axis_sum, arare, rule_version,
plan, bet_type, legs (車番／予測倍率／モデル確率／賭け金) を含む。

candidate は**研究用候補**であり「購入可」を意味しない。
common_price_gate_pass は平均想定払戻・最低予測倍率の検査のみ。
実売の出走表完全性、軸ゲート、日次50%枠、時間制約・例外日は**未適用**。
「研究版 v0 の独立起動」と「実戦予測が可能」は別の到達段階である。

## 次工程（v1）

- オリジナルの価格モデル・モデル重みと、大量の過去データの正しい時点スナップショットを接続。
- 出走前特徴量の未来情報混入、ライン欠測、予測値の較正を検証。
- 元の入稿ゲートの比較用読み取り専用実装を適用し、元型ラボ結果との同一性を検証。
- AI論文・ベンター統合モデルはこの固定基準に対して別々に学習・比較する。
