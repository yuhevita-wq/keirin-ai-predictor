# 最有力・固定基準AI: 春キューブ 2026-03-21 復元版 v1

**Status:** FLAGSHIP / FROZEN / DO NOT MODIFY IN PLACE  
**User decision:** 2026-10-23 — この研究で最重要のAIとして保存・優先し、他のAIより先に比較基準にする。  
**Alias:** `最有力AI`, `春キューブ3/21固定版`, `FLAGSHIP_HARUQUBE_20260321_V1`

## 不変の出所

- Original haruqube commit: `336fbd87b4a04632ffe65c58de91553d431003e2` (2026-03-21 22:56 JST).
- Restoration successful GitHub Actions run: [39934967427](https://github.com/yuhevita-wq/keirin-ai-predictor/actions/runs/39934967427), completed 2026-10-23 19:07 JST.
- Exact frozen, verified reproduction commit: `7709f396a885847d0b9bc40034c2cfc7b5e2b012`.
- Frozen branch: [frozen/flagship-haruqube-20260321-v1](https://github.com/yuhevita-wq/keirin-ai-predictor/tree/frozen/flagship-haruqube-20260321-v1).
- Original source blobs: `config.py` `011c2091816d16358dfe478726bcb0ff2bc74bf9`; `features/builder.py` `b47b793b5cf763bcb6d05137a115105006e50eba`; `features/line_features.py` `ecf3ab7c762475946054f806bb1d4b802c889389`.
- SHA256 of the original GitHub Actions artifact zip: `ecd322d4e2e90bafb1cfa308025cce89a4f95189dbf89f42099d9269b756eaaf`.
- Artifact ID: `12494171474`. **Expires 2027-01-21; use the pinned GitHub Release archive rather than relying on that 90-day artifact**.
- Source data: newly collected independently encrypted daily SQLite shards, 2023-01-01 onwards. Do not substitute original author's released DB/model or smaller sample data.

## Fixed reproducible model and experiment

LightGBM LambdaRank, original 44-feature source of March 21, historical class/gear correction applied at training. Date-wise chronological split: training 2023–2024, validation 2025 H1, out-of-sample holdout 2025-07-01 to 2026-09-30.

- Merged 98,897 races; quality-valid 91,489.
- Training 48,297; validation 11,741; holdout 30,162.
- Holdout Top1 15,390/30,162 = **51.0245%**.
- Holdout Top3 overlap **65.5682%**; NDCG@3 **0.8340656**.

### Original March 21 betting logic — locked

- 2車単: predicted first -> predicted #2–#5, 4 tickets. Score gap HIGH ≥1.00 ¥500/ticket; MED+ ≥0.80 ¥200; MED ≥0.50 ¥100.
- 3連単: predicted first -> permutations of predicted #2/#3/#4, 6 tickets. Score gap S3-HIGH ≥2.00 ¥500/ticket; S3-MED+ ≥1.50 ¥200; S3-MED ≥1.00 ¥100.
- Excludes F2. Does **not** apply March 22 modifications to March 21 bet decisions.

### Verified holdout ROI — all payout combos verified against actual finishing order

| Source model+strategy | Races | Invested JPY | Returned JPY | Profit JPY | ROI |
|---|---:|---:|---:|---:|---:|
| March 21 exacta | 7,154 | 8,704,800 | 11,085,300 | +2,380,500 | 127.347% |
| **March 21 trifecta** | **3,344** | **4,623,600** | **8,676,640** | **+4,053,040** | **187.660%** |
| **March 21 combined** | **10,498 bets** | **13,328,400** | **19,761,940** | **+6,433,540** | **148.269%** |
| March 22 rule on SAME March 21 model, diagnostic only | 1,323 | 2,646,000 | 5,306,550 | +2,660,550 | 200.550% |

March 21 original 3連単 subgroups:
- S3-HIGH: 928R, ROI **253.414%**, profit +¥4,271,050.
- S3-MED+: 650R, ROI 74.541%, loss -¥198,580.
- S3-MED: 1,766R, ROI 98.166%, loss -¥19,430.

## Freeze / research policy

1. **Never retrain, overwrite, delete, optimize, or modify this flagship snapshot in place.** The frozen model and all associated metrics remain a historical checkpoint.
2. New models, thresholds, feature sets, market integration, trifecta shapes, and retraining are **separate named challenger versions**; benchmark against this flagship with identically scoped data.
3. Preserve model pickle, training report, selected tickets ledger, and collection manifest together with SHA256 checksums. The workflow artifact expires after 90 days: create and verify a GitHub Release containing all four files; avoid depending solely on the action artifact.
4. Do not claim in-sample model quality, return rate or 2024–26 author-reported 315% is confirmed. This is a **historical out-of-sample backtest**, not guaranteed future performance. Live line bonus and per-bet odds/torigami checks were unavailable, and the 3/21 versus 3/22 model training/evaluation populations differ. Future no-leakage, live-timestamp odds auditing remains required.
5. If future research beats it, do not erase or demote it silently. Explain robust out-of-time superiority and ask the user before changing flagship status.

**Canonical recovery phrase:** "最有力AI＝春キューブ3月21日固定版、GitHub frozen/flagship-haruqube-20260321-v1 / successful run 39934967427 / release haruqube-20260321-flagship-v1".
