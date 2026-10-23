"""HaruQ v3 user-authorized only-HIGH policy on unchanged March 21 model.

F2 is eligible (applied by flagship_live_predict). No retraining,
new ranking, invented tickets or discretionary bets.
Both markets keep ORIGINAL high thresholds and fixed formations:
2車単 HIGH gap>=1.00: 4 * 500 yen; 3連単 S3-HIGH gap>=2.00: 6 * 500 yen.
MED+, MED and LOW are NEVER purchased. Retain original bets for audit.
"""
DISPLAY_NAME = "ハルキュー"
POLICY_VERSION = "HARUQ_HIGH_ONLY_ALL_MARKETS_F2_ELIGIBLE_V3"


def apply_user_policy(original_bets: dict) -> dict:
    exacta = original_bets["exacta"]
    trifecta = original_bets["trifecta"]
    ex_tier = exacta["label"]
    tier = trifecta["label"]
    assert ex_tier in {"HIGH", "MED+", "MED", "LOW", "SKIP"}
    assert tier in {"S3-HIGH", "S3-MED+", "S3-MED", "LOW", "SKIP"}

    if ex_tier == "HIGH":
        assert exacta["points"] == 4 and len(set(exacta["tickets"])) == 4
        assert exacta["stake_per_ticket_yen"] == 500
        assert exacta["total_yen"] == 2000
        approved_exacta = {**exacta, "tickets":list(exacta["tickets"]), "decision":"BUY", "decision_ja":"買う"}
    else:
        approved_exacta = {
            "label":ex_tier,
            "stake_per_ticket_yen":0,"points":0,"tickets":[],"total_yen":0,
            "decision":"REJECT","decision_ja":"買わない（HIGH未満）",
        }

    if tier == "S3-HIGH":
        assert trifecta["points"] == 6 and len(set(trifecta["tickets"])) == 6
        assert trifecta["stake_per_ticket_yen"] == 500
        assert trifecta["total_yen"] == 3000
        approved_tri = {**trifecta, "tickets":list(trifecta["tickets"]),
                        "decision":"BUY","decision_ja":"必ず買う"}
        decision = "BUY"
        label = "必ず買う"
    else:
        approved_tri = {
            "label":tier,
            "decision":"REJECT",
            "decision_ja":"買わない（S3-HIGH未満）",
            "tickets":[],"points":0,"stake_per_ticket_yen":0,"total_yen":0,
        }
        decision = "REJECT"
        label = "買わない（S3-HIGH未満）"

    # Compatibility: the website must never treat lower confidence as review/buy.
    under_review = {
        "trifecta":{
            "label":tier,"tickets":[],"points":0,"stake_per_ticket_yen":0,
            "conditional_total_yen":0,"counts_as_purchase":False,
            "decision":"なし（MEDも購入対象外）",
        }
    }
    return {
        "name":DISPLAY_NAME,
        "policy_version":POLICY_VERSION,
        "trifecta_tier":tier,
        "trifecta_decision":decision,
        "trifecta_decision_ja":label,
        "approved_bets":{"exacta":approved_exacta,"trifecta":approved_tri},
        "under_review":under_review,
        "approved_total_yen":approved_exacta["total_yen"]+approved_tri["total_yen"],
        "never_modify_model":True,
        "user_review_required_for_med":False,
        "exacta_high_only":True,
        "trifecta_high_only":True,
        "f2_eligible":True,
    }
