"""ハルキュー運用方針: 3/21 frozen model output -> user-approved S3 actions.

Separates the user's 2026-10-23 S3 purchase policy from the immutable
March 21 original source and preserved trained model. No inference,
re-ranking, new bets, odds filters, or probabilistic judgment here.
"""

DISPLAY_NAME = "ハルキュー"
POLICY_VERSION = "HARUQ_S3_HIGH_BUY_MEDPLUS_REJECT_MED_REVIEW_F2_ELIGIBLE_V2"


def apply_user_policy(original_bets: dict) -> dict:
    """Filter user S3 tiers; called after explicit F2 purchase-eligibility override."""
    exacta = original_bets["exacta"]
    tri = original_bets["trifecta"]
    tier = tri["label"]

    expected = {
        "S3-HIGH": (500, 6),
        "S3-MED+": (200, 6),
        "S3-MED": (100, 6),
    }
    if tier in expected:
        each, points = expected[tier]
        assert tri["stake_per_ticket_yen"] == each
        assert tri["points"] == points
        assert len(set(tri["tickets"])) == points
        assert tri["total_yen"] == each * points
    else:
        assert tier in {"LOW", "SKIP"}, f"Unexpected original 3連単 category: {tier}"
        assert tri["total_yen"] == 0 and not tri["tickets"]

    if tier == "S3-HIGH":
        decision, label = "BUY", "絶対買う"
        approved = tri["tickets"][:]
        candidate = []
        each_yen = 500
        candidate_each_yen = 0
    elif tier == "S3-MED":
        decision, label = "REVIEW", "検討"
        approved = []
        candidate = tri["tickets"][:]
        each_yen = 0
        candidate_each_yen = 100
    elif tier == "S3-MED+":
        decision, label = "REJECT", "買わない"
        approved = []
        candidate = []
        each_yen = 0
        candidate_each_yen = 0
    else:
        decision, label = "SKIP", "見送り"
        approved = []
        candidate = []
        each_yen = 0
        candidate_each_yen = 0

    approved_tri = {
        "label": tier,
        "decision": decision,
        "decision_ja": label,
        "tickets": approved,
        "points": len(approved),
        "stake_per_ticket_yen": each_yen,
        "total_yen": each_yen * len(approved),
    }
    review_tri = {
        "label": tier,
        "tickets": candidate,
        "points": len(candidate),
        "stake_per_ticket_yen": candidate_each_yen,
        "conditional_total_yen": candidate_each_yen * len(candidate),
        "counts_as_purchase": False,
        "decision": "ユーザー検討待ち" if candidate else "なし",
    }
    return {
        "name": DISPLAY_NAME,
        "policy_version": POLICY_VERSION,
        "trifecta_tier": tier,
        "trifecta_decision": decision,
        "trifecta_decision_ja": label,
        "approved_bets": {"exacta": exacta, "trifecta": approved_tri},
        "under_review": {"trifecta": review_tri},
        "approved_total_yen": exacta["total_yen"] + approved_tri["total_yen"],
        "never_modify_model": True,
        "user_review_required_for_med": (tier == "S3-MED"),
    }
