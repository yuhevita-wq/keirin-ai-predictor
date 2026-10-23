#!/usr/bin/env python3
"""Mechanical purchase-logic assertions for frozen March 21, no training/data."""
import pandas as pd
from scripts.flagship_live_predict import strict_tickets, normalize_race_id, create_race_id

ordered=pd.DataFrame({"bike_number":[5,3,7,2,1,4,6]})
expected_tri=["5>3>7","5>3>2","5>7>3","5>7>2","5>2>3","5>2>7"]
for confidence,expected_ex,expected_tri_amount in [
    (0.49,0,0),
    (0.50,100,0),
    (0.80,200,0),
    (1.00,500,100),
    (1.50,500,200),
    (2.00,500,500),
]:
    result=strict_tickets(ordered,confidence,"F1")
    assert result["exacta"]["stake_per_ticket_yen"]==expected_ex,(confidence,result)
    assert result["trifecta"]["stake_per_ticket_yen"]==expected_tri_amount,(confidence,result)
    assert result["exacta"]["tickets"]==(["5>3","5>7","5>2","5>1"] if expected_ex else [])
    assert result["trifecta"]["tickets"]==(expected_tri if expected_tri_amount else [])
    assert result["exacta"]["total_yen"]==4*expected_ex
    assert result["trifecta"]["total_yen"]==6*expected_tri_amount

for grade in ["F2"]:
    result=strict_tickets(ordered,3.0,grade)
    assert result["exacta"]["points"]==0 and result["trifecta"]["points"]==0

assert normalize_race_id("202610234812")=="202610234812"
assert create_race_id("2026-10-23","四日市",12)=="202610234812"
from scripts.haruq_operating_policy import apply_user_policy, DISPLAY_NAME, POLICY_VERSION

# These new operating rules MUST NOT mutate the original 03/21 bet output.
for confidence, tier, decision, approved_tri_points, review_tri_points, original_tri_points in [
    (0.49, "LOW", "SKIP", 0, 0, 0),
    (0.9999, "LOW", "SKIP", 0, 0, 0),
    (1.00, "S3-MED", "REVIEW", 0, 6, 6),
    (1.499999, "S3-MED", "REVIEW", 0, 6, 6),
    (1.50, "S3-MED+", "REJECT", 0, 0, 6),
    (1.999999, "S3-MED+", "REJECT", 0, 0, 6),
    (2.00, "S3-HIGH", "BUY", 6, 0, 6),
    (2.5, "S3-HIGH", "BUY", 6, 0, 6),
]:
    original = strict_tickets(ordered, confidence, "F1")
    before = repr(original)
    policy = apply_user_policy(original)
    assert repr(original) == before
    assert policy["name"] == DISPLAY_NAME == "ハルキュー"
    assert policy["policy_version"] == POLICY_VERSION
    assert policy["trifecta_tier"] == tier
    assert policy["trifecta_decision"] == decision
    assert original["trifecta"]["points"] == original_tri_points
    assert policy["approved_bets"]["trifecta"]["points"] == approved_tri_points
    assert policy["under_review"]["trifecta"]["points"] == review_tri_points
    assert policy["approved_bets"]["exacta"] == original["exacta"]
    assert policy["approved_total_yen"] == (
        original["exacta"]["total_yen"] + (3000 if decision == "BUY" else 0)
    )
    if decision == "REVIEW":
        assert policy["under_review"]["trifecta"]["tickets"] == expected_tri
        assert policy["under_review"]["trifecta"]["conditional_total_yen"] == 600
        assert policy["under_review"]["trifecta"]["counts_as_purchase"] is False
    if decision == "REJECT":
        assert policy["approved_bets"]["trifecta"]["tickets"] == []
        assert policy["under_review"]["trifecta"]["tickets"] == []
    if decision == "BUY":
        assert policy["approved_bets"]["trifecta"]["tickets"] == expected_tri
        assert policy["approved_bets"]["trifecta"]["stake_per_ticket_yen"] == 500

f2 = apply_user_policy(strict_tickets(ordered, 10.0, "F2"))
assert f2["approved_total_yen"] == 0
assert f2["trifecta_decision"] == "SKIP"
print("HARUQ_USER_POLICY_TESTS_PASSED")

print("FLAGSHIP_FIXED_LOGIC_TESTS_PASSED")
