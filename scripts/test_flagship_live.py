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

# Frozen 03/21 baseline is unchanged at every exact score threshold.
for score,ex_orig,tri_orig in [(0.49,0,0),(0.5,100,0),(0.8,200,0),
                               (1.0,500,100),(1.49999,500,100),
                               (1.5,500,200),(1.99999,500,200),(2.0,500,500)]:
    orig=strict_tickets(ordered,score,"F1")
    assert orig["exacta"]["stake_per_ticket_yen"]==ex_orig
    assert orig["trifecta"]["stake_per_ticket_yen"]==tri_orig
    before=repr(orig)
    decision=apply_user_policy(orig)
    assert repr(orig)==before
    assert decision["policy_version"]==POLICY_VERSION
    assert decision["name"]==DISPLAY_NAME=="ハルキュー"
    assert decision["approved_bets"]["exacta"]["total_yen"]==(2000 if score>=1.0 else 0)
    assert decision["approved_bets"]["trifecta"]["total_yen"]==(3000 if score>=2.0 else 0)
    assert decision["approved_total_yen"]==(2000 if score>=1.0 else 0)+(3000 if score>=2.0 else 0)
    assert decision["under_review"]["trifecta"]["tickets"]==[]
    assert decision["user_review_required_for_med"] is False
    assert decision["trifecta_decision"]==("BUY" if score>=2.0 else "REJECT")
    assert decision["trifecta_high_only"] and decision["exacta_high_only"]
    if score>=2.0:assert decision["approved_bets"]["trifecta"]["tickets"]==expected_tri
    if score<1.0:assert decision["approved_bets"]["exacta"]["tickets"]==[]

# F2 is eligible ONLY for user-approved HaruQ, not retroactively for original.
for score in [0.4,0.5,0.8,1.0,1.5,2.0,2.5]:
    f2_original=strict_tickets(ordered,score,"F2")
    assert f2_original["exacta"]["points"]==0 and f2_original["trifecta"]["points"]==0
    result=apply_user_policy(strict_tickets(ordered,score,"F2",allow_f2=True))
    assert result["approved_bets"]["exacta"]["total_yen"]==(2000 if score>=1 else 0)
    assert result["approved_bets"]["trifecta"]["total_yen"]==(3000 if score>=2 else 0)
    assert result["approved_total_yen"]==(2000 if score>=1 else 0)+(3000 if score>=2 else 0)
# All-L1/unknown class entries: fail closed as girls, before touching SQLite.
from scripts.flagship_live_predict import record_entries
for title in ("ガールズ一般","ガールズ決勝"):
    entries=[{"rider_id":str(i), "bike_number":i, "class":None} for i in range(1,8)]
    try:
        record_entries("202610231301",{"entries":entries,"grade":"F2","race_name":title})
        raise AssertionError("Girls was incorrectly treated as F2 purchasable")
    except RuntimeError as e:
        assert "ガールズ" in str(e)
print("HARUQ_GIRLS_FAIL_CLOSED_PASSED")

print("HARUQ_HIGH_ONLY_V3_TESTS_PASSED")
print("HARUQ_USER_POLICY_TESTS_PASSED")
print("FLAGSHIP_FIXED_LOGIC_TESTS_PASSED")
