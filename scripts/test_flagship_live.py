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
print("FLAGSHIP_FIXED_LOGIC_TESTS_PASSED")
