#!/usr/bin/env python3
"""Independent arithmetic audit of the original March21 released CSV ledger.

NO model inference, no DB. Proves report arithmetic and payout transaction
consistency only. Grades/years, concentration and missingness are included.
A distinct full model-ledger reconstruction audits F2 inclusive performance.
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

def summary(rows):
    verified=[r for r in rows if r["payout_verified"].strip().lower()=="true"]
    inv=sum(int(x["invested_yen"]) for x in verified)
    ret=sum(int(x["returned_yen"]) for x in verified)
    return {"attempts":len(rows),"verified":len(verified),"unverifiable":len(rows)-len(verified),
            "hits":sum(int(x["hit"]) for x in verified),
            "stake_yen":inv,"return_yen":ret,"profit_yen":ret-inv,
            "roi":round(100*ret/inv,3) if inv else None}

def check(rows):
    keys=set()
    for r in rows:
        k=(r["version"],r["race_id"])
        if k in keys: raise RuntimeError("duplicate race/strategy:"+str(k))
        keys.add(k)
        all_tickets=r["tickets"].split(";")
        if len(all_tickets)!=int(r["ticket_count"]) or len(all_tickets)!=len(set(all_tickets)):
            raise RuntimeError("duplicate or incorrect ticket count:"+str(k))
        actual=r["winning_combo"]
        hit=int(actual in all_tickets)
        if r["payout_verified"].strip().lower()=="true":
            if hit!=int(r["hit"]): raise RuntimeError("hit mismatch:"+str(k))
            if int(r["invested_yen"])!=int(r["stake_each_yen"])*len(all_tickets):
                raise RuntimeError("stake mismatch:"+str(k))
            if not hit and int(r["returned_yen"])!=0:raise RuntimeError("losing ticket paid:"+str(k))
            if hit and int(r["returned_yen"])<int(r["stake_each_yen"]):
                raise RuntimeError("unreasonably small paid hit:"+str(k))
            # Winning payout in archived ledger must imply an integer netkeirin
            # displayed yen return at 100-yen unit stake.
            if hit and (int(r["returned_yen"])*100)%int(r["stake_each_yen"])!=0:
                raise RuntimeError("settlement fractional yen:"+str(k))
        elif int(r["invested_yen"])!=0 or int(r["returned_yen"])!=0:
            raise RuntimeError("unverifiable paid row:"+str(k))
    return {"checked":len(rows),"all_outcome_stake_invariants_valid":True}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--ledger",type=Path,required=True)
    a.add_argument("--report",type=Path,required=True)
    a.add_argument("--output",type=Path,default=Path("haruq_ledger_quick_audit.json"))
    args=a.parse_args()
    rows=list(csv.DictReader(args.ledger.open(encoding="utf-8")))
    rep=json.loads(args.report.read_text(encoding="utf-8"))
    tests=check(rows)
    grouped=defaultdict(list)
    for r in rows:
        if r["version"] not in {"2026-03-21_exacta","2026-03-21_trifecta","2026-03-22_rule_on_0321_model"}:
            raise RuntimeError("Unrecognized version "+r["version"])
        grouped[(r["version"],r["tier"])].append(r)
    for version in {"2026-03-21_exacta","2026-03-21_trifecta","2026-03-22_rule_on_0321_model"}:
        ver_rows=[r for r in rows if r["version"]==version]
        s=summary(ver_rows)
        target=rep["holdout_betting"][version]
        for our,key in (("attempts","attempted_races"),("verified","payout_verified_races"),
                        ("hits","hits"),("stake_yen","invested_yen"),("return_yen","returned_yen")):
            if s[our]!=target[key]:
                raise RuntimeError(f"Original report mismatch: {version}/{our}: {s[our]} != {target[key]}")
        for tier in {k[1] for k in grouped if k[0]==version}:
            ss=summary(grouped[(version,tier)])
            if version in ("2026-03-21_exacta","2026-03-21_trifecta"):
                recorded=target["tiers"][tier]
                if ss["stake_yen"]!=recorded["invested_yen"] or ss["return_yen"]!=recorded["returned_yen"]:
                    raise RuntimeError(f"tier gross totals mismatch {version} {tier}")

    tiers={}
    grade_year={}
    high=[r for r in rows if r["version"]=="2026-03-21_trifecta" and r["tier"]=="S3-HIGH"]
    for (v,t),rs in sorted(grouped.items()):
        tiers[v+"/"+t]=summary(rs)
        for r in rs:
            key="/".join((v,t,r["grade"],r["date"][:4]))
            grade_year.setdefault(key,[]).append(r)
    withtop=sorted(high,key=lambda x:int(x["returned_yen"]),reverse=True)
    stress={str(n):summary(withtop[n:]) for n in [0,1,3,5,10,20,50,100]}
    p=Path(args.output)
    audit={
      "status":"LEDGER_ARITHMETIC_VERIFIED_NOT_LIVE",
      "verification":tests,
      "original_report_totals_match":True,
      "record_count":len(rows),
      "grades":sorted(set(r["grade"] for r in rows)),
      "f2_appears_in_original_selected_ledger":any(r["grade"]=="F2" for r in rows),
      "source":{"train":"2023-01-01 to 2024-12-31",
                "validation":"2025-01-01 to 2025-06-30",
                "holdout":"2025-07-01 to 2026-09-30"},
      "original_report":rep["holdout_betting"],
      "tiers":tiers,
      "grade_year":{k:summary(v) for k,v in sorted(grade_year.items())},
      "high_without_largest_winners":stress,
      "caveats":[
        "Arithmetic and archived ledger checked, not independent re-generation of AI picks.",
        "Archived release ledger intentionally contains no F2 wagers; the 253.414% cannot represent F2-inclusive strategy.",
        "Historical OOS return is simulated at eventual payout, not real-time bets.",
        "Choosing tiers based on this holdout introduces selection bias.",
        "Live feature parity with offline batch feature generation is not proven."
      ]
    }
    p.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HARUQ_QUICK_AUDIT_COMPLETE "+json.dumps({
      "rows":len(rows),"f2_in_archive":audit["f2_appears_in_original_selected_ledger"],
      "S3-HIGH":tiers.get("2026-03-21_trifecta/S3-HIGH"),
      "S3-MED+":tiers.get("2026-03-21_trifecta/S3-MED+"),
      "S3-MED":tiers.get("2026-03-21_trifecta/S3-MED"),
      "top5_removed":stress["5"]},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
