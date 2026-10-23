#!/usr/bin/env python3
"""Fixed-base-AI vs market holdout: NO retraining, NO purchased model, NO tuning.

Venue-code identity inferred on 2025 Q3 ONLY by independent official podium IDs.
Economic/economic-forecast test on 2025 Q4 and 2026 Q1 ONLY.
"""
import csv
import gzip
import json
import math
import random
from collections import defaultdict, Counter
from datetime import date, timedelta
from pathlib import Path

OUT = Path("market_edge_outputs")
OUT.mkdir(exist_ok=True)
SEED = 20261023

def find_one(pattern):
    items = list(Path(".").glob(pattern))
    if len(items) != 1:
        raise RuntimeError(f"Expected 1 file: {pattern}; got {items}")
    return items[0]

def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def gz_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def decode_id(rid):
    s = str(rid)
    if len(s) != 16 or not s.isdigit():
        return None
    try:
        d = date(int(s[2:6]),int(s[6:8]),int(s[8:10]))
        offset=int(s[10:12])
        num=int(s[12:16])
        if not 1 <= offset <= 7 or not 1 <= num <= 12:
            return None
        return d + timedelta(days=offset-1), s[:2], num
    except ValueError:
        return None

def base_key(rid):
    s=str(rid)
    if len(s) != 12 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]),int(s[4:6]),int(s[6:8])),s[8:10],int(s[10:12])
    except ValueError:
        return None

def ci_diff(xs):
    if not xs:return [None,None]
    rng=random.Random(SEED)
    n=len(xs)
    draws=[]
    for _ in range(2500):
        draws.append(sum(xs[rng.randrange(n)] for _ in range(n))/n)
    draws.sort()
    return [round(draws[int(.025*len(draws))],5),
            round(draws[int(.975*len(draws))],5)]

def headline(rows):
    n=len(rows)
    if not n: return {"races":0}
    a=sum(x["ai_hit"] for x in rows)
    m=sum(x["market_hit"] for x in rows)
    p=sum(x["implied_ai_win_p"] for x in rows)
    market_p=sum(x["implied_market_win_p"] for x in rows)
    var=sum(x["implied_ai_win_p"]*(1-x["implied_ai_win_p"]) for x in rows)
    market_var=sum(x["implied_market_win_p"]*(1-x["implied_market_win_p"]) for x in rows)
    d=[x["ai_hit"]-x["market_hit"] for x in rows]
    diffs=[x["ai_hit"]-x["implied_ai_win_p"] for x in rows]
    discordance=Counter((x["ai_hit"],x["market_hit"]) for x in rows if x["ai_hit"]!=x["market_hit"])
    return {
      "races":n,"ai_hits":a,"ai_rate":round(a/n,5),
      "market_hits":m,"market_rate":round(m/n,5),
      "ai_market_top1_disagreements":sum(x["ai_bike"]!=x["market_bike"] for x in rows),
      "ai_only_hits":discordance[(1,0)],"market_only_hits":discordance[(0,1)],
      "ai_vs_market_accuracy_pp":round((a-m)*100/n,3),
      "ai_vs_market_accuracy_diff95_pp":[round(z*100,3) for z in ci_diff(d)],
      "market_implied_ai_top1_hits":round(p,2),
      "market_implied_market_top1_hits":round(market_p,2),
      "market_observed_minus_implied_rate_pp":round(100*(m-market_p)/n,3),
      "market_selection_conditional_z":round((m-market_p)/math.sqrt(market_var),3) if market_var else None,
      "ai_hits_minus_implied":round(a-p,2),
      "ai_selection_conditional_z":round((a-p)/math.sqrt(var),3) if var else None,
      "observed_minus_implied_rate_pp":round(100*(a-p)/n,3),
      "observed_minus_implied_diff95_pp":[round(z*100,3) for z in ci_diff(diffs)],
    }

def odds_metrics(rows, label):
    n=len(rows)
    if not n:return {"name":label,"races":0}
    out={"name":label,"races":n}
    for which in ("ai","market"):
        one_hits=0;one_return=0.0; six_hits=0;six_return=0.0
        for x in rows:
            book=x["book"]
            c=x[which+"_exact"]
            if c in book and c==x["actual_exact"]:
                one_hits+=1;one_return+=100*book[c]
            for ticket in x[which+"_six"]:
                if ticket==x["actual_exact"] and ticket in book:
                    six_hits+=1;six_return+=100*book[ticket]
        out[which+"_single"]={"tickets":n,"hits":one_hits,
            "investment_yen":n*100,"payout_yen":round(one_return),
            "roi_percent":round(one_return/n,2)}
        out[which+"_six"]={"tickets":n*6,"hits":six_hits,
            "investment_yen":n*600,"payout_yen":round(six_return),
            "roi_percent":round(six_return/n/6,2)}
    return out

def main():
    cap=read_csv(find_one("base_cap/**/race_predictions.csv"))
    pred=read_csv(find_one("base_predictions/**/holdout_existing_predictions.csv"))
    capby={r["race_id"]:r for r in cap}
    joined={}
    for r in pred:
        rid=r["race_id"]
        if rid not in capby:continue
        a=capby[rid]
        if int(r["top1_hit"])!=int(a["new_hit"]):
            raise AssertionError(f"Prediction mismatch {rid}")
        k=base_key(rid)
        if k is not None:
            joined[rid]={"cap":a,"pred":r,"date":k[0],"site":k[1],"num":k[2]}
    print("PREDICTIONS",len(pred),"JOINED",len(joined),flush=True)
    assert len(joined)==3973, len(joined)
    base_lookup=defaultdict(list)
    for rid,x in joined.items():
        base_lookup[(x["date"],x["num"],(int(x["pred"]["actual1"]),int(x["pred"]["actual2"]),int(x["pred"]["actual3"])))].append(x["site"])

    # Identity inference on a disjoint pre-evaluation quarter.
    votes=defaultdict(Counter)
    results_count=Counter()
    for year in (2025,2026):
        p=Path(f"market/results_{year}.jsonl.gz")
        for r in gz_rows(p):
            info=decode_id(r["race_id"])
            if info is None:continue
            d,market_site,num=info
            if d.year!=year:continue
            if year!=2025 or not date(2025,7,1)<=d<date(2025,10,1):continue
            actual=tuple(int(x) for x in r["finish_order"])
            actual_sites=base_lookup.get((d,num,actual),[])
            if len(set(actual_sites))==1:
                votes[market_site][actual_sites[0]]+=1
            results_count[market_site]+=1
    proposed=[]
    for market_site,c in votes.items():
        name, count=c.most_common(1)[0]
        total=sum(c.values())
        if count>=5 and count/total >= .75:
            proposed.append((count/total,count,market_site,name,total))
    proposed.sort(reverse=True)
    map_site={};taken=set()
    for frac,count,k,v,total in proposed:
        if v not in taken:
            map_site[k]=v;taken.add(v)
    print("VENUE IDENTITY MAPPING",len(map_site),json.dumps(map_site),flush=True)
    print("VENUE MATCH VOTES",json.dumps({k:dict(v) for k,v in votes.items()}),flush=True)

    # Load 2025Q4 / 2026Q1 final 3-way markets. No selection based on result.
    odds_index={}
    book_counts=Counter()
    for year in (2025,2026):
        for r in gz_rows(Path(f"market/odds_{year}.jsonl.gz")):
            info=decode_id(r["race_id"])
            if info is None:book_counts["invalid_id"]+=1;continue
            d,market_site,num=info
            if not date(2025,10,1)<=d<date(2026,4,1):continue
            if market_site not in map_site:book_counts["unmapped_site"]+=1;continue
            key=(d,map_site[market_site],num)
            if key in odds_index:
                book_counts["multiple_market_ids_for_event"]+=1
                del odds_index[key]
                continue
            odds_index[key]=r
            book_counts["unambiguous_market_books"]+=1
    print("MARKET BOOK COUNTS",json.dumps(dict(book_counts)),flush=True)

    all_rows=[];book_unusable=Counter();not_found=Counter()
    for rid,x in joined.items():
        if not date(2025,10,1)<=x["date"]<date(2026,4,1):continue
        m=odds_index.get((x["date"],x["site"],x["num"]))
        if not m:
            not_found[x["site"]]+=1;continue
        raw_book=m.get("odds",{}).get("3連単",{})
        try:
            book={k:float(v) for k,v in raw_book.items() if float(v)>1.0}
            inv=defaultdict(float);total=0.0; num_prices=0
            for combo,odds in book.items():
                parts=combo.split("-")
                if len(parts)!=3 or len(set(parts))!=3:
                    continue
                car=int(parts[0]);p=1/odds
                inv[car]+=p;total+=p;num_prices+=1
            if not total or num_prices<50:
                book_unusable["not_full"]+=1;continue
            probs={b:v/total for b,v in inv.items()}
            p=x["pred"];c=x["cap"]
            ai_bike=int(p["pred1"])
            if ai_bike not in probs:
                book_unusable["ai_bike_absent"]+=1;continue
            market_bike=max(sorted(probs),key=lambda b:probs[b])
            actual_first=int(p["actual1"])
            actual_exact=f'{p["actual1"]}-{p["actual2"]}-{p["actual3"]}'
            ai_exact=f'{p["pred1"]}-{p["pred2"]}-{p["pred3"]}'
            # 6 fixed formation combos for the top-1 and the next three
            # Only top three available in prior artifact; do not invent 4th.
            # Therefore this is a two-ticket top3-swapped diagnostic, NOT 6pt.
            ai_pair=[f'{p["pred1"]}-{p["pred2"]}-{p["pred3"]}',
                     f'{p["pred1"]}-{p["pred3"]}-{p["pred2"]}']
            # Six least-price market combinations: deliberately not used for ROI
            market_sorted=sorted(book,key=lambda k:(book[k],k))
            market_exact=market_sorted[0]
            market_six=market_sorted[:6]
            rec={"race_id":rid,"date":str(x["date"]),"period":str(x["date"])[:7],
                 "score_gap":float(c["score_gap"]),
                 "ai_bike":ai_bike,"market_bike":market_bike,
                 "market_p_ai":probs[ai_bike], "implied_ai_win_p":probs[ai_bike],
                 "implied_market_win_p":probs[market_bike],
                 "ai_hit":int(ai_bike==actual_first),
                 "market_hit":int(market_bike==actual_first),
                 "ai_exact":ai_exact,"market_exact":market_exact,
                 "ai_pair":ai_pair,"market_six":market_six,
                 "actual_exact":actual_exact,"book":book}
            all_rows.append(rec)
        except (ValueError,TypeError,KeyError) as e:
            book_unusable[type(e).__name__]+=1
    print("VALID JOINED EVENT BOOKS",len(all_rows), "ERRORS",dict(book_unusable),flush=True)
    slices={"all_matched":all_rows,"confidence_ge_1_5":[x for x in all_rows if x["score_gap"]>=1.5],
      "confidence_lt_1_5":[x for x in all_rows if x["score_gap"]<1.5],
      "2025Q4":[x for x in all_rows if x["date"]<"2026-01-01"],
      "2026Q1":[x for x in all_rows if x["date"]>="2026-01-01"],
      "ai_disagrees_with_market":[x for x in all_rows if x["ai_bike"]!=x["market_bike"]]}
    bins={"market_ai_win_prob_under_15pct":[x for x in all_rows if x["market_p_ai"]<.15],
      "market_ai_win_prob_15_25pct":[x for x in all_rows if .15<=x["market_p_ai"]<.25],
      "market_ai_win_prob_25_40pct":[x for x in all_rows if .25<=x["market_p_ai"]<.40],
      "market_ai_win_prob_40pct_plus":[x for x in all_rows if x["market_p_ai"]>=.40]}
    slices.update(bins)
    scores={k:headline(v) for k,v in slices.items()}
    def return_stats(rows):
        if not rows:return {"n":0}
        pay_ai=pay_market=pay_pair=0.0
        hits_ai=hits_market=hits_pair=0
        for x in rows:
            b=x["book"];win=x["actual_exact"]
            if x["ai_exact"]==win:
                pay_ai+=b[win];hits_ai+=1
            if x["market_exact"]==win:
                pay_market+=b[win];hits_market+=1
            if win in x["ai_pair"]:
                pay_pair+=b[win];hits_pair+=1
        n=len(rows)
        return {"n":n,"AI_top3_order_1pt":{"hits":hits_ai,"stake_yen":100*n,
                            "returned_yen":round(pay_ai*100),
                            "roi_percent":round(pay_ai*100/n,2)},
                "market_lowest_odds_trifecta_1pt":{"hits":hits_market,
                            "stake_yen":100*n,"returned_yen":round(pay_market*100),
                            "roi_percent":round(pay_market*100/n,2)},
                "AI_top1_top2top3_both_orders_2pt":{"hits":hits_pair,
                            "stake_yen":200*n,"returned_yen":round(pay_pair*100),
                            "roi_percent":round(pay_pair*50/n,2)}}
    rois={k:return_stats(v) for k,v in slices.items()}
    for k in ("2025-10","2025-11","2025-12","2026-01","2026-02","2026-03"):
        month_rows=[x for x in all_rows if x["period"]==k]
        rois[k]=return_stats(month_rows)
    report={"hypothesis":"Fixed LambdaRank has no incremental price-adjusted information beyond normalized trifecta market",
       "scope":"2025Q4 through 2026Q1 strictly disjoint from venue identity inference Q3",
       "data_provenance":{"predictions":"previous unmodified base-ai-capability-evidence and base-ai-44-improvement-evidence",
           "odds":"keirin-market-analysis-/data/odds_only: final 3連単 full books",
           "venue_identity":"Q3 2025 podium matches exclusively; no outcome used for test-period identity",
           "no_retraining":True,"no_threshold_tuning":True},
       "holdout_races_source":len(joined),"venue_map":map_site,
       "venue_mapping_votes":{k:dict(v) for k,v in votes.items()},
       "market_books":dict(book_counts),"unmatched_by_base_venue":dict(not_found),
       "unusable_books":dict(book_unusable),
       "match_counts":{"matched":len(all_rows),"confidence_high":len(slices["confidence_ge_1_5"])},
       "metrics":scores,"retrospective_3ren_traces":rois,
       "method":"Market 1st probability = normalize sum(1/3連単 odds) across every 3連単 with given first rider. Compare AI selected first rider realized win rate to this price-implied expectation; compare AI vs market top1; 3連単 diagnostics on fixed tickets.",
       "limitations":[
         "This is a price-adjusted calibration diagnostic, not proof of causal market superiority.",
         "Final odds are not proven available at actual decision time; ex-ante ROI not established.",
         "The original currently-enabled strategy is 2車単4点, which CANNOT be ROI-tested from this archive (only 3連単 and 3連複).",
         "Prior predictions are pure offline LambdaRank output, not complete live line-bonus scores.",
         "Site mapping is learned on a separate quarter and restricts coverage; no result from evaluation interval is used in ID matching.",
         "Model and original database were not modified.",
         "No confidence threshold or stake was optimized on the test interval.",
       ]}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"joined_test_rows.csv").open("w",encoding="utf-8",newline="") as f:
        cols=["race_id","date","period","score_gap","ai_bike","market_bike",
              "market_p_ai","implied_market_win_p","ai_hit","market_hit","ai_exact",
              "market_exact","actual_exact"]
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in all_rows:writer.writerow({k:r[k] for k in cols})
    print("TEST HEADLINES",json.dumps({k:scores[k] for k in
        ["all_matched","confidence_ge_1_5","confidence_lt_1_5","2025Q4","2026Q1","ai_disagrees_with_market"]},ensure_ascii=False),flush=True)
    print("ROI HEADLINES",json.dumps({k:rois[k] for k in
        ["all_matched","confidence_ge_1_5","confidence_lt_1_5"]},ensure_ascii=False),flush=True)
    print("ROI QUARTERLY",json.dumps({k:rois[k] for k in ("2025Q4","2026Q1")},ensure_ascii=False),flush=True)
    print("ROI MONTHLY",json.dumps({k:rois[k] for k in ("2025-10","2025-11","2025-12","2026-01","2026-02","2026-03")},ensure_ascii=False),flush=True)
    print("COMPLETE",flush=True)

if __name__=="__main__":
    main()
